"""Hexagon NPU (HTP V68) accelerated runtime for Laya decision models."""
import os
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import onnxruntime as ort
import onnxruntime_qnn as q
from transformers import AutoTokenizer
from laya.common import (
    QTYPES,
    TEMP_MAX,
    TEMP_MIN,
    build_sequence,
    clamp_temperature,
    confidence_from_probs,
    render_options,
    temp_bucket,
)

MASK_PENALTY = -15.0
MAX_SEQ_LEN = 64
HEAD_MAX_LEN = 64

class NPUAgent:
    """High-performance Laya Agent with ModernBERT backbone offloaded to Qualcomm Hexagon HTP."""

    def __init__(self, model_id_or_path: str = "models/multilingual"):
        self.model_dir = str(Path(model_id_or_path).resolve())
        cfg_path = os.path.join(self.model_dir, "rl_agent_config.json")
        with open(cfg_path) as f:
            self.cfg = json.load(f)

        # Tokenizer
        tok_dir = os.path.join(self.model_dir, "tokenizer")
        self.tok = AutoTokenizer.from_pretrained(tok_dir if os.path.exists(tok_dir) else self.cfg.get("encoder"))

        # Temperatures
        self.temperature_raw = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options_raw = self.cfg.get("temperature_by_options", {})
        self.temperature = [clamp_temperature(t) for t in self.temperature_raw]
        self.temperature_by_options = {k: clamp_temperature(v) for k, v in self.temperature_by_options_raw.items()}

        # Load Decision Heads from safetensors
        from safetensors.torch import load_file
        weights = load_file(os.path.join(self.model_dir, "model.safetensors"))

        # Build CPU modules for embeddings and decision heads
        d = 768
        self.tok_embed = torch.nn.Embedding(self.cfg.get("vocab_size", 256000), d, padding_idx=0)
        self.tok_embed.weight.data.copy_(weights["encoder.embeddings.tok_embeddings.weight"])
        self.tok_embed.eval()

        self.type_emb = torch.nn.Embedding(3, d)
        self.type_emb.weight.data.copy_(weights["type_emb.weight"])
        self.type_emb.eval()

        # Head layers
        head_layers = self.cfg.get("head_layers", 2)
        nhead = max(1, d // 64)
        layer = torch.nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.head = torch.nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        if self.head is not None:
            head_sd = {}
            for k, v in weights.items():
                if k.startswith("head."):
                    head_sd[k[len("head."):]] = v
            self.head.load_state_dict(head_sd)
            self.head.eval()

        # Scorer
        self.scorer = torch.nn.Sequential(
            torch.nn.LayerNorm(d),
            torch.nn.Linear(d, d),
            torch.nn.GELU(),
            torch.nn.Linear(d, 1)
        )
        scorer_sd = {}
        for k, v in weights.items():
            if k.startswith("scorer."):
                scorer_sd[k[len("scorer."):]] = v
        self.scorer.load_state_dict(scorer_sd)
        self.scorer.eval()

        # Act head
        n_act = len(self.cfg.get("act_costs", {})) + 1
        self.act_head = torch.nn.Sequential(
            torch.nn.Linear(d + 4, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, n_act)
        )
        act_sd = {}
        for k, v in weights.items():
            if k.startswith("act_head."):
                act_sd[k[len("act_head."):]] = v
        self.act_head.load_state_dict(act_sd)
        self.act_head.eval()

        # Initialize QNN Hexagon NPU Session
        npu_model_path = os.environ.get(
            "LAYA_NPU_MODEL",
            "/home/ubuntu/laya-service/npu/modernbert_clamped_qdq.onnx"
        )
        ort.register_execution_provider_library("QNNExecutionProvider", q.get_library_path())
        devices = [dev for dev in ort.get_ep_devices() if dev.ep_name == "QNNExecutionProvider"]
        opts = ort.SessionOptions()
        # Strictly enforce NO fallback to CPU EP
        opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        provider_options = {
            "backend_path": q.get_qnn_htp_path(),
            "htp_arch": "68",
            "htp_performance_mode": "burst",
            "profiling_file_path": "/home/ubuntu/laya-service/npu/qnn_service_prof.csv"
        }
        opts.add_provider_for_devices(devices, provider_options)
        self.npu_sess = ort.InferenceSession(npu_model_path, sess_options=opts)

    @staticmethod
    def _to_internal(qdef: Dict) -> Dict:
        t = qdef["type"]
        crit = qdef.get("criteria")
        if t == "choice" and isinstance(crit, list):
            crit = {c: None for c in crit}
        ins = qdef["instructions"]
        if not isinstance(ins, str):
            ins = json.dumps(ins)
        return {"t": t, "ins": ins, "crit": crit}

    @torch.no_grad()
    def predict(self, state: Union[str, dict, list], questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        ids = list(questions.keys())
        all_answers = {}
        total_tokens = 0

        for qid in ids:
            qdef = self._to_internal(questions[qid])
            seq, markers = build_sequence(self.tok, state, qdef, max_len=MAX_SEQ_LEN, head_max_len=HEAD_MAX_LEN)
            if len(markers) != len(render_options(qdef)):
                raise ValueError("question %r options exceed head_max_len=%d" % (qid, HEAD_MAX_LEN))
            total_tokens += len(seq)

            padded = seq + [self.tok.pad_token_id] * (MAX_SEQ_LEN - len(seq))
            embeds = self.tok_embed(torch.tensor([padded])).numpy()
            mask = np.zeros((1, 1, MAX_SEQ_LEN, MAX_SEQ_LEN), dtype=np.float32)
            mask[0, 0, :, len(seq):] = MASK_PENALTY

            # NPU Forward Pass
            h_last = self.npu_sess.run(None, {
                "inputs_embeds": embeds,
                "attn_mask": mask,
                "sliding_mask": mask,
            })[0]

            h = torch.from_numpy(h_last)
            qtype = QTYPES[qdef["t"]]
            qtype_t = torch.tensor([qtype])
            h = h + self.type_emb(qtype_t)[:, None, :]

            # Head Attention
            if self.head is not None:
                pad = torch.zeros((1, MAX_SEQ_LEN), dtype=torch.bool)
                pad[0, len(seq):] = True
                for layer in self.head.layers:
                    h = layer(h, src_key_padding_mask=pad)

            # Scorer on markers
            m_pos = torch.tensor([markers])
            idx_t = m_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
            m = torch.gather(h, 1, idx_t)
            logits = self.scorer(m).squeeze(-1).float()

            # Act head
            p = torch.softmax(logits.detach(), -1)
            k = torch.tensor([float(len(markers))])
            ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
            if p.size(-1) >= 2:
                top2 = p.topk(2, -1).values
            else:
                top1 = p.topk(1, -1).values
                top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
            feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
            pooled = h[:, 0].float()
            act_logits = self.act_head(torch.cat([pooled, feats], -1))

            logits_np = logits.numpy()
            act_np = torch.softmax(act_logits.float(), -1).numpy()

            k_len = len(markers)
            t_scale = self.temperature_by_options.get(temp_bucket(qtype, k_len), self.temperature[qtype])
            z = logits_np[0, :k_len] / t_scale
            p_final = np.exp(z - z.max())
            p_final = p_final / p_final.sum()

            conf_score = round(confidence_from_probs(p_final, k_len), 4)
            ext = {"act_probability": round(float(act_np[0, 0]), 4)}

            if qdef["t"] == "choice":
                keys = list(qdef["crit"].keys())
                all_answers[qid] = {
                    "type": "choice",
                    "choice": keys[int(p_final.argmax())],
                    "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p_final)},
                    "confidence": conf_score,
                    "action": ext,
                }
            elif qdef["t"] == "score":
                exp_score = float((np.arange(k_len) * p_final).sum())
                all_answers[qid] = {
                    "type": "score",
                    "score": round(exp_score, 4),
                    "legend": {str(i): c for i, c in enumerate(qdef["crit"])},
                    "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p_final)},
                    "confidence": conf_score,
                    "action": ext,
                }
            else:
                all_answers[qid] = {
                    "type": "noul",
                    "noul": round(float(p_final[1]), 4),
                    "confidence": round(max(float(p_final[1]), 1.0 - float(p_final[1])), 4),
                    "action": ext,
                }

        return {
            "model": "laya-rl-agent",
            "answers": all_answers,
            "usage": {"input_tokens": total_tokens, "output_tokens": 0},
        }

def load(model_id_or_path: str = "models/multilingual", device: Optional[str] = "npu") -> NPUAgent:
    return NPUAgent(model_id_or_path)
