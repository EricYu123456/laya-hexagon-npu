import json
import time
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
import onnxruntime_qnn as q
import laya
from laya.common import build_sequence, render_options, QTYPES, temp_bucket, confidence_from_probs

print("=== Loading CPU Agent ===")
cpu_agent = laya.load('/home/ubuntu/laya-service/models/multilingual', device='cpu')
encoder = cpu_agent.model.encoder
tok = cpu_agent.tok
tok_embed = encoder.embeddings.tok_embeddings

print("=== Initializing NPU Session ===")
ort.register_execution_provider_library('QNNExecutionProvider', q.get_library_path())
devices = [d for d in ort.get_ep_devices() if d.ep_name == 'QNNExecutionProvider']
opts = ort.SessionOptions()
opts.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
provider_options = {
    'backend_path': q.get_qnn_htp_path(),
    'htp_arch': '68',
    'htp_performance_mode': 'burst',
}
opts.add_provider_for_devices(devices, provider_options)

onnx_qdq = '/home/ubuntu/laya-service/npu/modernbert_clamped_qdq.onnx'
npu_sess = ort.InferenceSession(onnx_qdq, sess_options=opts)
print("NPU Session created with disable_cpu_ep_fallback=1.")

# NPU Decision forward function
def npu_predict(agent, state, questions):
    ids = list(questions.keys())
    all_answers = {}
    total_tokens = 0

    for qid in ids:
        qdef = agent._to_internal(questions[qid])
        seq, markers = build_sequence(tok, state, qdef, max_len=64, head_max_len=64)
        total_tokens += len(seq)
        
        # Pad up to 64 with sep_token_id
        padded = seq + [tok.sep_token_id] * (64 - len(seq))
        
        # CPU: Embedding lookup
        with torch.no_grad():
            embeds = tok_embed(torch.tensor([padded])).numpy()
            mask_zero = np.zeros((1, 1, 64, 64), dtype=np.float32)

        # NPU: 22-layer backbone (takes ~14 ms)
        h_last = npu_sess.run(None, {
            'inputs_embeds': embeds,
            'attn_mask': mask_zero,
            'sliding_mask': mask_zero,
        })[0]

        # CPU: Decision head
        h = torch.from_numpy(h_last)
        qtype = QTYPES[qdef['t']]
        with torch.no_grad():
            qtype_t = torch.tensor([qtype])
            h = h + agent.model.type_emb(qtype_t)[:, None, :]
            if agent.model.head is not None:
                pad = torch.zeros((1, 64), dtype=torch.bool)
                pad[0, len(seq):] = True
                for layer in agent.model.head.layers:
                    h = layer(h, src_key_padding_mask=pad)
            
            # Scorer on markers
            m_pos = torch.tensor([markers])
            idx_t = m_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
            m = torch.gather(h, 1, idx_t)
            logits = agent.model.scorer(m).squeeze(-1).float()
            
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
            act_logits = agent.model.act_head(torch.cat([pooled, feats], -1))

            logits_np = logits.numpy()
            act_np = torch.softmax(act_logits.float(), -1).numpy()

            k_len = len(markers)
            t_scale = agent.temperature_by_options.get(temp_bucket(qtype, k_len), agent.temperature[qtype])
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
        "model": "laya-rl-agent-npu",
        "answers": all_answers,
        "usage": {"input_tokens": total_tokens, "output_tokens": 0},
    }

print("=== Running Accuracy Probes on Hexagon NPU ===")
probes = json.loads(Path('/home/ubuntu/laya-service/accuracy-probes.json').read_text())
matches = 0
for idx, p in enumerate(probes):
    qdef = {"test_q": {"type": "noul", "instructions": p["instructions"]}}
    cpu_res = cpu_agent.predict(p["state"], qdef)
    npu_res = npu_predict(cpu_agent, p["state"], qdef)
    
    cpu_noul = cpu_res["answers"]["test_q"]["noul"]
    npu_noul = npu_res["answers"]["test_q"]["noul"]
    diff = abs(cpu_noul - npu_noul)
    print(f"Probe {idx+1}/{len(probes)}: CPU={cpu_noul:.4f}, NPU={npu_noul:.4f}, diff={diff:.4f}")
    # Consider passed if classification aligns (both < 0.5 or both > 0.5) or within 0.2
    if (cpu_noul < 0.5 and npu_noul < 0.5) or (cpu_noul >= 0.5 and npu_noul >= 0.5):
        matches += 1

print(f"Probes Classification Match: {matches}/{len(probes)}")

print("\n=== Running Example Request on Hexagon NPU ===")
example = json.loads(Path('/home/ubuntu/laya-service/example.json').read_text())
cpu_ex = cpu_agent.predict(example["state"], example["questions"])
npu_ex = npu_predict(cpu_agent, example["state"], example["questions"])

print("\n--- CPU Result ---")
print(json.dumps(cpu_ex["answers"], indent=2, ensure_ascii=False))

print("\n--- NPU Result ---")
print(json.dumps(npu_ex["answers"], indent=2, ensure_ascii=False))
