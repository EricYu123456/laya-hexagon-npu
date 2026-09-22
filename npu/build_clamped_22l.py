import os
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
import onnxruntime as ort
import onnxruntime_qnn as q
import laya
from laya.common import build_sequence, render_options, QTYPES, temp_bucket, confidence_from_probs

MASK_PENALTY = -15.0

print("=== Step 1: Loading PyTorch Model and Wrapping with Clamped MLP ===")
agent = laya.load('/home/ubuntu/laya-service/models/multilingual', device='cpu')
encoder = agent.model.encoder
encoder.eval()
tok = agent.tok

class ClampedMLP(nn.Module):
    def __init__(self, orig_mlp, clamp_val=50.0):
        super().__init__()
        self.Wi = orig_mlp.Wi
        self.act = orig_mlp.act
        self.drop = orig_mlp.drop
        self.Wo = orig_mlp.Wo
        self.clamp_val = clamp_val

    def forward(self, hidden_states):
        x = self.Wi(hidden_states)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        act_out = self.act(x1) * x2
        clamped = torch.clamp(act_out, -self.clamp_val, self.clamp_val)
        return self.Wo(self.drop(clamped))

for layer in encoder.layers:
    layer.mlp = ClampedMLP(layer.mlp, 50.0)

class ModernBertClamped22L(nn.Module):
    def __init__(self, encoder, max_seq_len=64):
        super().__init__()
        self.encoder = encoder
        self.max_seq_len = max_seq_len
        position_ids = torch.arange(max_seq_len).unsqueeze(0)
        for layer_type in set(encoder.config.layer_types):
            cos, sin = encoder.rotary_emb(torch.zeros(1, max_seq_len, 768), position_ids, layer_type)
            self.register_buffer(f'cos_{layer_type}', cos)
            self.register_buffer(f'sin_{layer_type}', sin)

    def forward(self, inputs_embeds, attn_mask, sliding_mask):
        h = self.encoder.embeddings.drop(self.encoder.embeddings.norm(inputs_embeds))
        for layer in self.encoder.layers:
            cos = getattr(self, f'cos_{layer.attention_type}')
            sin = getattr(self, f'sin_{layer.attention_type}')
            pos_emb = (cos, sin)
            out = layer(
                h,
                attention_mask=attn_mask,
                sliding_window_mask=sliding_mask,
                position_embeddings=pos_emb,
            )
            h = out[0]
        return self.encoder.final_norm(h)

model_clamped = ModernBertClamped22L(encoder, 64)
model_clamped.eval()

fp32_path = '/home/ubuntu/laya-service/npu/modernbert_clamped_fp32.onnx'
print("=== Step 2: Exporting to ONNX Opset 20 ===")
dummy_emb = torch.randn(1, 64, 768)
dummy_mask = torch.zeros(1, 1, 64, 64)
torch.onnx.export(
    model_clamped,
    (dummy_emb, dummy_mask, dummy_mask),
    fp32_path,
    input_names=['inputs_embeds', 'attn_mask', 'sliding_mask'],
    output_names=['last_hidden_state'],
    opset_version=20,
    do_constant_folding=True
)
print("ONNX FP32 exported.")

print("=== Step 3: Converting LayerNorm Biases to Initializers ===")
model = onnx.load(fp32_path)
const_nodes = {node.output[0]: node for node in model.graph.node if node.op_type == 'Constant'}
to_remove = set()
converted = 0
for node in list(model.graph.node):
    if node.op_type == 'LayerNormalization' and len(node.input) >= 3 and node.input[2] in const_nodes:
        b_name = node.input[2]
        c = const_nodes[b_name]
        for a in c.attribute:
            if a.name == 'value':
                t = a.t
                t.name = b_name
                model.graph.initializer.append(t)
                to_remove.add(c.name)
                converted += 1

keep_nodes = [node for node in model.graph.node if node.name not in to_remove]
del model.graph.node[:]
model.graph.node.extend(keep_nodes)
onnx.save(model, fp32_path)
print(f"Converted {converted} LayerNorm biases to Initializers.")

print("=== Step 4: Preparing Calibration Data (Mask Penalty = -15.0) ===")
probes = json.loads(Path('/home/ubuntu/laya-service/accuracy-probes.json').read_text())
example = json.loads(Path('/home/ubuntu/laya-service/example.json').read_text())

samples = []
for p in probes:
    seq, _ = build_sequence(tok, p['state'], {'t': 'noul', 'ins': p['instructions'], 'crit': None}, max_len=64, head_max_len=64)
    samples.append(seq)
for qid, qd in example['questions'].items():
    seq, _ = build_sequence(tok, example['state'], agent._to_internal(qd), max_len=64, head_max_len=64)
    samples.append(seq)

calib_data = []
tok_embed = encoder.embeddings.tok_embeddings
with torch.no_grad():
    for s in samples:
        padded = s + [agent.tok.pad_token_id] * (64 - len(s))
        emb = tok_embed(torch.tensor([padded])).numpy()
        mask = np.zeros((1, 1, 64, 64), dtype=np.float32)
        mask[0, 0, :, len(s):] = MASK_PENALTY
        calib_data.append({
            'inputs_embeds': emb,
            'attn_mask': mask,
            'sliding_mask': mask,
        })

class CalibReader(CalibrationDataReader):
    def __init__(self, d): self.d = d; self.i = 0
    def get_next(self):
        if self.i >= len(self.d): return None
        res = self.d[self.i]; self.i += 1
        return res

qdq_path = '/home/ubuntu/laya-service/npu/modernbert_clamped_qdq.onnx'
print("=== Step 5: Static QDQ Quantization ===")
quantize_static(
    fp32_path,
    qdq_path,
    CalibReader(calib_data),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QUInt8,
)
print("Quantization complete!")

print("=== Step 6: Initializing Qualcomm Hexagon NPU (HTP V68) ===")
ort.register_execution_provider_library('QNNExecutionProvider', q.get_library_path())
devices = [d for d in ort.get_ep_devices() if d.ep_name == 'QNNExecutionProvider']
opts = ort.SessionOptions()
opts.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
provider_options = {
    'backend_path': q.get_qnn_htp_path(),
    'htp_arch': '68',
    'htp_performance_mode': 'burst',
    'profiling_file_path': '/home/ubuntu/laya-service/npu/prof_final.csv'
}
opts.add_provider_for_devices(devices, provider_options)

t0 = time.time()
npu_sess = ort.InferenceSession(qdq_path, sess_options=opts)
print(f"Hexagon HTP Session created in {time.time() - t0:.2f}s with disable_cpu_ep_fallback=1!")

def npu_predict(agent, state, questions):
    ids = list(questions.keys())
    all_answers = {}
    total_tokens = 0

    for qid in ids:
        qdef = agent._to_internal(questions[qid])
        seq, markers = build_sequence(tok, state, qdef, max_len=64, head_max_len=64)
        total_tokens += len(seq)
        
        padded = seq + [tok.pad_token_id] * (64 - len(seq))
        with torch.no_grad():
            embeds = tok_embed(torch.tensor([padded])).numpy()
            mask = np.zeros((1, 1, 64, 64), dtype=np.float32)
            mask[0, 0, :, len(seq):] = MASK_PENALTY

        h_last = npu_sess.run(None, {
            'inputs_embeds': embeds,
            'attn_mask': mask,
            'sliding_mask': mask,
        })[0]

        h = torch.from_numpy(h_last)
        qtype = QTYPES[qdef['t']]
        with torch.no_grad():
            qtype_t = torch.tensor([qtype])
            h = h + agent.model.type_emb(qtype_t)[:, None, :]
            
            # Run decision head
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

print("\n=== Step 7: Running Accuracy Probes ===")
matches = 0
for idx, p in enumerate(probes):
    qdef = {"test_q": {"type": "noul", "instructions": p["instructions"]}}
    cpu_res = agent.predict(p["state"], qdef)
    npu_res = npu_predict(agent, p["state"], qdef)
    cpu_noul = cpu_res["answers"]["test_q"]["noul"]
    npu_noul = npu_res["answers"]["test_q"]["noul"]
    diff = abs(cpu_noul - npu_noul)
    align = (cpu_noul < 0.5 and npu_noul < 0.5) or (cpu_noul >= 0.5 and npu_noul >= 0.5)
    print(f"Probe {idx+1}/{len(probes)}: CPU={cpu_noul:.4f}, NPU={npu_noul:.4f}, diff={diff:.4f}, align={align}")
    if align:
        matches += 1
print(f"Probes Passed: {matches}/{len(probes)}")

print("\n=== Step 8: Running Example Request ===")
cpu_ex = agent.predict(example["state"], example["questions"])
npu_ex = npu_predict(agent, example["state"], example["questions"])

print("\n--- CPU Result ---")
print(json.dumps(cpu_ex["answers"], indent=2, ensure_ascii=False))

print("\n--- NPU Result ---")
print(json.dumps(npu_ex["answers"], indent=2, ensure_ascii=False))
