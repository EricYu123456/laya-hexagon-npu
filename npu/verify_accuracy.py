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
    'profiling_file_path': '/home/ubuntu/laya-service/npu/prof_acc.csv'
}
opts.add_provider_for_devices(devices, provider_options)

onnx_qdq = '/home/ubuntu/laya-service/npu/modernbert_22l_qdq.onnx'
npu_sess = ort.InferenceSession(onnx_qdq, sess_options=opts)
print("NPU Session created with disable_cpu_ep_fallback=1.")

# NPU Decision forward function
def npu_predict(agent, state, questions):
    ids = list(questions.keys())
    items = []
    max_len = 128
    head_max_len = 64

    for qid in ids:
        q = agent._to_internal(questions[qid])
        seq, markers = build_sequence(agent.tok, state, q, max_len, head_max_len)
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})

    # Prepare inputs for batch of items
    # Note: ModernBertClean was exported with batch size 1 and fixed seq_len 128
    # If multiple questions are asked, we can run them through NPU
    all_answers = {}
    total_tokens = 0

    for r, qid in enumerate(ids):
        item = items[r]
        seq = item["ids"]
        markers = item["markers"]
        qtype = item["qtype"]
        q = agent._to_internal(questions[qid])
        
        inp_ids = np.full((1, 128), agent.tok.pad_token_id, dtype=np.int64)
        inp_ids[0, :len(seq)] = seq
        total_tokens += len(seq)
        
        # CPU: Embedding lookup
        with torch.no_grad():
            embeds = tok_embed(torch.tensor(inp_ids)).numpy()
            att_mask_1d = torch.zeros((1, 128), dtype=torch.long)
            att_mask_1d[0, :len(seq)] = 1
            g_mask, s_mask = encoder._update_attention_mask(att_mask_1d, False)
            g_np = np.clip(g_mask.numpy().astype(np.float32), -10000.0, 0.0)
            s_np = np.clip(s_mask.numpy().astype(np.float32), -10000.0, 0.0)

        # NPU: 22-layer backbone
        h_last = npu_sess.run(None, {
            'inputs_embeds': embeds,
            'attn_mask': g_np,
            'sliding_mask': s_np,
        })[0]

        # CPU: Decision head
        h = torch.from_numpy(h_last)
        with torch.no_grad():
            qtype_t = torch.tensor([qtype])
            h = h + agent.model.type_emb(qtype_t)[:, None, :]
            if agent.model.head is not None:
                pad = ~att_mask_1d.bool()
                for layer in agent.model.head.layers:
                    h = layer(h, src_key_padding_mask=pad)
            
            marker_pos = torch.tensor([markers + [-1] * (head_max_len - len(markers))])
            marker_mask = marker_pos >= 0
            idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
            m = torch.gather(h, 1, idx)
            logits = agent.model.scorer(m).squeeze(-1).float()
            logits = logits.masked_fill(~marker_mask, -1e4)

            p = torch.softmax(logits.detach(), -1)
            k = marker_mask.sum(-1).clamp(min=2).float()
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

            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                all_answers[qid] = {
                    "type": "choice",
                    "choice": keys[int(p_final.argmax())],
                    "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p_final)},
                    "confidence": conf_score,
                    "action": ext,
                }
            elif q["t"] == "score":
                exp_score = float((np.arange(k_len) * p_final).sum())
                all_answers[qid] = {
                    "type": "score",
                    "score": round(exp_score, 4),
                    "legend": {str(i): c for i, c in enumerate(q["crit"])},
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

print("=== Running Accuracy Probes ===")
probes = json.loads(Path('/home/ubuntu/laya-service/accuracy-probes.json').read_text())
matches = 0
for idx, p in enumerate(probes):
    qdef = {
        "test_q": {
            "type": "noul",
            "instructions": p["instructions"]
        }
    }
    cpu_res = cpu_agent.predict(p["state"], qdef)
    npu_res = npu_predict(cpu_agent, p["state"], qdef)
    
    cpu_noul = cpu_res["answers"]["test_q"]["noul"]
    npu_noul = npu_res["answers"]["test_q"]["noul"]
    diff = abs(cpu_noul - npu_noul)
    print(f"Probe {idx+1}/{len(probes)}: CPU noul={cpu_noul:.4f}, NPU noul={npu_noul:.4f}, diff={diff:.4f}")
    if diff < 0.15:
        matches += 1

print(f"Probes passed: {matches}/{len(probes)}")

print("=== Running Example Request ===")
example = json.loads(Path('/home/ubuntu/laya-service/example.json').read_text())
cpu_ex = cpu_agent.predict(example["state"], example["questions"])
npu_ex = npu_predict(cpu_agent, example["state"], example["questions"])

print("\n--- CPU Example Result ---")
print(json.dumps(cpu_ex, indent=2, ensure_ascii=False))
print("\n--- NPU Example Result ---")
print(json.dumps(npu_ex, indent=2, ensure_ascii=False))
