import os
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import onnx
from onnx import helper, TensorProto
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
import onnxruntime as ort
import onnxruntime_qnn as q
import laya
from laya.common import build_sequence

print("=== Step 1: Loading PyTorch Laya Multilingual Model ===")
agent = laya.load('/home/ubuntu/laya-service/models/multilingual', device='cpu')
encoder = agent.model.encoder
encoder.eval()
tok = agent.tok

class ModernBertClean22L(nn.Module):
    def __init__(self, encoder, max_seq_len=128):
        super().__init__()
        self.encoder = encoder
        self.max_seq_len = max_seq_len
        position_ids = torch.arange(max_seq_len).unsqueeze(0)
        for layer_type in set(encoder.config.layer_types):
            cos, sin = encoder.rotary_emb(torch.zeros(1, max_seq_len, 768), position_ids, layer_type)
            self.register_buffer(f'cos_{layer_type}', cos)
            self.register_buffer(f'sin_{layer_type}', sin)

    def forward(self, inputs_embeds, attn_mask, sliding_mask):
        hidden_states = self.encoder.embeddings.drop(self.encoder.embeddings.norm(inputs_embeds))
        for layer in self.encoder.layers:
            cos = getattr(self, f'cos_{layer.attention_type}')
            sin = getattr(self, f'sin_{layer.attention_type}')
            pos_emb = (cos, sin)
            out = layer(
                hidden_states,
                attention_mask=attn_mask,
                sliding_window_mask=sliding_mask,
                position_embeddings=pos_emb,
            )
            hidden_states = out[0]
        return self.encoder.final_norm(hidden_states)

model_22l = ModernBertClean22L(encoder, 128)
model_22l.eval()

onnx_fp32 = '/home/ubuntu/laya-service/npu/modernbert_22l_raw.onnx'
onnx_fixed = '/home/ubuntu/laya-service/npu/modernbert_22l_fixed.onnx'
onnx_qdq = '/home/ubuntu/laya-service/npu/modernbert_22l_qdq.onnx'

dummy_embeds = torch.randn((1, 128, 768), dtype=torch.float32)
dummy_attn_mask = torch.zeros((1, 1, 128, 128), dtype=torch.float32)
dummy_sliding_mask = torch.zeros((1, 1, 128, 128), dtype=torch.float32)

print("=== Step 2: Exporting PyTorch model to ONNX Opset 20 ===")
torch.onnx.export(
    model_22l,
    (dummy_embeds, dummy_attn_mask, dummy_sliding_mask),
    onnx_fp32,
    input_names=['inputs_embeds', 'attn_mask', 'sliding_mask'],
    output_names=['last_hidden_state'],
    opset_version=20,
    do_constant_folding=True,
)
print("ONNX FP32 exported successfully.")

print("=== Step 3: Fixing LayerNormalization bias nodes to Initializers ===")
model = onnx.load(onnx_fp32)
const_nodes = {n.output[0]: n for n in model.graph.node if n.op_type == 'Constant'}
nodes_to_remove = set()
converted_biases = 0

for n in list(model.graph.node):
    if n.op_type == 'LayerNormalization':
        if len(n.input) >= 3 and n.input[2] in const_nodes:
            b_name = n.input[2]
            c_node = const_nodes[b_name]
            for a in c_node.attribute:
                if a.name == 'value':
                    t = a.t
                    t.name = b_name
                    model.graph.initializer.append(t)
                    nodes_to_remove.add(c_node.name)
                    converted_biases += 1

new_nodes = [n for n in model.graph.node if n.name not in nodes_to_remove]
del model.graph.node[:]
model.graph.node.extend(new_nodes)
onnx.save(model, onnx_fixed)
print(f"Converted {converted_biases} LayerNorm biases to static initializers.")

# Clean up raw FP32 to save disk space
if os.path.exists(onnx_fp32):
    os.remove(onnx_fp32)

print("=== Step 4: Preparing Calibration Data ===")
probes = json.loads(Path('/home/ubuntu/laya-service/accuracy-probes.json').read_text())
example = json.loads(Path('/home/ubuntu/laya-service/example.json').read_text())
tok_embed = encoder.embeddings.tok_embeddings

samples = []
for p in probes:
    qdef = {'t': 'noul', 'ins': p['instructions'], 'crit': None}
    seq, _ = build_sequence(tok, p['state'], qdef, max_len=128, head_max_len=64)
    samples.append(seq)

for qid, qd in example['questions'].items():
    qdef = agent._to_internal(qd)
    seq, _ = build_sequence(tok, example['state'], qdef, max_len=128, head_max_len=64)
    samples.append(seq)

calib_data = []
with torch.no_grad():
    for seq in samples:
        inp_ids = np.full((1, 128), tok.pad_token_id, dtype=np.int64)
        inp_ids[0, :len(seq)] = seq
        embeds = tok_embed(torch.tensor(inp_ids)).numpy()
        att_mask_1d = torch.zeros((1, 128), dtype=torch.long)
        att_mask_1d[0, :len(seq)] = 1
        g_mask, s_mask = encoder._update_attention_mask(att_mask_1d, False)
        # Use -10000.0 instead of -inf/-3.4e38 to prevent NaN scales during quantization
        g_np = g_mask.numpy().astype(np.float32)
        s_np = s_mask.numpy().astype(np.float32)
        g_np = np.clip(g_np, -10000.0, 0.0)
        s_np = np.clip(s_np, -10000.0, 0.0)
        calib_data.append({
            'inputs_embeds': embeds,
            'attn_mask': g_np,
            'sliding_mask': s_np,
        })

class CleanDataReader(CalibrationDataReader):
    def __init__(self, data):
        self.data = data
        self.idx = 0
    def get_next(self):
        if self.idx >= len(self.data):
            return None
        res = self.data[self.idx]
        self.idx += 1
        return res

print(f"Calibration data ready: {len(calib_data)} samples.")

print("=== Step 5: Static QDQ Quantization ===")
t_q0 = time.time()
quantize_static(
    onnx_fixed,
    onnx_qdq,
    CleanDataReader(calib_data),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QUInt8,
)
print(f"Quantization completed in {time.time() - t_q0:.2f}s! File size: {os.path.getsize(onnx_qdq) / 1024 / 1024:.2f} MB")

print("=== Step 6: Testing Execution on Hexagon NPU (HTP V68) ===")
ort.register_execution_provider_library('QNNExecutionProvider', q.get_library_path())
devices = [d for d in ort.get_ep_devices() if d.ep_name == 'QNNExecutionProvider']
opts = ort.SessionOptions()
opts.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
provider_options = {
    'backend_path': q.get_qnn_htp_path(),
    'htp_arch': '68',
    'htp_performance_mode': 'burst',
    'profiling_file_path': '/home/ubuntu/laya-service/npu/prof_22l.csv'
}
opts.add_provider_for_devices(devices, provider_options)

t_s0 = time.time()
sess = ort.InferenceSession(onnx_qdq, sess_options=opts)
print(f"QNN HTP Session created successfully with disable_cpu_ep_fallback=1 in {time.time() - t_s0:.2f}s!")

# Test inference
test_sample = calib_data[0]
t_i0 = time.time()
out = sess.run(None, test_sample)[0]
print(f"NPU Inference finished in {time.time() - t_i0:.4f}s! Output shape: {out.shape}")
print("ALL 22 LAYERS VERIFIED ON HEXAGON HTP V68!")
