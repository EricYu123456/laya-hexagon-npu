import json, os
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import onnxruntime as ort
import onnxruntime_qnn as q
ROOT=Path(__file__).resolve().parent
init=[nh.from_array(np.array(.1,dtype=np.float32),'s'),nh.from_array(np.array(128,dtype=np.uint8),'z')]
nodes=[]
for name in ['x','y']:
 nodes.extend([h.make_node('QuantizeLinear',[name,'s','z'],[name+'q']),h.make_node('DequantizeLinear',[name+'q','s','z'],[name+'dq'])])
nodes.extend([h.make_node('Add',['xdq','ydq'],['a']),h.make_node('QuantizeLinear',['a','s','z'],['aq']),h.make_node('DequantizeLinear',['aq','s','z'],['out'])])
g=h.make_graph(nodes,'npu_smoke',[h.make_tensor_value_info(n,T.FLOAT,[1,16]) for n in ['x','y']],[h.make_tensor_value_info('out',T.FLOAT,[1,16])],init)
m=h.make_model(g,opset_imports=[h.make_opsetid('',18)],ir_version=10)
onnx.save(m,ROOT/'smoke.onnx')
ort.register_execution_provider_library('QNNExecutionProvider',q.get_library_path())
devices=[d for d in ort.get_ep_devices() if d.ep_name=='QNNExecutionProvider']
print('QNN devices:',len(devices),flush=True)
opts=ort.SessionOptions(); opts.add_session_config_entry('session.disable_cpu_ep_fallback','1'); opts.enable_profiling=True;opts.profile_file_prefix=str(ROOT/'smoke-profile')
opts.add_provider_for_devices(devices,{'backend_path':os.environ.get('QNN_BACKEND',q.get_qnn_htp_path()),'htp_arch':'68','htp_performance_mode':'burst','profiling_level':'basic','profiling_file_path':str(ROOT/'smoke_qnn_profile.csv'),'htp_signed_pd':os.environ.get('QNN_SIGNED_PD','0')})
s=ort.InferenceSession(str(ROOT/'smoke.onnx'),sess_options=opts)
x=np.ones((1,16),dtype=np.float32)
y=s.run(None,{'x':x,'y':x})[0]
np.testing.assert_allclose(y,2,atol=.11)
profile=s.end_profiling()
print('NPU SMOKE PASSED',y.tolist(),profile,flush=True)
