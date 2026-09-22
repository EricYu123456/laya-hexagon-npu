import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'legacy/wheel'))
import numpy as np
import onnxruntime as ort
print('ORT',ort.__version__,ort.get_available_providers(),flush=True)
o=ort.SessionOptions()
o.add_session_config_entry('session.disable_cpu_ep_fallback','1')
o.enable_profiling=True
o.profile_file_prefix=str(ROOT/'legacy-profile')
s=ort.InferenceSession(str(ROOT/'smoke.onnx'),sess_options=o,providers=[('QNNExecutionProvider',{'backend_path':str(ROOT/'vendor/usr/lib/libQnnHtp.so'),'htp_arch':'68','htp_performance_mode':'burst'})])
x=np.ones((1,16),dtype=np.float32)
y=s.run(None,{'x':x,'y':x})[0]
np.testing.assert_allclose(y,2,atol=.11)
print('NPU SMOKE PASSED',y.tolist(),s.end_profiling())
