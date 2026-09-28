"""Bound CPU calibration memory without discarding calibration samples.

ORT 1.30's calibration session hardcodes its session options. This scoped adapter
disables the CPU arena and memory pattern for that session only; graph execution
and MinMax range merging remain ORT's implementation. Use CalibStridedMinMax=1
with a range-capable reader to merge every example before releasing its outputs.
"""
from contextlib import contextmanager


@contextmanager
def low_memory_calibration(threads):
    import onnxruntime as ort
    from onnxruntime.quantization.calibrate import CalibraterBase

    original = CalibraterBase.create_inference_session

    def create_session(calibrator):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        calibrator.infer_session = ort.InferenceSession(
            calibrator.augmented_model_path, sess_options=options,
            providers=calibrator.execution_providers,
        )

    CalibraterBase.create_inference_session = create_session
    try:
        yield
    finally:
        CalibraterBase.create_inference_session = original
