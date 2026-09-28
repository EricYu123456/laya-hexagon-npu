import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantType
from onnxruntime.quantization.calibrate import CalibraterBase

from npu.calibration_memory import low_memory_calibration


class Reader(CalibrationDataReader):
    def __init__(self):
        self.data = [{"x": np.array([[a, b]], dtype=np.float32)}
                     for a, b in [(-123, 42), (4, 5), (-6, 99)]]
        self.set_range(0, len(self.data))

    def get_next(self):
        return next(self.iterator, None)

    def __len__(self):
        return len(self.data)

    def set_range(self, start_index, end_index):
        self.iterator = iter(self.data[start_index:end_index])


class CalibrationMemoryTests(unittest.TestCase):
    def test_all_sample_extrema_match_regular_calibration(self):
        graph = helper.make_graph(
            [helper.make_node("Add", ["x", "bias"], ["y"])], "probe",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
            [numpy_helper.from_array(np.array([1, -2], dtype=np.float32), "bias")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.onnx"
            regular = Path(directory) / "regular.onnx"
            bounded = Path(directory) / "bounded.onnx"
            onnx.save(model, source)
            quantize_static(source, regular, Reader(), activation_type=QuantType.QUInt16,
                            weight_type=QuantType.QUInt8)
            with low_memory_calibration(1):
                quantize_static(source, bounded, Reader(), activation_type=QuantType.QUInt16,
                                weight_type=QuantType.QUInt8,
                                extra_options={"CalibStridedMinMax": 1})
            left = {t.name: numpy_helper.to_array(t) for t in onnx.load(regular).graph.initializer}
            right = {t.name: numpy_helper.to_array(t) for t in onnx.load(bounded).graph.initializer}
            self.assertEqual(left.keys(), right.keys())
            for name in left:
                np.testing.assert_array_equal(left[name], right[name])
            # The first sample contains the minimum: dropping early samples would
            # give a much narrower input quantization range and fail this check.
            self.assertGreater(float(right["x_scale"]), 200 / 65535)

    def test_session_factory_restored_after_exception(self):
        original = CalibraterBase.create_inference_session
        with self.assertRaisesRegex(RuntimeError, "probe"):
            with low_memory_calibration(1):
                self.assertIsNot(CalibraterBase.create_inference_session, original)
                raise RuntimeError("probe")
        self.assertIs(CalibraterBase.create_inference_session, original)


if __name__ == "__main__":
    unittest.main()
