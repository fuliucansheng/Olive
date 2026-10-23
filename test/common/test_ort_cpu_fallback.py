# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Copyright (c) 2026 Contributors to the Olive project.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""Fallback controls tested with actual ORT sessions where available."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from olive.common.ort_inference import OrtSessionFallbackError, get_ort_available_providers, get_ort_inference_session


class TestCpuFallbackPolicy(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="olive fallback ")
        self.addCleanup(self.temporary.cleanup)
        self.model = Path(self.temporary.name) / "add.onnx"
        graph = helper.make_graph(
            [helper.make_node("Add", ["input", "bias"], ["output"])],
            "fallback",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])],
            [numpy_helper.from_array(np.array([1, 2], dtype=np.float32), "bias")],
        )
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10), self.model)

    def session(self, **settings):
        return get_ort_inference_session(self.model, "cpu", settings)

    def test_explicit_cpu_runs_when_allowed(self):
        session = self.session(execution_provider=["CPUExecutionProvider"], allow_cpu_fallback=True)
        actual = session.run(None, {"input": np.array([3, 4], dtype=np.float32)})[0]
        np.testing.assert_array_equal(actual, np.array([4, 6], dtype=np.float32))

    def test_missing_provider_is_rejected_before_session_creation(self):
        provider = "MissingExecutionProvider"
        assert provider not in get_ort_available_providers()
        with pytest.raises(OrtSessionFallbackError, match="unavailable"):
            self.session(execution_provider=[provider], allow_cpu_fallback=False)

    def test_cpu_provider_conflicts_with_strict_non_cpu_target(self):
        with pytest.raises(ValueError, match="explicit non-CPU"):
            self.session(execution_provider=["CPUExecutionProvider"], allow_cpu_fallback=False)

    def test_strict_policy_requires_explicit_target(self):
        with pytest.raises(ValueError, match="explicit non-CPU"):
            self.session(allow_cpu_fallback=False)

    def test_string_false_is_not_a_boolean_policy(self):
        with pytest.raises(ValueError, match="must be a bool"):
            self.session(execution_provider=["CPUExecutionProvider"], allow_cpu_fallback="false")

    def optimization_pass(self, opt_level):
        from olive.hardware import DEFAULT_GPU_CUDA_ACCELERATOR
        from olive.passes.olive_pass import create_pass_from_dict
        from olive.passes.onnx.transformer_optimization import OrtTransformersOptimization

        return create_pass_from_dict(
            OrtTransformersOptimization,
            {"model_type": "bert", "use_gpu": True, "opt_level": opt_level},
            disable_search=True,
            accelerator_spec=DEFAULT_GPU_CUDA_ACCELERATOR,
        )

    def test_unavailable_gpu_optimizer_rejected(self):
        import onnxruntime as ort

        from olive.model import ONNXModelHandler

        if "CUDAExecutionProvider" in ort.get_available_providers():
            self.skipTest("This control requires an ORT package without CUDA EP")
        with pytest.raises(ValueError, match=r"Requested optimization provider.*unavailable"):
            self.optimization_pass(1).run(
                ONNXModelHandler(model_path=str(self.model)),
                str(Path(self.temporary.name) / "gpu-output"),
            )

    def test_offline_symbolic_optimization_does_not_require_gpu(self):
        from olive.model import ONNXModelHandler

        output = self.optimization_pass(0).run(
            ONNXModelHandler(model_path=str(self.model)),
            str(Path(self.temporary.name) / "offline-output"),
        )
        onnx.checker.check_model(onnx.load(output.model_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
