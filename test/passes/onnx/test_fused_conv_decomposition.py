# Copyright (c) 2026 Contributors to the Olive project.
# Licensed under the MIT License.
"""FusedConv semantics and opt-in conversion boundaries."""

import unittest

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from olive.passes.onnx.fused_conv import decompose_fused_conv


class TestFusedConvDecomposition(unittest.TestCase):
    def model(self, activation="Relu", residual=False, extra=None):
        attrs = {"kernel_shape": [1, 1], **(extra or {})}
        if activation is not None:
            attrs["activation"] = activation
        inputs = ["input", "weights", "bias"]
        initializers = [
            numpy_helper.from_array(np.array([[[[0.75]]]], dtype=np.float32), "weights"),
            numpy_helper.from_array(np.array([0.125], dtype=np.float32), "bias"),
        ]
        if residual:
            inputs.append("z")
            initializers.append(numpy_helper.from_array(np.ones((1, 1, 2, 2), dtype=np.float32), "z"))
        node = helper.make_node("FusedConv", inputs, ["output"], name="fused", domain="com.microsoft", **attrs)
        graph = helper.make_graph(
            [node],
            "fused",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 2, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 2, 2])],
            initializers,
        )
        return helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)], ir_version=10
        )

    def run_model(self, model):
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        session.disable_fallback()
        value = np.array([[[[-2, -1], [0.5, 1]]]], dtype=np.float32)
        return session.run(None, {"input": value})[0]

    def compare(self, model):
        before = model.SerializeToString()
        converted = decompose_fused_conv(model)
        assert before == model.SerializeToString()
        onnx.checker.check_model(converted)
        np.testing.assert_allclose(self.run_model(converted), self.run_model(model), rtol=0, atol=1e-6)
        return converted

    def test_relu_matches_actual_fused_kernel(self):
        converted = self.compare(self.model())
        assert [n.op_type for n in converted.graph.node] == ["Conv", "Relu"]

    def test_identity_matches_actual_fused_kernel(self):
        converted = self.compare(self.model(activation=None))
        assert [n.op_type for n in converted.graph.node] == ["Conv"]

    def test_unsupported_activation_rejected(self):
        with pytest.raises(ValueError, match=r"activation.*not supported"):
            decompose_fused_conv(self.model(activation="LeakyRelu"))

    def test_activation_parameters_rejected(self):
        with pytest.raises(ValueError, match=r"activation.*not supported"):
            decompose_fused_conv(self.model(extra={"activation_params": [0.1]}))

    def test_residual_input_rejected(self):
        with pytest.raises(ValueError, match="residual Z input"):
            decompose_fused_conv(self.model(residual=True))

    def test_unknown_attributes_rejected(self):
        with pytest.raises(ValueError, match="unsupported attributes"):
            decompose_fused_conv(self.model(extra={"future_semantics": 1}))

    def test_intermediate_names_do_not_collide(self):
        model = self.model()
        model.graph.initializer.append(numpy_helper.from_array(np.array(1, dtype=np.float32), "output_before_relu"))
        converted = self.compare(model)
        assert converted.graph.node[0].output[0] != "output_before_relu"

    def test_nested_graph_preserves_output(self):
        model = self.model()
        branch_node = onnx.NodeProto()
        branch_node.CopyFrom(model.graph.node[0])
        branch_node.output[0] = "branch_output"
        branch = helper.make_graph(
            [branch_node],
            "branch",
            [],
            [helper.make_tensor_value_info("branch_output", TensorProto.FLOAT, [1, 1, 2, 2])],
        )
        del model.graph.node[:]
        model.graph.node.append(
            helper.make_node("If", ["condition"], ["output"], then_branch=branch, else_branch=branch)
        )
        model.graph.initializer.append(numpy_helper.from_array(np.array(True), "condition"))
        converted = self.compare(model)
        for attribute in converted.graph.node[0].attribute:
            assert all(node.op_type != "FusedConv" for node in attribute.g.node)

    def test_nested_capture_keeps_ancestor_value(self):
        model = self.model()
        fused = onnx.NodeProto()
        fused.CopyFrom(model.graph.node[0])
        fused.output[0] = "branch_output"
        captured = helper.make_graph(
            [helper.make_node("Identity", ["branch_output_before_relu"], ["captured_inner"])],
            "captured",
            [],
            [helper.make_tensor_value_info("captured_inner", TensorProto.FLOAT, [1, 1, 2, 2])],
        )
        branch = helper.make_graph(
            [
                fused,
                helper.make_node("If", ["condition"], ["captured_output"], then_branch=captured, else_branch=captured),
                helper.make_node("Add", ["branch_output", "captured_output"], ["branch_result"]),
            ],
            "outer_branch",
            [],
            [helper.make_tensor_value_info("branch_result", TensorProto.FLOAT, [1, 1, 2, 2])],
        )
        del model.graph.node[:]
        model.graph.node.append(
            helper.make_node("If", ["condition"], ["output"], then_branch=branch, else_branch=branch)
        )
        model.graph.initializer.extend(
            [
                numpy_helper.from_array(np.array(True), "condition"),
                numpy_helper.from_array(np.full((1, 1, 2, 2), 10, dtype=np.float32), "branch_output_before_relu"),
            ]
        )
        self.compare(model)

    def test_nested_function_fused_conv_is_rejected(self):
        model = self.model()
        node = onnx.NodeProto()
        node.CopyFrom(model.graph.node[0])
        branch = helper.make_graph(
            [node],
            "function_branch",
            [],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 2, 2])],
        )
        function = helper.make_function(
            "local",
            "ContainsNestedFused",
            ["input", "weights", "bias", "condition"],
            ["output"],
            [helper.make_node("If", ["condition"], ["output"], then_branch=branch, else_branch=branch)],
            opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
        )
        model.functions.append(function)
        with pytest.raises(ValueError, match="local function bodies"):
            decompose_fused_conv(model)
