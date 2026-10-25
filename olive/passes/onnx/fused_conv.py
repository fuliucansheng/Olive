# Copyright (c) 2026 Contributors to the Olive project.
# Licensed under the MIT License.
"""Explicit FusedConv decomposition for targets needing standard ONNX operators."""

from copy import deepcopy

import onnx


def decompose_fused_conv(model: onnx.ModelProto) -> onnx.ModelProto:
    """Convert identity/Relu FusedConv without discarding unsupported semantics.

    This is an opt-in compatibility step, not a claim that every target or
    mixed-precision graph needs decomposition. Unsupported activations, residual
    inputs, attributes and function bodies fail before quantization.
    """
    result = deepcopy(model)

    def child_graphs(node):
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield attribute.g
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                yield from attribute.graphs

    def all_nodes(nodes):
        for node in nodes:
            yield node
            for graph in child_graphs(node):
                yield from all_nodes(graph.node)

    for function in result.functions:
        if any(node.domain == "com.microsoft" and node.op_type == "FusedConv" for node in all_nodes(function.node)):
            raise ValueError("FusedConv decomposition in local function bodies is not supported.")

    # Subgraphs may capture values from any ancestor. Reserve names before any
    # conversion so a new value cannot shadow an existing lexical binding.
    used = set()

    def reserve_graph(graph):
        used.update(value.name for value in [*graph.input, *graph.output, *graph.value_info, *graph.initializer])
        for initializer in graph.sparse_initializer:
            used.update((initializer.values.name, initializer.indices.name))
        for node in graph.node:
            used.update(value for value in [node.name, *node.input, *node.output] if value)
            for child in child_graphs(node):
                reserve_graph(child)

    reserve_graph(result.graph)
    for function in result.functions:
        used.update([*function.input, *function.output])
        used.update(value.name for value in function.value_info)
        for node in all_nodes(function.node):
            used.update(value for value in [node.name, *node.input, *node.output] if value)
            for child in child_graphs(node):
                reserve_graph(child)

    def unique(stem):
        name = stem
        index = 0
        while name in used:
            index += 1
            name = f"{stem}_{index}"
        used.add(name)
        return name

    conv_attributes = {"auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"}

    def convert(graph):
        replacement = []
        for node in graph.node:
            for child in child_graphs(node):
                convert(child)
            if node.domain != "com.microsoft" or node.op_type != "FusedConv":
                replacement.append(node)
                continue
            if len(node.output) != 1 or not 2 <= len(node.input) <= 4:
                raise ValueError(f"FusedConv {node.name!r} has an unsupported input/output signature.")
            if len(node.input) == 4 and node.input[3]:
                raise ValueError(f"FusedConv {node.name!r} has a residual Z input; decomposition does not support it.")
            attrs = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
            unknown = set(attrs) - conv_attributes - {"activation", "activation_params"}
            if unknown:
                raise ValueError(f"FusedConv {node.name!r} has unsupported attributes: {sorted(unknown)}.")
            activation = attrs.get("activation", b"")
            if isinstance(activation, bytes):
                activation = activation.decode("utf-8")
            if activation not in ("", "Relu") or attrs.get("activation_params"):
                raise ValueError(
                    f"FusedConv {node.name!r} activation {activation!r} is not supported by decomposition; "
                    "only identity and Relu without activation_params are implemented."
                )
            conv = deepcopy(node)
            conv.op_type = "Conv"
            conv.domain = ""
            del conv.attribute[:]
            conv.attribute.extend(attribute for attribute in node.attribute if attribute.name in conv_attributes)
            del conv.input[3:]
            if activation == "Relu":
                intermediate = unique(f"{node.output[0]}_before_relu")
                conv.output[0] = intermediate
                replacement.extend(
                    [
                        conv,
                        onnx.helper.make_node(
                            "Relu", [intermediate], list(node.output), name=unique(f"{node.name}_relu")
                        ),
                    ]
                )
            else:
                replacement.append(conv)
        del graph.node[:]
        graph.node.extend(replacement)

    convert(result.graph)
    return result
