# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Provider routing regressions; real CUDA placement is validated separately."""
import copy
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from olive.common import ort_inference

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"
PLUGIN = "QNNExecutionProvider"


def ep(name, device_type):
    return SimpleNamespace(ep_name=name, device=SimpleNamespace(type=device_type))


class TestBuiltinAndDeviceProviders(unittest.TestCase):
    def setUp(self):
        self.options = mock.Mock()
        self.session = mock.Mock()
        self.session.get_providers.return_value = [CUDA, CPU]
        self.session._enable_fallback = True
        policies = {name: name for name in [
            "DEFAULT", "PREFER_CPU", "PREFER_NPU", "PREFER_GPU",
            "MAX_PERFORMANCE", "MAX_EFFICIENCY", "MIN_OVERALL_POWER"
        ]}
        self.ort = SimpleNamespace(
            SessionOptions=mock.Mock(return_value=self.options),
            InferenceSession=mock.Mock(return_value=self.session),
            get_available_providers=mock.Mock(return_value=[CUDA, CPU]),
            get_all_providers=mock.Mock(return_value=[PLUGIN, CUDA, CPU]),
            get_ep_devices=mock.Mock(return_value=[ep(CPU, "cpu")]),
            OrtHardwareDeviceType=SimpleNamespace(CPU="cpu", GPU="gpu", NPU="npu"),
            OrtExecutionProviderDevicePolicy=SimpleNamespace(**policies),
        )
        modules = mock.patch.dict(sys.modules, {"onnxruntime": self.ort})
        modules.start()
        self.addCleanup(modules.stop)
        windows = mock.patch.object(ort_inference.platform, "system", return_value="Windows")
        windows.start()
        self.addCleanup(windows.stop)

    def settings(self, providers=None, options=None, **extra):
        return {
            "execution_provider": providers if providers is not None else [CUDA],
            "provider_options": options if options is not None else [{"use_tf32": 0}],
            **extra,
        }

    def test_availability_unions_builtins_and_registered_devices_in_priority_order(self):
        self.ort.get_ep_devices.return_value.append(ep(PLUGIN, "gpu"))
        self.assertEqual(ort_inference.get_ort_available_providers(), [PLUGIN, CUDA, CPU])

    def test_builtin_legacy_path_preserves_complete_order_and_options(self):
        settings = self.settings(
            [CUDA, CPU], [{"use_tf32": 0, "device_id": 2}, {"arena_extend_strategy": "kSameAsRequested"}]
        )
        before = copy.deepcopy(settings)
        returned = ort_inference.get_ort_inference_session("model.onnx", "gpu", settings, device_id=7)
        self.assertIs(returned, self.session)
        kwargs = self.ort.InferenceSession.call_args.kwargs
        self.assertEqual(kwargs["providers"], [CUDA, CPU])
        self.assertEqual(kwargs["provider_options"], [
            {"use_tf32": "0", "device_id": "7"}, {"arena_extend_strategy": "kSameAsRequested"}
        ])
        self.options.add_provider_for_devices.assert_not_called()
        self.session.disable_fallback.assert_called_once_with()
        self.assertEqual(settings, before)

    def test_builtin_strict_path_retains_native_cpu_node_gate(self):
        ort_inference.get_ort_inference_session(
            "model.onnx", "gpu", self.settings(allow_cpu_fallback=False)
        )
        self.options.add_session_config_entry.assert_called_once_with("session.disable_cpu_ep_fallback", "1")
        self.assertEqual(self.ort.InferenceSession.call_args.kwargs["providers"], [CUDA])

    def test_registered_builtin_keeps_device_api_and_options(self):
        device = ep(CUDA, "gpu")
        self.ort.get_ep_devices.return_value += [device, ep(CUDA, "gpu")]
        ort_inference.get_ort_inference_session("model.onnx", "gpu", self.settings())
        self.options.add_provider_for_devices.assert_called_once_with([device], {"use_tf32": "0"})
        self.assertNotIn("providers", self.ort.InferenceSession.call_args.kwargs)

    def test_registered_plugin_keeps_device_api(self):
        device = ep(PLUGIN, "gpu")
        self.ort.get_available_providers.return_value = [CPU]
        self.ort.get_ep_devices.return_value.append(device)
        self.session.get_providers.return_value = [PLUGIN, CPU]
        ort_inference.get_ort_inference_session(
            "model.onnx", "gpu", self.settings([PLUGIN], [{"backend_path": "QnnHtp.dll"}])
        )
        self.options.add_provider_for_devices.assert_called_once_with([device], {"backend_path": "QnnHtp.dll"})
        self.assertNotIn("providers", self.ort.InferenceSession.call_args.kwargs)

    def test_builtin_only_and_device_only_combination_is_explicitly_rejected(self):
        self.ort.get_ep_devices.return_value.append(ep(PLUGIN, "gpu"))
        with self.assertRaisesRegex(ValueError, "Cannot combine built-in-only providers"):
            ort_inference.get_ort_inference_session(
                "model.onnx", "gpu", self.settings([CUDA, PLUGIN], [{}, {}])
            )
        self.ort.InferenceSession.assert_not_called()
        self.options.add_provider_for_devices.assert_not_called()

    def test_policy_cannot_silently_replace_requested_builtin(self):
        with self.assertRaisesRegex(ValueError, "provider_selection_policy requires registered EP devices"):
            ort_inference.get_ort_inference_session(
                "model.onnx", "gpu", self.settings(provider_selection_policy="prefer_gpu")
            )
        self.ort.InferenceSession.assert_not_called()

    def test_registered_device_policy_is_preserved(self):
        self.ort.get_ep_devices.return_value.append(ep(CUDA, "gpu"))
        ort_inference.get_ort_inference_session(
            "model.onnx", "gpu", self.settings(provider_selection_policy="prefer_gpu")
        )
        self.options.set_provider_selection_policy.assert_called_once_with("PREFER_GPU")

    def test_constructor_fallback_to_cpu_is_not_accepted_as_cuda(self):
        self.session.get_providers.return_value = [CPU]
        with self.assertRaisesRegex(ort_inference.OrtSessionFallbackError, "CUDAExecutionProvider"):
            ort_inference.get_ort_inference_session("model.onnx", "gpu", self.settings())
        self.assertEqual(self.ort.InferenceSession.call_args.kwargs["providers"], [CUDA])

    def test_linux_keeps_legacy_constructor(self):
        with mock.patch.object(ort_inference.platform, "system", return_value="Linux"):
            ort_inference.get_ort_inference_session("model.onnx", "gpu", self.settings())
        self.ort.get_ep_devices.assert_not_called()
        self.assertEqual(self.ort.InferenceSession.call_args.kwargs["providers"], [CUDA])

    def test_older_ort_without_device_api_keeps_legacy_constructor(self):
        del self.ort.get_ep_devices
        ort_inference.get_ort_inference_session("model.onnx", "gpu", self.settings())
        self.assertEqual(self.ort.InferenceSession.call_args.kwargs["providers"], [CUDA])
