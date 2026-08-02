import importlib
import sys
import types
from types import SimpleNamespace


def test_batched_server_sets_cuda_device_before_loading_model(monkeypatch):
    fake_base_framework_module = types.ModuleType("starVLA.model.framework.base_framework")
    fake_base_framework_module.baseframework = object
    monkeypatch.setitem(sys.modules, "starVLA.model.framework.base_framework", fake_base_framework_module)
    sys.modules.pop("deployment.model_server.server_policy_batched", None)

    from deployment.model_server import server_policy_batched
    server_policy_batched = importlib.reload(server_policy_batched)

    events = []

    class FakeCuda:
        @staticmethod
        def set_device(device):
            events.append(("set_device", str(device)))

    class FakeTorch:
        cuda = FakeCuda()
        bfloat16 = "bf16"

        @staticmethod
        def device(device):
            events.append(("device", str(device)))
            return device

    class FakeModel:
        def to(self, target):
            events.append(("to", str(target)))
            return self

        def eval(self):
            events.append(("eval", None))
            return self

    class FakeBaseFramework:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            events.append(("from_pretrained", None))
            return FakeModel()

    class FakeServer:
        def __init__(self, **kwargs):
            events.append(("server_init", kwargs["port"]))

        def serve_forever(self):
            events.append(("serve_forever", None))

    monkeypatch.setattr(server_policy_batched, "torch", FakeTorch)
    monkeypatch.setattr(server_policy_batched, "baseframework", FakeBaseFramework)
    monkeypatch.setattr(server_policy_batched, "BatchedWebsocketPolicyServer", FakeServer)

    args = SimpleNamespace(
        ckpt_path="/tmp/model.pt",
        cuda="3",
        use_bf16=True,
        port=12345,
        max_batch_size=4,
        batch_timeout_ms=20,
    )

    server_policy_batched.main(args)

    assert events.index(("set_device", "cuda:3")) < events.index(("from_pretrained", None))
