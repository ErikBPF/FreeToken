import torch
from types import SimpleNamespace


def test_down_copy_wait_is_between_gate_and_down_gemv(monkeypatch):
    import freetoken.kernel.gguf as gguf
    import freetoken.moe.fused_q4_k_q5_k as fused

    calls = []
    monkeypatch.setattr(fused, "QWEN_GGUF_FUSION_LEVEL", 3)
    monkeypatch.setattr(
        gguf,
        "ggml_moe_a8_vec",
        lambda *_args: calls.append("gate") or torch.zeros(2, 8),
    )
    monkeypatch.setattr(
        gguf,
        "ggml_moe_a8_vec_silu_reduce",
        lambda *_args: calls.append("down") or torch.zeros(1, 4),
    )

    class Ready:
        def wait(self):
            calls.append("wait")

    fused.fused_experts_gguf_q4_k_q5_k(
        torch.zeros(1, 4),
        torch.zeros(2, 8, 1),
        torch.zeros(2, 4, 1),
        torch.ones(1, 2),
        torch.tensor([[0, 1]], dtype=torch.int32),
        "silu",
        down_ready_event=Ready(),
    )

    assert calls == ["gate", "wait", "down"]


def test_qwen_starts_expert_copy_before_shared_compute(monkeypatch):
    import freetoken.kernel.triton.moe_shared_gate as shared_gate
    import freetoken.moe.fused_q4_k_q5_k as fused
    from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE

    calls = []

    class Experts:
        def prepare_qwen_decode_overlap(self, hidden, logits):
            calls.append("prepare")
            return "prepared"

        def decode_qwen_prepared(self, prepared, hidden, **kwargs):
            assert prepared == "prepared"
            assert kwargs["hidden_q8"] is not None
            calls.append("routed")
            return kwargs["residual"]

    class Shared:
        gate_up_proj = type("GateUp", (), {"forward_q8_0_with_q8": object()})()

        def forward_q8_0_with_q8(self, hidden, *, fused_silu):
            calls.append("shared")
            return hidden * 2, hidden

    moe = object.__new__(Qwen3_5MoE)
    moe.experts = Experts()
    moe.gate = type("Gate", (), {"forward": lambda _self, hidden: calls.append("router") or hidden})()
    moe.shared_expert = Shared()
    moe.shared_expert_gate = type("SharedGate", (), {"weight": torch.ones(1, 4)})()
    monkeypatch.setattr(
        "freetoken.models.qwen3_5_moe.moe.get_global_ctx",
        lambda: SimpleNamespace(batch=SimpleNamespace(is_prefill=False, size=1)),
    )
    monkeypatch.setattr(fused, "QWEN_GGUF_FUSION_LEVEL", 3)
    monkeypatch.setattr(
        shared_gate,
        "shared_gate_sigmoid",
        lambda *_args: calls.append("shared_gate") or torch.ones(1),
    )

    output = moe.forward(torch.ones(1, 4))

    assert calls == ["router", "prepare", "shared", "shared_gate", "routed"]
    assert torch.equal(output, torch.full((1, 4), 2.0))


def test_q6_layer_keeps_its_auxiliary_cache_path(monkeypatch):
    import freetoken.layers.moe as moe_module
    from freetoken.layers.moe import OffloadMoELayer

    layer = object.__new__(OffloadMoELayer)
    layer.offload_cache = SimpleNamespace(quant_format="q4_k_q5_k", decode_target="gpu")
    layer.auxiliary_offload_cache = object()
    monkeypatch.setattr(moe_module, "_QWEN_GGUF_DECODE_OVERLAP", True)
    monkeypatch.setattr(torch.version, "hip", "7.14", raising=False)
    monkeypatch.setattr(
        moe_module,
        "fused_topk",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("Q6 must not prepare Q5")),
    )
    monkeypatch.setattr(
        moe_module,
        "get_global_ctx",
        lambda: SimpleNamespace(batch=SimpleNamespace(is_prefill=False)),
    )

    assert layer.prepare_qwen_decode_overlap(torch.zeros(1, 4), torch.zeros(1, 4)) is None
