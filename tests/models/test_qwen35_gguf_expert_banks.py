"""Shape and registration checks for Qwen's mixed GGUF routed-expert banks."""

import inspect

import pytest
import torch

from freetoken.models.gguf.dequant import GGML_Q4_K, GGML_Q5_K, row_bytes
from freetoken.models.qwen3_5_moe.gguf import _expert_specs
from freetoken.moe.offload_cache import _BANK_BYTES_PER_EXPERT, _BANK_SCHEMAS


class _Config:
    """Small geometry carrier matching Qwen3.6-35B-A3B's routed MoE."""

    num_experts = 256
    hidden_size = 2048
    moe_intermediate_size = 512


def test_qwen_mixed_gguf_bank_shapes_preserve_each_tensor_encoding():
    """Gate/up and down rows keep their distinct Q4_K and Q5_K byte strides."""
    specs = _expert_specs(_Config())
    gate_shape, gate_dtype = specs["gate_up"]
    down_shape, down_dtype = specs["down"]

    assert gate_shape == (256, 1024, row_bytes(2048, GGML_Q4_K))
    assert down_shape == (256, 2048, row_bytes(512, GGML_Q5_K))
    assert str(gate_dtype) == "torch.uint8"
    assert str(down_dtype) == "torch.uint8"


def test_qwen_mixed_gguf_bank_budget_matches_the_two_exact_row_layouts():
    """Cache planning counts Q4_K gate/up bytes and Q5_K down bytes separately."""
    hidden, intermediate = 2048, 512
    expected = 2 * intermediate * row_bytes(hidden, GGML_Q4_K) + hidden * row_bytes(
        intermediate, GGML_Q5_K
    )
    assert _BANK_SCHEMAS["q4_k_q5_k"] == ("gate_up", "down")
    assert _BANK_BYTES_PER_EXPERT["q4_k_q5_k"](hidden, intermediate) == expected


def test_qwen_weighted_route_reduction_matches_torch():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA or ROCm")
    from freetoken.moe.fused_q4_k_q5_k import weighted_route_reduce

    torch.manual_seed(7)
    routed = torch.randn((3, 8, 2048), device="cuda", dtype=torch.bfloat16)
    weights = torch.softmax(torch.randn((3, 8), device="cuda"), dim=-1)
    expected = (routed * weights.to(routed.dtype).unsqueeze(-1)).sum(dim=1)

    actual = weighted_route_reduce(routed, weights)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

def test_qwen_down_path_fuses_silu_into_q8_quantization():
    from freetoken.moe import fused_q4_k_q5_k, fused_q4_k_q6_k

    for module in (fused_q4_k_q5_k, fused_q4_k_q6_k):
        source = inspect.getsource(module)
        assert "ggml_moe_a8_vec_silu" in source
        assert "QWEN_GGUF_FUSION_LEVEL < 2" in source


def test_qwen_gguf_fusion_modes_are_ordered():
    from freetoken.moe.fused_q4_k_q5_k import _fusion_level

    assert [_fusion_level(mode) for mode in ("none", "q8", "silu", "all")] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="FREETOKEN_QWEN_GGUF_FUSIONS"):
        _fusion_level("unknown")


def test_qwen_gguf_none_mode_uses_original_unfused_route(monkeypatch):
    from freetoken.kernel import gguf
    from freetoken.moe import fused_q4_k_q5_k as fused

    calls = []

    def regular(x, weight, ids, top_k, quant_type, row, tokens):
        calls.append(("regular", quant_type))
        return torch.zeros((tokens * top_k, row), dtype=torch.bfloat16)

    monkeypatch.setattr(fused, "QWEN_GGUF_FUSION_LEVEL", 0)
    monkeypatch.setattr(fused, "silu_and_mul", lambda x: x[:, : x.shape[1] // 2])
    monkeypatch.setattr(gguf, "ggml_moe_a8_vec", regular)
    monkeypatch.setattr(gguf, "ggml_moe_a8_vec_q8", lambda *args: calls.append("q8"))
    monkeypatch.setattr(gguf, "ggml_moe_a8_vec_silu", lambda *args: calls.append("silu"))
    monkeypatch.setattr(
        gguf, "ggml_moe_a8_vec_silu_reduce", lambda *args: calls.append("reduce")
    )

    output = fused.fused_experts_gguf_q4_k_q5_k(
        torch.zeros((1, 4), dtype=torch.bfloat16),
        torch.zeros((4, 8, 1), dtype=torch.uint8),
        torch.zeros((4, 4, 1), dtype=torch.uint8),
        torch.tensor([[0.75, 0.25]]),
        torch.tensor([[0, 1]], dtype=torch.int32),
        "silu",
        hidden_q8=torch.zeros((1, 9), dtype=torch.int32),
    )

    assert output.shape == (1, 4)
    assert calls == [("regular", 12), ("regular", 13)]


def test_qwen_fused_silu_q8_down_matches_unfused():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA or ROCm")
    from freetoken.kernel.gguf import (
        ggml_moe_a8_vec,
        ggml_moe_a8_vec_silu,
        ggml_moe_a8_vec_silu_reduce,
        ggml_mul_mat_vec_a8,
        ggml_mul_mat_vec_q8_0_silu,
    )
    from freetoken.models.gguf.dequant import GGML_Q8_0
    from freetoken.layers.activation import silu_and_mul

    torch.manual_seed(11)
    experts, routes, hidden, rows = 4, 8, 512, 64
    gate_up = torch.randn((routes, 2 * hidden), device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(
        0, 256,
        (experts, rows, row_bytes(hidden, GGML_Q5_K)),
        device="cuda",
        dtype=torch.uint8,
    )
    blocks = weight.view(experts, rows, -1, 176)
    blocks[..., 0] = 0
    blocks[..., 1] = 60
    blocks[..., 2] = 0
    blocks[..., 3] = 60
    ids = (torch.arange(routes, device="cuda", dtype=torch.int32) % experts).view(1, routes)

    expected = ggml_moe_a8_vec(
        silu_and_mul(gate_up), weight, ids, 1, int(GGML_Q5_K), rows, routes
    )
    actual = ggml_moe_a8_vec_silu(gate_up, weight, ids, int(GGML_Q5_K), rows)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    route_weights = torch.softmax(torch.randn((1, routes), device="cuda"), dim=-1)
    reduced_expected = (expected.view(1, routes, rows) * route_weights.to(
        expected.dtype
    ).unsqueeze(-1)).sum(dim=1)
    reduced_actual = ggml_moe_a8_vec_silu_reduce(
        gate_up, weight, ids, route_weights, int(GGML_Q5_K), rows
    )
    torch.testing.assert_close(reduced_actual, reduced_expected, rtol=0, atol=0)
    residual = torch.randn_like(reduced_expected)
    residual_actual = ggml_moe_a8_vec_silu_reduce(
        gate_up, weight, ids, route_weights, int(GGML_Q5_K), rows, residual
    )
    torch.testing.assert_close(
        residual_actual, reduced_expected + residual, rtol=0, atol=0
    )
    residual_gate = torch.sigmoid(torch.randn((1,), device="cuda", dtype=torch.bfloat16))
    gated_actual = ggml_moe_a8_vec_silu_reduce(
        gate_up, weight, ids, route_weights, int(GGML_Q5_K), rows,
        residual, residual_gate,
    )
    torch.testing.assert_close(
        gated_actual,
        reduced_expected + residual * residual_gate[:, None],
        rtol=0,
        atol=0,
    )

    dense = torch.randint(
        0, 256, (rows, row_bytes(hidden, GGML_Q8_0)),
        device="cuda", dtype=torch.uint8,
    )
    dense_blocks = dense.view(rows, -1, 34)
    dense_blocks[..., 0] = 0
    dense_blocks[..., 1] = 60
    dense_expected = ggml_mul_mat_vec_a8(
        dense, silu_and_mul(gate_up), int(GGML_Q8_0), rows
    )
    dense_actual = ggml_mul_mat_vec_q8_0_silu(dense, gate_up, rows)
    torch.testing.assert_close(dense_actual, dense_expected, rtol=0, atol=0)


def test_qwen_shared_q8_activation_reuse_matches_independent_quantization():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA or ROCm")
    from freetoken.kernel.gguf import (
        ggml_moe_a8_vec,
        ggml_moe_a8_vec_q8,
        ggml_mul_mat_vec_a8,
        ggml_mul_mat_vec_q8_0_with_q8,
    )
    from freetoken.models.gguf.dequant import GGML_Q8_0

    torch.manual_seed(13)
    experts, routes, hidden, rows = 4, 8, 512, 64
    x = torch.randn((1, hidden), device="cuda", dtype=torch.bfloat16)
    dense = torch.randint(
        0, 256, (2 * hidden, row_bytes(hidden, GGML_Q8_0)),
        device="cuda", dtype=torch.uint8,
    )
    dense_blocks = dense.view(2 * hidden, -1, 34)
    dense_blocks[..., 0] = 0
    dense_blocks[..., 1] = 60
    routed = torch.randint(
        0, 256, (experts, rows, row_bytes(hidden, GGML_Q4_K)),
        device="cuda", dtype=torch.uint8,
    )
    routed_blocks = routed.view(experts, rows, -1, 144)
    routed_blocks[..., 0] = 0
    routed_blocks[..., 1] = 60
    routed_blocks[..., 2] = 0
    routed_blocks[..., 3] = 60
    ids = (torch.arange(routes, device="cuda", dtype=torch.int32) % experts).view(1, routes)

    dense_expected = ggml_mul_mat_vec_a8(dense, x, int(GGML_Q8_0), 2 * hidden)
    dense_actual, q8 = ggml_mul_mat_vec_q8_0_with_q8(dense, x, 2 * hidden)
    routed_expected = ggml_moe_a8_vec(
        x, routed, ids, routes, int(GGML_Q4_K), rows, 1
    )
    routed_actual = ggml_moe_a8_vec_q8(x, q8, routed, ids, routes, rows, 1)

    torch.testing.assert_close(dense_actual, dense_expected, rtol=0, atol=0)
    torch.testing.assert_close(routed_actual, routed_expected, rtol=0, atol=0)


def test_qwen_shared_gate_reduction_matches_bf16_linear_sigmoid():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA or ROCm")
    from freetoken.kernel.triton.moe_shared_gate import shared_gate_sigmoid

    torch.manual_seed(19)
    hidden = torch.randn((1, 2048), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((2048,), device="cuda", dtype=torch.bfloat16)
    expected = torch.sigmoid(hidden @ weight[:, None]).view(-1)
    actual = shared_gate_sigmoid(hidden, weight).to(torch.bfloat16)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qwen_shared_q8_reuse_keeps_shared_expert_for_multi_token_batch(monkeypatch):
    from types import SimpleNamespace

    from freetoken.kernel.triton import moe_shared_gate
    from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE

    hidden = torch.ones((2, 3), dtype=torch.bfloat16)
    shared = torch.tensor([[2, 4, 6], [8, 10, 12]], dtype=torch.bfloat16)
    routed = torch.full_like(shared, 3)
    gates = torch.tensor([0.25, 0.5], dtype=torch.bfloat16)

    class SharedExpert:
        gate_up_proj = SimpleNamespace(forward_q8_0_with_q8=True)

        def forward_q8_0_with_q8(self, _hidden, *, fused_silu):
            assert fused_silu is True
            return shared, object()

    class Experts:
        def forward(self, **kwargs):
            self.kwargs = kwargs
            return routed

    moe = Qwen3_5MoE.__new__(Qwen3_5MoE)
    moe.gate = SimpleNamespace(forward=lambda _hidden: torch.zeros((2, 4)))
    moe.shared_expert = SharedExpert()
    moe.shared_expert_gate = SimpleNamespace(weight=torch.ones((1, 3)))
    moe.experts = Experts()
    monkeypatch.setattr(
        "freetoken.models.qwen3_5_moe.moe.get_global_ctx",
        lambda: SimpleNamespace(batch=SimpleNamespace(is_prefill=False, size=2)),
    )
    monkeypatch.setattr(moe_shared_gate, "shared_gate_sigmoid", lambda *_args: gates)

    actual = moe.forward(hidden)

    torch.testing.assert_close(actual, routed + shared * gates[:, None], rtol=0, atol=0)
    assert moe.experts.kwargs["hidden_q8"] is not None
    assert moe.experts.kwargs.get("residual") is None
    assert moe.experts.kwargs.get("residual_gate") is None


def test_qwen_prefill_router_projects_each_request_separately(monkeypatch):
    from types import SimpleNamespace

    from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE

    calls = []
    moe = Qwen3_5MoE.__new__(Qwen3_5MoE)
    moe.gate = SimpleNamespace(
        forward=lambda x: calls.append(x.shape[0]) or x[:, :2]
    )
    batch = SimpleNamespace(
        is_prefill=True,
        size=2,
        reqs=[SimpleNamespace(extend_len=3), SimpleNamespace(extend_len=2)],
    )
    monkeypatch.setattr(
        "freetoken.models.qwen3_5_moe.moe.get_global_ctx",
        lambda: SimpleNamespace(batch=batch),
    )
    hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)

    actual = moe._router_logits(hidden)

    assert calls == [3, 2]
    torch.testing.assert_close(actual, hidden[:, :2])
