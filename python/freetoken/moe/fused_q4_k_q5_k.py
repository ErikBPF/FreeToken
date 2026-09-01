"""Mixed Q4_K/Q5_K GGUF routed-expert execution for Qwen3.6 MoE checkpoints.

The GGUF model recipe names this combination ``Q4_K_M``, but its tensor table
stores the gate and up expert projections as Q4_K and the down projection as
Q5_K.  The borrowed HIP GGML kernels dispatch one quant type per matrix, so
this module intentionally launches one packed Q4_K MoE GEMV followed by one
packed Q5_K MoE GEMV.  Neither weight is dequantized to a persistent bf16 copy.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from freetoken.layers.activation import silu_and_mul
from freetoken.models.gguf.dequant import GGML_Q4_K, GGML_Q5_K


def _fusion_level(mode: str) -> int:
    levels = {"none": 0, "q8": 1, "silu": 2, "all": 3}
    try:
        return levels[mode.strip().lower()]
    except KeyError:
        raise ValueError(
            "FREETOKEN_QWEN_GGUF_FUSIONS must be none, q8, silu, or all, "
            f"got {mode!r}"
        ) from None


QWEN_GGUF_FUSION_LEVEL = _fusion_level(
    os.environ.get("FREETOKEN_QWEN_GGUF_FUSIONS", "all")
)


@triton.jit
def _weighted_route_reduce_kernel(
    routed_ptr,
    weights_ptr,
    output_ptr,
    hidden: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < hidden
    accumulator = tl.zeros((BLOCK,), dtype=tl.float32)
    for route in tl.static_range(top_k):
        value = tl.load(
            routed_ptr + (token * top_k + route) * hidden + offsets,
            mask=mask,
            other=0.0,
        )
        weight = tl.load(weights_ptr + token * top_k + route)
        weight_bits = weight.to(tl.uint32, bitcast=True)
        weight_lsb = (weight_bits >> 16) & 1
        weight = ((weight_bits + 0x7FFF + weight_lsb) & 0xFFFF0000).to(
            tl.float32, bitcast=True
        )
        product = value.to(tl.float32) * weight.to(tl.float32)
        bits = product.to(tl.uint32, bitcast=True)
        lsb = (bits >> 16) & 1
        rounded = ((bits + 0x7FFF + lsb) & 0xFFFF0000).to(
            tl.float32, bitcast=True
        )
        accumulator += rounded
    tl.store(output_ptr + token * hidden + offsets, accumulator, mask=mask)


def weighted_route_reduce(
    routed: torch.Tensor, topk_weights: torch.Tensor
) -> torch.Tensor:
    """Apply BF16 route weights and reduce routes in one kernel."""
    assert routed.is_contiguous() and topk_weights.is_contiguous()
    tokens, top_k, hidden = routed.shape
    assert topk_weights.shape == (tokens, top_k)
    output = torch.empty((tokens, hidden), dtype=routed.dtype, device=routed.device)
    block = min(triton.next_power_of_2(hidden), 1024)
    _weighted_route_reduce_kernel[(tokens, triton.cdiv(hidden, block))](
        routed,
        topk_weights,
        output,
        hidden,
        top_k,
        BLOCK=block,
        num_warps=8,
    )
    return output


def fused_experts_gguf_q4_k_q5_k(
    hidden_states: torch.Tensor,
    gate_up_q4_k: torch.Tensor,
    down_q5_k: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    hidden_q8: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    residual_gate: torch.Tensor | None = None,
    down_ready_event: torch.cuda.Event | None = None,
) -> torch.Tensor:
    """Run packed Q4_K gate/up then packed Q5_K down over routed experts.

    ``topk_ids`` already name the materialized GGUF expert-cache slots.  Qwen
    uses SwiGLU, so only ``silu`` is accepted here.  Explicit validation prevents
    a future model family from silently receiving Qwen's activation semantics.
    """
    if activation != "silu":
        raise ValueError(
            "Qwen mixed GGUF experts require the checkpoint's silu SwiGLU activation, "
            f"got {activation!r}"
        )
    from freetoken.kernel.gguf import (
        ggml_moe_a8_vec,
        ggml_moe_a8_vec_q8,
        ggml_moe_a8_vec_silu,
        ggml_moe_a8_vec_silu_reduce,
    )

    tokens = hidden_states.shape[0]
    top_k = topk_ids.shape[1]
    fused_width = gate_up_q4_k.shape[1]
    hidden_size = down_q5_k.shape[1]
    if hidden_q8 is None or QWEN_GGUF_FUSION_LEVEL < 1:
        gate_up = ggml_moe_a8_vec(
            hidden_states, gate_up_q4_k, topk_ids, top_k,
            int(GGML_Q4_K), fused_width, tokens,
        )
    else:
        gate_up = ggml_moe_a8_vec_q8(
            hidden_states, hidden_q8, gate_up_q4_k, topk_ids,
            top_k, fused_width, tokens,
        )
    if down_ready_event is not None:
        down_ready_event.wait()
    if QWEN_GGUF_FUSION_LEVEL < 2:
        intermediate = silu_and_mul(gate_up)
        output = ggml_moe_a8_vec(
            intermediate, down_q5_k, topk_ids, 1,
            int(GGML_Q5_K), hidden_size, tokens * top_k,
        ).reshape(tokens, top_k, hidden_size)
        return (output * topk_weights.reshape(tokens, top_k, 1).to(output.dtype)).sum(dim=1)
    if tokens == 1 and QWEN_GGUF_FUSION_LEVEL >= 3:
        return ggml_moe_a8_vec_silu_reduce(
            gate_up, down_q5_k, topk_ids, topk_weights, int(GGML_Q5_K), hidden_size,
            residual,
            residual_gate,
        )
    output = ggml_moe_a8_vec_silu(
        gate_up, down_q5_k, topk_ids, int(GGML_Q5_K), hidden_size
    )
    output = output.reshape(tokens, top_k, hidden_size)
    return weighted_route_reduce(output, topk_weights)


__all__ = ["fused_experts_gguf_q4_k_q5_k", "weighted_route_reduce"]
