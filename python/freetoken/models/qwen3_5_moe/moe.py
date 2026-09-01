from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    make_moe_layer,
    silu_and_mul,
)

from freetoken.kernel.triton.fp8_block_linear import Fp8BlockColMerged, Fp8BlockLinear

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class _SharedExpert(BaseOP):
    """Always-present shared SwiGLU expert of width ``shared_expert_intermediate_size``."""

    def __init__(self, config: ModelConfig, hidden_size: int, intermediate_size: int):
        if getattr(config, "expert_quant", "none") == "fp8_block":
            self.gate_up_proj = Fp8BlockColMerged(
                hidden_size, [intermediate_size, intermediate_size], has_bias=False
            )
            self.down_proj = Fp8BlockLinear(intermediate_size, hidden_size, has_bias=False)
        elif getattr(config, "dense_quant", "none") == "nvfp4":
            # NVFP4 checkpoint: keep the shared expert's NVFP4 weights native (W4A16).
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseColMerged, Nvfp4DenseLinear

            self.gate_up_proj = Nvfp4DenseColMerged(
                hidden_size, [intermediate_size, intermediate_size], has_bias=False
            )
            self.down_proj = Nvfp4DenseLinear(intermediate_size, hidden_size, has_bias=False)
        else:
            self.gate_up_proj = LinearColParallelMerged(
                hidden_size, [intermediate_size, intermediate_size], has_bias=False
            )
            self.down_proj = LinearRowParallel(intermediate_size, hidden_size, has_bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))

    def forward_q8_0_with_q8(
        self, x: torch.Tensor, *, fused_silu: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_up, q8 = self.gate_up_proj.forward_q8_0_with_q8(x)
        shared = (
            self.down_proj.forward_q8_0_silu(gate_up)
            if fused_silu
            else self.down_proj.forward(silu_and_mul(gate_up))
        )
        return shared, q8


class Qwen3_5DenseMLP(_SharedExpert):
    """Dense (non-MoE) SwiGLU MLP for dense Qwen3.x checkpoints (e.g. 27B): ``gate_up_proj``
    (fused gate|up) + ``down_proj`` at full ``intermediate_size``. Same structure (and quant
    dispatch) as the shared expert -- NVFP4 (W4A16) when ``dense_quant=="nvfp4"``, else bf16 --
    so it reuses ``_SharedExpert`` directly and keeps the state-dict keys flat
    (``...layers.N.mlp.{gate_up_proj,down_proj}``)."""

    def __init__(self, config: ModelConfig):
        super().__init__(config, config.hidden_size, config.intermediate_size)


class Qwen3_5MoE(BaseOP):
    """Routed MoE (256 experts, top-8) plus a gated shared expert:

        out = routed(x) + sigmoid(shared_expert_gate(x)) * shared_expert(x)

    Router softmaxes over all experts, takes top-k, and renormalizes (HF semantics).
    """

    def __init__(self, config: ModelConfig, layer_id: int | None = None):
        weight_format = (
            "fp8_block" if getattr(config, "expert_quant", "none") == "fp8_block" else "bf16"
        )
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            weight_format=weight_format,
        )
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _SharedExpert(
            config, config.hidden_size, config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    def _router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch = get_global_ctx().batch
        if batch.is_prefill and batch.size > 1:
            lengths = [req.extend_len for req in batch.reqs]
            return torch.cat([self.gate.forward(x) for x in hidden_states.split(lengths)])
        return self.gate.forward(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # Compute the router + shared expert BEFORE the routed experts: the fused MoE
        # kernel may write into ``hidden_states`` in place, which would corrupt the
        # shared expert's input (HF also evaluates the shared expert first).
        from freetoken.moe.fused_q4_k_q5_k import QWEN_GGUF_FUSION_LEVEL

        reuse_q8 = (
            QWEN_GGUF_FUSION_LEVEL >= 1
            and hidden_states.shape[0] <= 6
            and hasattr(self.shared_expert.gate_up_proj, "forward_q8_0_with_q8")
        )
        router_logits = self._router_logits(hidden_states)
        prepared = (
            self.experts.prepare_qwen_decode_overlap(hidden_states, router_logits)
            if hasattr(self.experts, "prepare_qwen_decode_overlap")
            else None
        )
        if reuse_q8:
            shared, hidden_q8 = self.shared_expert.forward_q8_0_with_q8(
                hidden_states, fused_silu=QWEN_GGUF_FUSION_LEVEL >= 2
            )
        else:
            shared = self.shared_expert.forward(hidden_states)
            hidden_q8 = None
        if hidden_q8 is None:
            shared_gate = torch.sigmoid(self.shared_expert_gate.forward(hidden_states))
            shared = shared * shared_gate
        else:
            from freetoken.kernel.triton.moe_shared_gate import shared_gate_sigmoid

            shared_gate = shared_gate_sigmoid(
                hidden_states, self.shared_expert_gate.weight.view(-1)
            ).to(hidden_states.dtype)
        if prepared is not None:
            fuse_shared = hidden_states.shape[0] == 1 and QWEN_GGUF_FUSION_LEVEL >= 3
            routed = self.experts.decode_qwen_prepared(
                prepared,
                hidden_states,
                hidden_q8=hidden_q8,
                residual=shared if fuse_shared else None,
                residual_gate=shared_gate.view(-1) if fuse_shared else None,
            )
            if fuse_shared:
                return routed.view(num_tokens, hidden_dim)
            shared = shared * shared_gate[:, None]
        elif hidden_q8 is None:
            routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        else:
            fuse_shared = hidden_states.shape[0] == 1 and QWEN_GGUF_FUSION_LEVEL >= 3
            routed = self.experts.forward(
                hidden_states=hidden_states,
                router_logits=router_logits,
                hidden_q8=hidden_q8,
                residual=shared if fuse_shared else None,
                residual_gate=shared_gate.view(-1) if fuse_shared else None,
            )
            if fuse_shared:
                return routed.view(num_tokens, hidden_dim)
            shared = shared * shared_gate[:, None]
        return (routed + shared).view(num_tokens, hidden_dim)


__all__ = ["Qwen3_5MoE", "Qwen3_5DenseMLP"]
