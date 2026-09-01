from types import SimpleNamespace
from unittest.mock import patch

import torch

from freetoken.models.gguf.tokenizer import _register_embedded_special_tokens
from freetoken.models.qwen3_5_moe.gguf import GGUFLMHead


class FakeTokenizer:
    bos_token = "<bos>"
    eos_token = "<eos>"
    unk_token = "<unk>"
    pad_token = "<pad>"

    def __init__(self):
        self.registered = None

    def add_special_tokens(self, tokens):
        self.registered = tokens


def test_qwen_control_and_user_tokens_are_registered_atomically():
    tokenizer = FakeTokenizer()

    _register_embedded_special_tokens(
        tokenizer,
        ["ordinary", "<think>", "</think>", "<eos>"],
        [1, 4, 3, 3],
    )

    assert tokenizer.registered == {
        "additional_special_tokens": ["<think>", "</think>"]
    }


def test_qwen_lm_head_scores_only_each_requests_last_prefill_token():
    head = object.__new__(GGUFLMHead)
    head.qweight = torch.empty(1)
    batch = SimpleNamespace(
        is_prefill=True,
        size=2,
        attn_metadata=SimpleNamespace(
            get_last_indices=lambda size: torch.tensor([1, 4])
        ),
    )
    hidden = torch.arange(15, dtype=torch.float32).reshape(5, 3)

    with (
        patch("freetoken.core.get_global_ctx", return_value=SimpleNamespace(batch=batch)),
        patch(
            "freetoken.layers.gguf.fused_mul_mat_gguf",
            side_effect=lambda x, _weight, _quant: x,
        ),
    ):
        output = head.forward(hidden)

    torch.testing.assert_close(output, hidden[[1, 4]])
