from pathlib import Path


def test_gguf_moe_output_skips_redundant_zero_fill():
    source = (
        Path(__file__).resolve().parents[2]
        / "python/freetoken/kernel/csrc/gguf/gguf_kernel.cu"
    ).read_text()
    function = source.split("torch::Tensor ggml_moe_a8_vec(", 1)[1].split(
        "torch::Tensor ggml_mul_mat_a8(", 1
    )[0]

    assert "torch::empty({tokens * top_k, row}" in function
    assert "torch::zeros({tokens * top_k, row}" not in function


def test_every_quant_dispatch_fails_closed():
    source = (
        Path(__file__).resolve().parents[2]
        / "python/freetoken/kernel/csrc/gguf/gguf_kernel.cu"
    ).read_text()

    assert source.count('TORCH_CHECK(false, "unsupported GGUF quant type: ", type);') == 5
