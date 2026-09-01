import ast
from pathlib import Path


def test_offload_expert_gemm_accepts_the_down_copy_event():
    source = Path(__file__).resolve().parents[2] / "python/freetoken/layers/moe.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OffloadMoELayer")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_expert_gemm"
    )

    assert "down_ready_event" in [arg.arg for arg in method.args.kwonlyargs]
