import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/bench_qwen36_graph_stream_overlap.py"


def load_benchmark():
    spec = importlib.util.spec_from_file_location("qwen36_graph_stream_overlap", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_summary_reports_overlap_speedup_from_medians():
    benchmark = load_benchmark()

    result = benchmark.summarize([4.0, 5.0, 6.0], [3.0, 4.0, 5.0])

    assert result == {
        "serial_ms_median": 5.0,
        "overlap_ms_median": 4.0,
        "overlap_speedup": 1.25,
        "overlap_improvement_percent": 20.0,
    }


def test_rig_respects_offload_cache_slot_floor():
    benchmark = load_benchmark()

    assert benchmark.CACHE_SLOTS >= benchmark.EXPERTS


def test_output_comparison_is_bit_exact(monkeypatch):
    benchmark = load_benchmark()
    uint8 = object()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(uint8=uint8, equal=lambda left, right: left == right),
    )

    class Tensor:
        def __init__(self, bits):
            self.bits = bits

        def view(self, dtype):
            assert dtype is uint8
            return self.bits

    assert benchmark.outputs_equal(Tensor(b"same"), Tensor(b"same"))
    assert not benchmark.outputs_equal(Tensor(b"left"), Tensor(b"right"))


def test_cli_selects_shared_expert_overlap(monkeypatch):
    benchmark = load_benchmark()
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--phase", "shared"])

    assert benchmark.parse_args().phase == "shared"
