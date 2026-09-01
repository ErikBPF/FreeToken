import json
import math
from pathlib import Path


def test_public_overlap_evidence_is_internally_consistent():
    artifact = Path(__file__).parents[1] / (
        "benchmarks/results/qwen36-graph-stream-overlap-gfx1201.json"
    )
    result = json.loads(artifact.read_text())["result"]

    assert result["graph_copy_verified"]
    assert result["output_exact"]
    assert math.isclose(
        result["overlap_speedup"],
        result["serial_ms_median"] / result["overlap_ms_median"],
    )
    assert math.isclose(
        result["overlap_improvement_percent"],
        (result["serial_ms_median"] - result["overlap_ms_median"])
        / result["serial_ms_median"]
        * 100,
    )
