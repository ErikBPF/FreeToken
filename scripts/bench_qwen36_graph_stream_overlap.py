#!/usr/bin/env python3
"""Compare serialized and overlapped Qwen3.6 expert copies under HIP graphs."""

from __future__ import annotations

import argparse
import json
import statistics

HIDDEN = 2048
INTERMEDIATE = 512
EXPERTS = 256
TOP_K = 8
CACHE_SLOTS = EXPERTS


def summarize(serial_ms: list[float], overlap_ms: list[float]) -> dict[str, float]:
    serial = statistics.median(serial_ms)
    overlap = statistics.median(overlap_ms)
    return {
        "serial_ms_median": serial,
        "overlap_ms_median": overlap,
        "overlap_speedup": serial / overlap,
        "overlap_improvement_percent": (serial - overlap) / serial * 100,
    }


def outputs_equal(left, right) -> bool:
    import torch

    return torch.equal(left.view(torch.uint8), right.view(torch.uint8))


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--misses", type=int, choices=range(1, TOP_K + 1), default=2)
    parser.add_argument("--warmup", type=positive_int, default=20)
    parser.add_argument("--samples", type=positive_int, default=21)
    parser.add_argument("--replays", type=positive_int, default=50)
    parser.add_argument("--phase", choices=("down", "shared"), default="down")
    return parser.parse_args()


def build_rig(device: torch.device, misses: int):
    import torch

    from freetoken.moe.offload_cache import OffloadMoeCache

    gate_row = HIDDEN // 256 * 144
    down_row = INTERMEDIATE // 256 * 176
    sources = {
        "gate_up": [torch.zeros(
            (EXPERTS, 2 * INTERMEDIATE, gate_row), dtype=torch.uint8, pin_memory=True
        )],
        "down": [torch.zeros(
            (EXPERTS, HIDDEN, down_row), dtype=torch.uint8, pin_memory=True
        )],
    }
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=EXPERTS,
        cache_size=CACHE_SLOTS,
        device=device,
        quant_format="q4_k_q5_k",
    )
    cache.set_bank_sources(sources)
    cache.evict_slots[:misses] = torch.arange(misses, dtype=torch.int32, device=device)
    cache.src_indices[:misses] = torch.arange(misses, dtype=torch.int32, device=device)
    cache.num_indices.fill_(misses)
    hidden = torch.ones((1, HIDDEN), dtype=torch.bfloat16, device=device)
    routes = torch.arange(TOP_K, dtype=torch.int32, device=device).reshape(1, TOP_K)
    shared_gate_up = torch.zeros(
        (2 * INTERMEDIATE, HIDDEN // 32 * 34), dtype=torch.uint8, device=device
    )
    shared_down = torch.zeros(
        (HIDDEN, INTERMEDIATE // 32 * 34), dtype=torch.uint8, device=device
    )
    return cache, hidden, routes, shared_gate_up, shared_down


def copy_bank(cache, bank: int) -> None:
    from freetoken.kernel import fast_index_copy_jit

    sources, slots = cache.banks[bank]
    fast_index_copy_jit(
        slots, cache.evict_slots, sources[0], cache.src_indices, cache.num_indices
    )


def gate_gemv(cache, hidden: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
    from freetoken.kernel.gguf import ggml_moe_a8_vec
    from freetoken.models.gguf.dequant import GGML_Q4_K

    return ggml_moe_a8_vec(
        hidden, cache.banks[0][1], routes, TOP_K, int(GGML_Q4_K),
        2 * INTERMEDIATE, 1,
    )


def shared_gemv(hidden: torch.Tensor, gate_up_weight, down_weight) -> torch.Tensor:
    from freetoken.kernel.gguf import (
        ggml_mul_mat_vec_q8_0_silu,
        ggml_mul_mat_vec_q8_0_with_q8,
    )

    gate_up, _ = ggml_mul_mat_vec_q8_0_with_q8(
        gate_up_weight, hidden, 2 * INTERMEDIATE
    )
    return ggml_mul_mat_vec_q8_0_silu(down_weight, gate_up, HIDDEN)


def capture(
    cache,
    hidden: torch.Tensor,
    routes: torch.Tensor,
    shared_weights,
    overlap: bool,
    phase: str,
):
    import torch

    main = torch.cuda.Stream()
    copy = torch.cuda.Stream()
    fork = torch.cuda.Event()
    joined = torch.cuda.Event()
    output: list[torch.Tensor] = []
    shared_output: list[torch.Tensor] = []

    def body() -> None:
        if phase == "shared":
            if overlap:
                fork.record(main)
                with torch.cuda.stream(copy):
                    copy.wait_event(fork)
                    copy_bank(cache, 0)
                    copy_bank(cache, 1)
                    joined.record(copy)
                shared_output[:] = [shared_gemv(hidden, *shared_weights)]
                main.wait_event(joined)
            else:
                shared_output[:] = [shared_gemv(hidden, *shared_weights)]
                copy_bank(cache, 0)
                copy_bank(cache, 1)
            output[:] = [gate_gemv(cache, hidden, routes)]
            return
        copy_bank(cache, 0)
        if overlap:
            fork.record(main)
            with torch.cuda.stream(copy):
                copy.wait_event(fork)
                copy_bank(cache, 1)
                joined.record(copy)
            output[:] = [gate_gemv(cache, hidden, routes)]
            main.wait_event(joined)
        else:
            copy_bank(cache, 1)
            output[:] = [gate_gemv(cache, hidden, routes)]

    main.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(main):
        body()
    main.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=main):
        body()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output[0]


def verify(graph, output: torch.Tensor, cache, misses: int) -> torch.Tensor:
    import torch

    for _, slots in cache.banks:
        slots[:misses].fill_(255)
    graph.replay()
    torch.cuda.synchronize()
    if any(torch.count_nonzero(slots[:misses]).item() for _, slots in cache.banks):
        raise RuntimeError("captured graph did not copy every selected expert row")
    return output.clone()


def time_pair(serial, overlap, warmup: int, samples: int, replays: int):
    import torch

    for _ in range(warmup):
        serial.replay()
        overlap.replay()
    torch.cuda.synchronize()
    serial_ms, overlap_ms = [], []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        start.record()
        for _ in range(replays):
            serial.replay()
        end.record()
        end.synchronize()
        serial_ms.append(start.elapsed_time(end) / replays)

        start.record()
        for _ in range(replays):
            overlap.replay()
        end.record()
        end.synchronize()
        overlap_ms.append(start.elapsed_time(end) / replays)
    return serial_ms, overlap_ms


def main() -> int:
    import torch

    args = parse_args()
    if not torch.cuda.is_available() or not torch.version.hip:
        raise SystemExit("ROCm GPU required")
    from freetoken.gpu_select import bind_assigned_gpu

    device = bind_assigned_gpu()
    cache, hidden, routes, shared_gate_up, shared_down = build_rig(device, args.misses)
    shared_weights = (shared_gate_up, shared_down)
    serial, serial_output = capture(
        cache, hidden, routes, shared_weights, overlap=False, phase=args.phase
    )
    overlap, overlap_output = capture(
        cache, hidden, routes, shared_weights, overlap=True, phase=args.phase
    )
    serial_output = verify(serial, serial_output, cache, args.misses)
    overlap_output = verify(overlap, overlap_output, cache, args.misses)
    if not outputs_equal(serial_output, overlap_output):
        raise RuntimeError("overlapped graph changed Q4_K output")

    serial_ms, overlap_ms = time_pair(
        serial, overlap, args.warmup, args.samples, args.replays
    )
    copied_per_expert = sum(slots[0].numel() for _, slots in cache.banks)
    result = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(device),
        "hip": torch.version.hip,
        "misses": args.misses,
        "top_k": TOP_K,
        "phase": args.phase,
        "copied_bytes": copied_per_expert * args.misses,
        "graph_copy_verified": True,
        "output_exact": True,
        "samples": args.samples,
        "replays_per_sample": args.replays,
        **summarize(serial_ms, overlap_ms),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
