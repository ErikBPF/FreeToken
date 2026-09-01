# ROCm HIP graph copy: upstream split

## FreeToken patch

HIP graph replay faults when `fast_index_copy_multi_jit` receives pinned-host
source addresses indirectly through its device pointer table. Keep the fused
path for eager ROCm and CUDA; while HIP is capturing, use the existing per-bank
copy whose source is a direct tensor argument:

```python
use_fused = self._copy_fused_ok and not (
    torch.version.hip and torch.cuda.is_current_stream_capturing()
)
```

The patch is confined to `OffloadMoeCache.copy_missing`. The CPU regression
`test_rocm_graph_capture_uses_per_bank_copy` checks dispatch. The slow HIP test
`test_qwen36_sized_pinned_rows_survive_rocm_graph_replay` uses the real Qwen3.6
geometry (40 layers, 256 experts, 2,117 slots, top-k 8, 1,179,648-byte Q4_K
gate/up rows, and 720,896-byte Q5_K down rows) for 512 replays.

```sh
pytest -q tests/moe/test_fused_copy.py::test_rocm_graph_capture_uses_per_bank_copy
pytest -q tests/moe/test_fused_copy.py::test_qwen36_sized_pinned_rows_survive_rocm_graph_replay
```

A machine with 128 GB RAM, a Ryzen 9 5950X CPU, and an RX 9070 XT GPU
(`gfx1201`) passed the real-row test in 17.96 seconds. The full endpoint then passed
three 1,024-token generations at 64.536 median decode tok/s with identical
graph-off output and canary hashes. A later 10-sample soak remained exact and
fault-free, although its current-host median was only 32.349 tok/s; do not mix
that separate performance drift into this correctness patch.

## Flashlib issue/reproducer

Report the lower-level problem separately: on ROCm gfx1201, capture a kernel
that copies several realistic pinned rows by reading source addresses from a
device-resident pointer table, then replay while changing route indices. The
small synthetic case is insufficient; use the row sizes and layer/cache shape
above. Compare these cases:

| Capture path | Full-model result |
|---|---|
| Skip expert copies | stable, 94.856 tok/s |
| Direct per-bank tensor arguments | stable and exact, 64.681 tok/s |
| Fused indirect source-pointer table | GPU page fault; later observed in `_lru_ensure_kernel` |

The likely mapping-lifetime explanation is an inference, not a proven runtime
root cause. The useful Flashlib acceptance gate is simpler: the realistic-row
reproducer survives repeated HIP graph replay with exact copied fingerprints.

## Reviewable series

1. FreeToken PR: fallback plus the two regressions; no kernel tuning.
2. Flashlib issue/PR: minimized indirect-pointer reproducer and runtime/kernel
   fix, after which FreeToken can remove the HIP capture fallback.
3. Performance PRs: dense Q8_0 first; split the remaining profile buckets before
   selecting another kernel or copy-scheduling change. Keep these out of the
   correctness change.

No issue or PR has been published from this checkout.
