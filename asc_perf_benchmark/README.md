# Ascend Chip Performance Benchmark

This directory contains a self-contained Ascend C benchmark operator that measures:

- **Compute performance** via matrix multiplication (Matmul API) → reports **TFLOPS**.
- **Memory bandwidth** via GM → UB → GM streaming copy → reports **GB/s**.

## Files

| File | Description |
|------|-------------|
| `asc_perf_benchmark.asc` | Kernel + host main, supports `compute` and `bandwidth` modes |
| `CMakeLists.txt` | Build configuration for `dav-2201` and `dav-3510` |
| `benchmark.sh` | One-shot build + run script |
| `README.md` | This file |

## Prerequisites

- Ascend NPU with CANN installed
- `source ${install_path}/cann/set_env.sh`
- CMake >= 3.16
- ASC CMake toolchain from CANN

## Build

```bash
mkdir -p build && cd build
cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..
make -j asc_perf_benchmark
```

For Ascend 950PR/950DT:

```bash
cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..
make -j asc_perf_benchmark
```

## Run

### Compute mode (TFLOPS)

```bash
./asc_perf_benchmark compute 4096 4096 4096
```

Optional warmup/measure iterations:

```bash
./asc_perf_benchmark compute 4096 4096 4096 3 10
```

Output example:

```text
=== Compute Benchmark (Matmul) ===
Shape            : 4096 x 4096 x 4096
Used cores       : 24
Warmup iters     : 3
Measure iters    : 10
Total time       : 123.456 ms
Avg time/iter    : 12.346 ms
TFLOPS           : 11.123
FLOPs/iter       : 1.374e+11
```

TFLOPS formula:

```text
FLOPs per matmul = 2 * M * N * K
TFLOPS = (FLOPs / 1e12) / (avg_time_seconds)
```

### Bandwidth mode (GB/s)

```bash
./asc_perf_benchmark bandwidth 67108864
```

Optional warmup/measure iterations:

```bash
./asc_perf_benchmark bandwidth 67108864 3 10
```

Output example:

```text
=== Bandwidth Benchmark (GM -> UB -> GM) ===
Total bytes      : 67108864
Total elements   : 33554432 (half)
Blocks used      : 8
Warmup iters     : 3
Measure iters    : 10
Total time       : 45.678 ms
Avg time/iter    : 4.568 ms
Bandwidth        : 14.678 GB/s
Note: bandwidth counts bytes moved (read + write).
```

Bandwidth formula:

```text
Bandwidth(GB/s) = total_bytes / avg_time_seconds / 1e9
```

## Quick run with `benchmark.sh`

```bash
# Compute benchmark, default dav-2201
./benchmark.sh compute 4096 4096 4096

# Bandwidth benchmark on Ascend 950PR/950DT
./benchmark.sh bandwidth 67108864 dav-3510
```

## Notes

- Compute mode uses the Matmul high-level API with FP16 A/B and FP32 accumulation.
- Bandwidth mode uses `half` elements; pass `total_bytes` as a multiple of 2.
- Reported bandwidth counts the bytes actually moved by the kernel (read + write).
- For deeper pipeline analysis, run the executable through `msprof op`:
  ```bash
  msprof op ./asc_perf_benchmark compute 4096 4096 4096
  ```
