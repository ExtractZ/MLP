# MLX C++ Metal Inference Engine

This project ports the Python MLX MLP inference path to native C++ on
Apple Silicon. It loads the trained SafeTensors checkpoint, runs inference
through MLX, and compares two custom Metal matrix-multiplication kernels:

- **Naive matmul:** one GPU thread computes one output element and reads
  matrix values directly from device memory.
- **Tiled matmul:** cooperatively loads 16x16 matrix tiles into Metal
  `threadgroup` memory and reuses them across multiply-add operations.

Both implementations also use a custom Metal ReLU kernel. The program runs
both inference paths and prints their predictions before running a larger
matmul benchmark.

## Requirements

- Apple Silicon Mac
- Full Xcode installation with the Metal compiler
- CMake 3.25 or newer
- C++20 compiler
- MLX checked out as the repository's `mlx` submodule
- `mlp_weights.safetensors` in the repository root

The full Xcode toolchain must be selected because MLX compiles Metal kernels:

```bash
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
xcrun --find metal
```

The second command should print the path to Xcode's `metal` compiler.

## Build

From the repository root:

```bash
git submodule update --init --recursive
cmake -S mlp_engine -B mlp_engine/build
cmake --build mlp_engine/build --parallel
```

The `mlp_engine/CMakeLists.txt` file builds MLX directly from the local
submodule, so no separately installed MLX C++ package is required.

## Run

Run the executable from `mlp_engine` because the checkpoint path is relative
to that directory:

```bash
cd mlp_engine
./build/inference_engine
```

The program:

1. Loads `../mlp_weights.safetensors`.
2. Runs the original and tiled Metal inference paths.
3. Prints both predicted classes for the four test points.
4. Benchmarks both matmul implementations using 512x512 matrices.

## Recorded Results

One benchmark run produced:

```text
--- Native C++ Inference Results ---
input=(-1.1, -1.1) -> baseline_class=0, tiled_class=0
input=(-1.1,  1.1) -> baseline_class=1, tiled_class=1
input=( 1.1, -1.1) -> baseline_class=2, tiled_class=2
input=( 1.1,  1.1) -> baseline_class=3, tiled_class=3

--- Matmul Benchmark ---
shape: [512, 512] x [512, 512]
warmup iterations: 5, timed iterations: 20
naive: 3.676 ms, 73.025 GFLOPS
tiled: 1.979 ms, 135.656 GFLOPS
tiled speedup: 1.86x
```

For this run, the tiled implementation reduced average matmul latency by
approximately 46% and achieved approximately 1.86x the throughput of the
naive implementation. Results vary with system load, GPU model, operating
system, matrix dimensions, and MLX/Metal kernel caching.

The benchmark reports GFLOPS using:

```text
2 * M * N * K / elapsed_seconds / 1,000,000,000
```

for a matrix multiplication of shape `[M, K] x [K, N]`.
