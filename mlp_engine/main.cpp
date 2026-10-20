#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/fast.h>
#include <mlx/mlx.h>

namespace mx = mlx::core;

const std::string custom_relu_source = R"(
    uint index = thread_position_in_grid.x;
    out[index] = max(input[index], 0.0f);
)";

const std::string custom_matmul_source = R"(
    uint row = thread_position_in_grid.y;
    uint column = thread_position_in_grid.x;

    if (row >= input_shape[0] || column >= weight_shape[1]) {
        return;
    }

    float sum = 0.0f;
    for (uint k = 0; k < input_shape[1]; ++k) {
        sum += input[row * input_shape[1] + k] *weight[k * weight_shape[1] + column];
    }
    out[row * weight_shape[1] + column] = sum;
)";

constexpr int tile_size = 16;

const std::string tiled_matmul_source = R"(
    constexpr uint tile_size = 16;
    threadgroup float input_tile[tile_size][tile_size];
    threadgroup float weight_tile[tile_size][tile_size];

    uint local_row = thread_position_in_threadgroup.y;
    uint local_column = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.y * tile_size + local_row;
    uint column = threadgroup_position_in_grid.x * tile_size + local_column;
    uint tile_count = (input_shape[1] + tile_size - 1) / tile_size;
    float sum = 0.0f;

    for (uint tile = 0; tile < tile_count; ++tile) {
        uint input_column = tile * tile_size + local_column;
        uint weight_row = tile * tile_size + local_row;

        input_tile[local_row][local_column] =
            row < input_shape[0] && input_column < input_shape[1]
                ? input[row * input_shape[1] + input_column]
                : 0.0f;
        weight_tile[local_row][local_column] =
            weight_row < weight_shape[0] && column < weight_shape[1]
                ? weight[weight_row * weight_shape[1] + column]
                : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < tile_size; ++k) {
            sum += input_tile[local_row][k] * weight_tile[k][local_column];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < input_shape[0] && column < weight_shape[1]) {
        out[row * weight_shape[1] + column] = sum;
    }
)";

mx::array custom_relu(const mx::array& input) {
    const auto kernel = mx::fast::metal_kernel(
        "custom_relu", {"input"}, {"out"}, custom_relu_source);

    const int element_count = static_cast<int>(input.size());
    return kernel(
        {input},
        {input.shape()}, //output same shape as input
        {input.dtype()},
        {element_count, 1, 1}, //grid size
        {64, 1, 1}, //64 gpu positions, input[0] -> out[0]...
        {},
        std::nullopt,
        false, // no verbose
        mx::Device(mx::Device::gpu))[0]; //run kernel on gpu
}

mx::array custom_matmul(
    const mx::array& input,
    const mx::array& weight) {
    if (input.ndim() != 2 || weight.ndim() != 2 ||
        input.shape(1) != weight.shape(0)) {
        throw std::invalid_argument(
            "custom_matmul expects [M, K] and [K, N] arrays");
    }

    static const auto kernel = mx::fast::metal_kernel(
        "custom_matmul",
        {"input", "weight"},
        {"out"},
        custom_matmul_source);

    const int rows = input.shape(0);
    const int columns = weight.shape(1);
    return kernel(
        {input, weight},
        {{rows, columns}},
        {input.dtype()},
        {columns, rows, 1},
        {8, 8, 1}, //for parallelism
        {},
        std::nullopt,
        false,
        mx::Device(mx::Device::gpu))[0];
}

mx::array tiled_matmul(
    const mx::array& input,
    const mx::array& weight) {
    if (input.ndim() != 2 || weight.ndim() != 2 ||
        input.shape(1) != weight.shape(0)) {
        throw std::invalid_argument(
            "tiled_matmul expects [M, K] and [K, N] arrays");
    }
    if (input.dtype() != mx::float32 || weight.dtype() != mx::float32) {
        throw std::invalid_argument(
            "tiled_matmul currently supports float32 arrays only");
    }

    static const auto kernel = mx::fast::metal_kernel(
        "tiled_matmul",
        {"input", "weight"},
        {"out"},
        tiled_matmul_source);
    const auto contiguous_input = mx::contiguous(input);
    const auto contiguous_weight = mx::contiguous(weight);
    const int rows = contiguous_input.shape(0);
    const int columns = contiguous_weight.shape(1);
    const int grid_rows = ((rows + tile_size - 1) / tile_size) * tile_size;
    const int grid_columns =
        ((columns + tile_size - 1) / tile_size) * tile_size;

    return kernel(
        {contiguous_input, contiguous_weight},
        {{rows, columns}},
        {contiguous_input.dtype()},
        {grid_columns, grid_rows, 1},
        {tile_size, tile_size, 1},
        {},
        std::nullopt,
        false,
        mx::Device(mx::Device::gpu))[0];
}

template <typename Matmul>
double benchmark_matmul(
    const std::string& name,
    Matmul matmul,
    const mx::array& input,
    const mx::array& weight,
    int warmup_iterations,
    int timed_iterations) {
    for (int i = 0; i < warmup_iterations; ++i) {
        auto output = matmul(input, weight);
        output.eval();
        static_cast<void>(output.template data<float>()[0]);
    }
    //time the kernel
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < timed_iterations; ++i) {
        auto output = matmul(input, weight);
        output.eval();
        static_cast<void>(output.template data<float>()[0]);
    }
    const auto end = std::chrono::steady_clock::now();

    const double elapsed_seconds =
        std::chrono::duration<double>(end - start).count();
    const double average_seconds = elapsed_seconds / timed_iterations;
    const double m = static_cast<double>(input.shape(0));
    const double k = static_cast<double>(input.shape(1));
    const double n = static_cast<double>(weight.shape(1));
    const double gflops = (2.0 * m * n * k) / average_seconds / 1.0e9;

    std::cout << std::fixed << std::setprecision(3)
              << name << ": " << average_seconds * 1.0e3
              << " ms, " << gflops << " GFLOPS\n";
    return average_seconds;
}

void run_matmul_benchmark() {
    constexpr int rows = 512;
    constexpr int inner = 512;
    constexpr int columns = 512;
    constexpr int warmup_iterations = 5;
    constexpr int timed_iterations = 20;

    std::vector<float> input_data(rows * inner);
    std::vector<float> weight_data(inner * columns);
    for (size_t i = 0; i < input_data.size(); ++i) { // create deterministic data for benchmarking
        input_data[i] = static_cast<float>((i % 17) - 8) * 0.01f;
    }
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = static_cast<float>((i % 23) - 11) * 0.01f;
    }

    const mx::array input(
        input_data.begin(), {rows, inner}, mx::float32);
    const mx::array weight(
        weight_data.begin(), {inner, columns}, mx::float32);

    std::cout << "\n--- Matmul Benchmark ---\n"
              << "shape: [" << rows << ", " << inner << "] x ["
              << inner << ", " << columns << "]\n"
              << "warmup iterations: " << warmup_iterations
              << ", timed iterations: " << timed_iterations << "\n";

    const double naive_time = benchmark_matmul(
        "naive", custom_matmul, input, weight,
        warmup_iterations, timed_iterations);
    const double tiled_time = benchmark_matmul(
        "tiled", tiled_matmul, input, weight,
        warmup_iterations, timed_iterations);

    std::cout << std::fixed << std::setprecision(2)
              << "tiled speedup: " << naive_time / tiled_time << "x\n";
}

int main() {
    std::cout << "Initializing Native C++ MLX Inference Engine...\n";

    // 1. Load the SafeTensors weights saved by train_mlp.py.
    const std::string weights_path = "../mlp_weights.safetensors";

    try {
        //infer variable type from the value on the right
        auto [weights, metadata] = mx::load_safetensors(weights_path);
        std::cout << "Successfully loaded weights from " << weights_path << "\n";

        // same inputs as infer_mlp.py
        const std::vector<float> raw_inputs = {
            -1.1f, -1.1f,
            -1.1f,  1.1f,
             1.1f, -1.1f,
             1.1f,  1.1f,
        };
        mx::array inputs(raw_inputs.begin(), {4, 2}, mx::float32);

        // 3. Extract the parameters for the three linear layers
        const mx::array& w0 = weights.at("layers.0.weight");
        const mx::array& b0 = weights.at("layers.0.bias");
        const mx::array& w1 = weights.at("layers.1.weight");
        const mx::array& b1 = weights.at("layers.1.bias");
        const mx::array& w2 = weights. at("layers.2.weight");
        const mx::array& b2 = weights.at("layers.2.bias");

        //call custom kernels for inference
        auto h1_linear = custom_matmul(inputs, mx::transpose(w0)) + b0;
        auto h1 = custom_relu(h1_linear);
        auto h2_linear = custom_matmul(h1, mx::transpose(w1)) + b1;
        auto h2 = custom_relu(h2_linear);
        auto logits = custom_matmul(h2, mx::transpose(w2)) + b2;
        auto predictions = mx::argmax(logits, 1);

        // Keep the original kernel above as the baseline and run the
        // threadgroup-memory implementation as a separate comparison.
        auto tiled_h1_linear = tiled_matmul(inputs, mx::transpose(w0)) + b0;
        auto tiled_h1 = custom_relu(tiled_h1_linear);
        auto tiled_h2_linear = tiled_matmul(tiled_h1, mx::transpose(w1)) + b1;
        auto tiled_h2 = custom_relu(tiled_h2_linear);
        auto tiled_logits = tiled_matmul(tiled_h2, mx::transpose(w2)) + b2;
        auto tiled_predictions = mx::argmax(tiled_logits, 1);

        logits.eval();
        predictions.eval();
        tiled_logits.eval();
        tiled_predictions.eval();

        std::cout << "\n--- Native C++ Inference Results ---\n";
        const auto* preds_data = predictions.data<int32_t>();
        const auto* tiled_preds_data = tiled_predictions.data<int32_t>();
        for (int i = 0; i < 4; ++i) {
            std::cout << "input=(" << raw_inputs[2 * i] << ", "
                      << raw_inputs[2 * i + 1] << ")"
                      << " -> baseline_class=" << preds_data[i]
                      << ", tiled_class=" << tiled_preds_data[i] << "\n";
        }

        run_matmul_benchmark();
    } catch (const std::exception& error) {
        std::cerr << "Inference failed: " << error.what() << "\n";
        return 1;
    }

    return 0;
}