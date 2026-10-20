#include <cstdint>
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

    const auto kernel = mx::fast::metal_kernel(
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

        logits.eval();
        predictions.eval();

        std::cout << "\n--- Native C++ Inference Results ---\n";
        const auto* preds_data = predictions.data<int32_t>();
        for (int i = 0; i < 4; ++i) {
            std::cout << "input=(" << raw_inputs[2 * i] << ", "
                      << raw_inputs[2 * i + 1] << ")"
                      << " -> predicted_class=" << preds_data[i] << "\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "Inference failed: " << error.what() << "\n";
        return 1;
    }

    return 0;
}