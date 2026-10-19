#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/mlx.h>

namespace mx = mlx::core;

int main() {
    std::cout << "Initializing Native C++ MLX Inference Engine...\n";

    // 1. Load the SafeTensors weights saved by train_mlp.py.
    const std::string weights_path = "../mlp_weights.safetensors";

    try {
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

        // 3. Extract the parameters for the three Linear layers.
        const mx::array& w0 = weights.at("layers.0.weight");
        const mx::array& b0 = weights.at("layers.0.bias");
        const mx::array& w1 = weights.at("layers.1.weight");
        const mx::array& b1 = weights.at("layers.1.bias");
        const mx::array& w2 = weights.at("layers.2.weight");
        const mx::array& b2 = weights.at("layers.2.bias");

        // 4. Execute the same forward pass in infer_mlp.py with the layers
        const mx::array zero(0.0f);
        auto h1 = mx::maximum(
            mx::matmul(inputs, mx::transpose(w0)) + b0, zero); //do reLU, after applying wiehgt and bias
        auto h2 = mx::maximum(
            mx::matmul(h1, mx::transpose(w1)) + b1, zero);
        auto logits = mx::matmul(h2, mx::transpose(w2)) + b2;
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