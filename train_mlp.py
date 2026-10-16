"""Train a small MLP on a deterministic four-class 2D dataset with MLX."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


EPOCHS = 200
LEARNING_RATE = 0.03
MIN_ACCURACY = 0.95


def make_dataset() -> tuple[mx.array, mx.array]:
    """Return four compact, linearly non-separable clusters."""
    centers = ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))
    offsets = (-0.30, -0.10, 0.10, 0.30)
    features: list[list[float]] = []
    labels: list[int] = []

    for label, (center_x, center_y) in enumerate(centers):
        for offset_x in offsets:
            for offset_y in offsets:
                features.append([center_x + offset_x, center_y + offset_y])
                labels.append(label)
    #print(mx.array(features, dtype=mx.float32), mx.array(labels, dtype=mx.int32))
    return mx.array(features, dtype=mx.float32), mx.array(labels, dtype=mx.int32) #this creates a list of 64 points of all combinations


class MLP(nn.Module):
    """A two-hidden-layer classifier for the four cluster classes."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = [
            nn.Linear(2, 16),
            nn.Linear(16, 16),
            nn.Linear(16, 4),
        ] #neural network layers with 2 inputs, 16 hidden units and 4 outputs (4 classes)

    def __call__(self, inputs: mx.array) -> mx.array:
        hidden = nn.relu(self.layers[0](inputs)) #describe forward propagation throught network
        hidden = nn.relu(self.layers[1](hidden))
        return self.layers[2](hidden)


def loss_fn(model: MLP, inputs: mx.array, targets: mx.array) -> mx.array: #how wrong the network is
    logits = model(inputs)
    return mx.mean(nn.losses.cross_entropy(logits, targets)) #find average loss across all samples


def main() -> None:
    mx.random.seed(7)
    inputs, targets = make_dataset() #array of features and labels [0,0,0...3,3,3]
    model = MLP()
    optimizer = optim.Adam(learning_rate=LEARNING_RATE)
    value_and_grad = nn.value_and_grad(model, loss_fn) #this returns average error

    for epoch in range(1, EPOCHS + 1):
        loss, gradients = value_and_grad(model, inputs, targets)
        optimizer.update(model, gradients) #this changes weights to reduce losses
        mx.eval(model.parameters(), optimizer.state, loss)

        if epoch == 1 or epoch % 20 == 0: #show epoch state
            predictions = mx.argmax(model(inputs), axis=1)
            accuracy = mx.mean(predictions == targets)
            mx.eval(accuracy)
            print(
                f"epoch {epoch:3d} | loss {loss.item():.4f} | "
                f"accuracy {accuracy.item():.2%}"
            )

    predictions = mx.argmax(model(inputs), axis=1)
    accuracy = mx.mean(predictions == targets)
    mx.eval(accuracy)
    if accuracy.item() < MIN_ACCURACY:
        raise RuntimeError(
            f"training did not reach {MIN_ACCURACY:.0%} accuracy "
            f"(got {accuracy.item():.2%})"
        )


if __name__ == "__main__":
    main()
