"""Run inference with a saved MLX MLP and export its lazy computation graph."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


DEFAULT_POINTS = (
    (-1.1, -1.1),
    (-1.1, 1.1),
    (1.1, -1.1),
    (1.1, 1.1),
)


class MLP(nn.Module):
    """The same architecture used by train_mlp.py."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = [
            nn.Linear(2, 16),
            nn.Linear(16, 16),
            nn.Linear(16, 4),
        ]

    def __call__(self, inputs: mx.array) -> mx.array:
        hidden = nn.relu(self.layers[0](inputs))
        hidden = nn.relu(self.layers[1](hidden))
        return self.layers[2](hidden)


def load_inputs(input_file: Path | None) -> mx.array:
    """Load x,y rows from CSV, or use four representative cluster points."""
    if input_file is None:
        return mx.array(DEFAULT_POINTS, dtype=mx.float32)

    rows: list[list[float]] = []
    with input_file.open(newline="") as file:
        for line_number, row in enumerate(csv.reader(file), start=1):
            if not row or all(not value.strip() for value in row): # if blank lines
                continue
            if len(row) != 2:
                raise ValueError(
                    f"{input_file}:{line_number} must contain exactly two values"
                )
            try:
                rows.append([float(row[0]), float(row[1])])
            except ValueError as error:
                raise ValueError(
                    f"{input_file}:{line_number} contains a non-numeric value"
                ) from error

    if not rows:
        raise ValueError(f"{input_file} does not contain any input rows")
    return mx.array(rows, dtype=mx.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("mlp_weights.safetensors"),
        help="safetensors file produced by train_mlp.py",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="optional CSV file containing one x,y point per row",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("mlp_inference.dot"),
        help="DOT file for the lazy inference graph",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(
            f"weights file not found: {args.weights}. "
            "Run train_mlp.py first or pass --weights."
        )

    inputs = load_inputs(args.input_file) # for the lazy inference graph (the .dot)
    model = MLP() #create empty model
    model.load_weights(str(args.weights)) #load the weights in the empty model

    # The model call builds a lazy graph. Export it before eval executes it. the following line runs the model by computation.
    logits = model(inputs)
    args.graph.parent.mkdir(parents=True, exist_ok=True)
    mx.export_to_dot(str(args.graph), logits, inputs=inputs)
    predictions = mx.argmax(logits, axis=1) # takes largest index of each logits row
    mx.eval(logits, predictions) # this executes the computations

    print(f"loaded weights: {args.weights}")
    print(f"exported graph: {args.graph}")
    for point, scores, prediction in zip(inputs.tolist(), logits.tolist(), predictions.tolist()):
        print(
            f"input=({point[0]:6.2f}, {point[1]:6.2f}) "
            f"logits={[round(score, 3) for score in scores]} "
            f"predicted_class={prediction}"
        )


if __name__ == "__main__":
    main()
