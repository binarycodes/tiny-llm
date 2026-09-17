import json
import math
import time
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np

from tiny_llm.config import (
    BATCH_SIZE,
    CONFIG_FILE,
    CONTEXT_SIZE,
    DIMS,
    EVAL_EVERY,
    LEARNING_RATE,
    NUM_HEADS,
    NUM_LAYERS,
    PATIENCE,
    RANDOM_SEED,
    REPORT_EVERY,
    TENSORS_FILE,
    TOKEN_DTYPE,
    TRAIN_FILE,
    TRAIN_STEPS,
    VALID_FILE,
    VOCAB_SIZE,
    WEIGHT_DECAY,
    create_directories,
)
from tiny_llm.model import TinyLM

type TokenArray = np.ndarray[tuple[int], np.dtype[TOKEN_DTYPE]]
type SequenceArray = np.ndarray[tuple[int, int], np.dtype[TOKEN_DTYPE]]
type TrainStep = Callable[[mx.array], mx.array]


def load_tokens(
    path: Path,
) -> TokenArray:
    data = np.fromfile(
        path,
        dtype=TOKEN_DTYPE,
    )
    return data.astype(TOKEN_DTYPE)


def make_sequences(
    dataset: TokenArray,
) -> SequenceArray:
    window = CONTEXT_SIZE + 1
    count = len(dataset) // window
    trimmed = dataset[: count * window]
    return trimmed.reshape(
        count,
        window,
    )


def load_data() -> tuple[SequenceArray, SequenceArray]:
    train_data = load_tokens(TRAIN_FILE)
    valid_data = load_tokens(VALID_FILE)

    train_sequences = make_sequences(train_data)
    valid_sequences = make_sequences(valid_data)

    print("train tokens:", len(train_data))
    print("valid tokens:", len(valid_data))
    print("train sequences:", len(train_sequences))
    print("valid sequences:", len(valid_sequences))

    return train_sequences, valid_sequences


def batches(
    sequences: SequenceArray,
) -> Iterator[mx.array]:
    while True:
        indexes = np.random.permutation(len(sequences))
        for start in range(
            0,
            len(indexes) - BATCH_SIZE + 1,
            BATCH_SIZE,
        ):
            selected = indexes[start : start + BATCH_SIZE]
            yield mx.array(sequences[selected])


def flat_parameters(module: nn.Module) -> dict[str, mx.array]:
    return dict(mlx.utils.tree_flatten(module.parameters()))


def build_model() -> TinyLM:
    model = TinyLM(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        dims=DIMS,
        num_heads=NUM_HEADS,
    )

    mx.eval(model.parameters())
    parameter_count = sum(value.size for value in flat_parameters(model).values())
    print(f"Parameters: {parameter_count:,} ({parameter_count / 1e6:.1f}M)")

    return model


def loss_fn(
    model: TinyLM,
    batch: mx.array,
) -> mx.array:
    x = batch[:, :-1]
    y = batch[:, 1:]
    logits = model(x)

    return nn.losses.cross_entropy(
        logits,
        y,
        reduction="mean",
    )


def make_train_step(
    model: TinyLM,
    optimizer: optim.Optimizer,
) -> tuple[TrainStep, list[object]]:
    state: list[object] = [
        model.state,
        optimizer.state,
    ]

    loss_and_grad = cast(
        Callable[[TinyLM, mx.array], tuple[mx.array, dict[str, Any]]],
        nn.value_and_grad(
            model,
            loss_fn,
        ),
    )

    @partial(
        mx.compile,
        inputs=state,
        outputs=state,
    )
    def train_step(
        batch: mx.array,
    ) -> mx.array:
        loss, gradients = loss_and_grad(
            model,
            batch,
        )
        optimizer.update(
            model,
            gradients,
        )
        return loss

    return train_step, state


def evaluate(
    model: TinyLM,
    valid_sequences: SequenceArray,
) -> float:
    losses: list[float] = []
    max_batches = min(
        20,
        len(valid_sequences) // BATCH_SIZE,
    )
    for index in range(max_batches):
        start = index * BATCH_SIZE
        batch = mx.array(valid_sequences[start : start + BATCH_SIZE])
        loss = loss_fn(
            model,
            batch,
        )
        mx.eval(loss)
        losses.append(float(loss))
    return sum(losses) / len(losses)


def train(
    model: TinyLM,
    train_sequences: SequenceArray,
    valid_sequences: SequenceArray,
) -> None:
    optimizer = optim.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    train_step, state = make_train_step(model, optimizer)

    train_batches = batches(train_sequences)

    # early stopping parmeters
    best_validation_loss = float("inf")
    evaluations_without_improvement = 0

    start_time = time.perf_counter()
    for step in range(
        1,
        TRAIN_STEPS + 1,
    ):
        batch = next(train_batches)
        loss = train_step(batch)
        mx.eval(state)
        if step % REPORT_EVERY == 0:
            elapsed = time.perf_counter() - start_time
            print(f"step={step:6d} loss={loss.item():.4f} steps/s={REPORT_EVERY / elapsed:.2f}")
            start_time = time.perf_counter()

        if step % EVAL_EVERY == 0:
            validation_loss = evaluate(model, valid_sequences)
            perplexity = math.exp(validation_loss)
            print(f"validation loss={validation_loss:.4f} ppl={perplexity:.2f}")

            # early stopping implementation
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                evaluations_without_improvement = 0

                mx.save_safetensors(
                    str(TENSORS_FILE),
                    flat_parameters(model),
                )

                print(f"Saved new best checkpoint (validation loss={validation_loss:.4f})")
            else:
                evaluations_without_improvement += 1

                # if we have seen PATIENCE counts of worse improvements then just stop
                if evaluations_without_improvement >= PATIENCE:
                    print("Early stopping.")
                    break


def save_config() -> None:
    config = {
        "vocab_size": VOCAB_SIZE,
        "context_size": CONTEXT_SIZE,
        "num_layers": NUM_LAYERS,
        "dims": DIMS,
        "num_heads": NUM_HEADS,
    }

    CONFIG_FILE.write_text(
        json.dumps(
            config,
            indent=2,
        )
    )

    print("Saved config.")


def main() -> None:
    create_directories()

    mx.random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train_sequences, valid_sequences = load_data()
    model = build_model()
    train(model, train_sequences, valid_sequences)
    save_config()


if __name__ == "__main__":
    main()
