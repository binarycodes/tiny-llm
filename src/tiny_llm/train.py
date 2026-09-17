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
    SOURCE_WEIGHTS,
    TENSORS_FILE,
    TOKEN_DTYPE,
    TRAIN_STEPS,
    VOCAB_SIZE,
    WEIGHT_DECAY,
    Split,
    create_directories,
    tokenized_files,
)
from tiny_llm.model import TinyLM

type TokenArray = np.ndarray[tuple[int], np.dtype[TOKEN_DTYPE]]
type SequenceArray = np.ndarray[tuple[int, int], np.dtype[TOKEN_DTYPE]]
type TrainStep = Callable[[mx.array], mx.array]
type Sources = dict[str, Split[SequenceArray]]


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


def source_weights() -> dict[str, float]:
    if not SOURCE_WEIGHTS or any(weight <= 0 for weight in SOURCE_WEIGHTS.values()):
        raise RuntimeError("SOURCE_WEIGHTS must name at least one source and every weight must be positive")
    total = sum(SOURCE_WEIGHTS.values())
    return {source: weight / total for source, weight in SOURCE_WEIGHTS.items()}


def load_data() -> Sources:
    sources: Sources = {}
    for source in SOURCE_WEIGHTS:
        paths = tokenized_files(source)
        train_data = load_tokens(paths.training)
        valid_data = load_tokens(paths.validation)
        sources[source] = Split(make_sequences(train_data), make_sequences(valid_data))

        print(f"{source}: train tokens:", len(train_data))
        print(f"{source}: valid tokens:", len(valid_data))
        print(f"{source}: train sequences:", len(sources[source].training))
        print(f"{source}: valid sequences:", len(sources[source].validation))

    return sources


def sequence_stream(
    sequences: SequenceArray,
) -> Iterator[TokenArray]:
    while True:
        for index in np.random.permutation(len(sequences)):
            yield sequences[index]


def batches(
    sources: Sources,
) -> Iterator[mx.array]:
    weights = source_weights()
    names = list(weights)
    probabilities = np.array([weights[name] for name in names])
    streams = {name: sequence_stream(sources[name].training) for name in names}

    while True:
        chosen = np.random.choice(names, size=BATCH_SIZE, p=probabilities)
        yield mx.array(np.stack([next(streams[name]) for name in chosen]))


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


def evaluate_source(
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


def evaluate(
    model: TinyLM,
    sources: Sources,
) -> float:
    """Validation loss per source, combined with the sampling weights so it matches the training objective."""
    weights = source_weights()
    combined = 0.0
    for source, weight in weights.items():
        loss = evaluate_source(model, sources[source].validation)
        print(f"  {source}: validation loss={loss:.4f} perplexity={math.exp(loss):.2f}")
        combined += weight * loss
    return combined


def train(
    model: TinyLM,
    sources: Sources,
) -> None:
    optimizer = optim.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    train_step, state = make_train_step(model, optimizer)

    train_batches = batches(sources)

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
            validation_loss = evaluate(model, sources)
            perplexity = math.exp(validation_loss)
            print(f"weighted validation loss={validation_loss:.4f} perplexity={perplexity:.2f}")

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

    sources = load_data()
    model = build_model()
    train(model, sources)
    save_config()


if __name__ == "__main__":
    main()
