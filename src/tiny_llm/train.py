import json
import math
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from tiny_llm.model import TinyLM

from tiny_llm.config import (
    CONFIG_FILE,
    CONTEXT_SIZE,
    DIMS,
    EVAL_EVERY,
    LEARNING_RATE,
    NUM_HEADS,
    NUM_LAYERS,
    PATIENCE,
    RANDOM_SEED,
    BATCH_SIZE,
    REPORT_EVERY,
    TENSORS_FILE,
    TRAIN_FILE,
    TRAIN_STEPS,
    VALID_FILE,
    VOCAB_SIZE,
    WEIGHT_DECAY,
    create_directories,
)

create_directories()

mx.random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_tokens(
    path: str,
):
    data = np.fromfile(
        path,
        dtype=np.uint32,
    )
    return data.astype(np.int32)


def make_sequences(
    dataset,
):
    window = CONTEXT_SIZE + 1
    count = len(dataset) // window
    trimmed = dataset[: count * window]
    return trimmed.reshape(
        count,
        window,
    )


train_data = load_tokens(TRAIN_FILE)
valid_data = load_tokens(VALID_FILE)

train_sequences = make_sequences(train_data)
valid_sequences = make_sequences(valid_data)

print("train tokens:", len(train_data))
print("valid tokens:", len(valid_data))
print("train sequences:", len(train_sequences))
print("valid sequences:", len(valid_sequences))


def batches(
    sequences,
):
    while True:
        indexes = np.random.permutation(len(sequences))
        for start in range(
            0,
            len(indexes) - BATCH_SIZE + 1,
            BATCH_SIZE,
        ):
            selected = indexes[start : start + BATCH_SIZE]
            yield mx.array(sequences[selected])


model = TinyLM(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    dims=DIMS,
    num_heads=NUM_HEADS,
)

mx.eval(model.parameters())
parameter_count = sum(value.size for _, value in tree_flatten(model.parameters()))
print(f"Parameters: " f"{parameter_count:,}")


def loss_fn(
    model,
    batch,
):
    x = batch[:, :-1]
    y = batch[:, 1:]
    logits = model(x)

    return nn.losses.cross_entropy(
        logits,
        y,
        reduction="mean",
    )


optimizer = optim.AdamW(
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

state = [
    model.state,
    optimizer.state,
]


@partial(
    mx.compile,
    inputs=state,
    outputs=state,
)
def train_step(
    batch,
):
    loss_and_grad = nn.value_and_grad(
        model,
        loss_fn,
    )
    loss, gradients = loss_and_grad(
        model,
        batch,
    )
    optimizer.update(
        model,
        gradients,
    )
    return loss


def evaluate():
    losses = []
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
        losses.append(loss.item())
    return sum(losses) / len(losses)


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
        print(
            f"step={step:6d} "
            f"loss={loss.item():.4f} "
            f"steps/s="
            f"{REPORT_EVERY / elapsed:.2f}"
        )
        start_time = time.perf_counter()

    if step % EVAL_EVERY == 0:
        validation_loss = evaluate()
        perplexity = math.exp(validation_loss)
        print(f"validation " f"loss={validation_loss:.4f} " f"ppl={perplexity:.2f}")

        # early stopping implementation
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            evaluations_without_improvement = 0

            weights = dict(tree_flatten(model.parameters()))

            mx.save_safetensors(
                str(TENSORS_FILE),
                weights,
            )

            print(
                f"Saved new best checkpoint " f"(validation loss={validation_loss:.4f})"
            )
        else:
            evaluations_without_improvement += 1

            # if we have seen PATIENCE counts of worse improvements then just stop
            if evaluations_without_improvement >= PATIENCE:
                print("Early stopping.")
                break

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
