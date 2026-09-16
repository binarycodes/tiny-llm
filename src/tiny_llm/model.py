import mlx.core as mx
import mlx.nn as nn


class TinyLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dims: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.dims = dims
        self.embedding = nn.Embedding(
            vocab_size,
            dims,
        )
        self.position = nn.SinusoidalPositionalEncoding(dims)
        self.transformer = nn.TransformerEncoder(
            num_layers,
            dims,
            num_heads,
            norm_first=True,
        )
        self.output = nn.Linear(
            dims,
            vocab_size,
            bias=False,
        )

    def __call__(
        self,
        tokens: mx.array,
    ) -> mx.array:
        length = tokens.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(length)
        x = self.embedding(tokens)
        positions = mx.arange(length)
        x = x + self.position(positions)
        x = self.transformer(x, mask)
        return self.output(x)
