import keras
from keras import ops
import numpy as np

from keras_rs.src.api_export import keras_rs_export


@keras_rs_export("keras_rs.layers.HSTUPositionalEncoder")
class HSTUPositionalEncoder(keras.layers.Layer):
    """Adds sinusoidal positional encodings to the input.

    Args:
        sequence_length: The maximum length of the input sequence.
        embedding_dim: The dimensionality of the embeddings.
    """

    def __init__(self, sequence_length: int, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

        # The positional encoding matrix is created once and reused.
        # It's not a trainable weight.
        position = np.arange(self.sequence_length)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, self.embedding_dim, 2)
            * -(np.log(10000.0) / self.embedding_dim)
        )
        pe = np.zeros((self.sequence_length, self.embedding_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        # Add a batch dimension and make it a non-trainable attribute.
        self.positional_encoding = ops.cast(pe[np.newaxis, ...], self.compute_dtype)


    def call(self, inputs):
        # inputs shape: (batch_size, sequence_length, embedding_dim)
        # Add the positional encoding to the input tensor.
        # The slice ensures that we only use the part of the encoding
        # that corresponds to the input sequence length.
        input_shape = ops.shape(inputs)
        return inputs + self.positional_encoding[:, : input_shape[1], :]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "embedding_dim": self.embedding_dim,
            }
        )
        return config
