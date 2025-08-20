import keras
from keras import layers

from keras_rs.src.api_export import keras_rs_export
from keras_rs.src.layers.sequence.hstu_positional_encoder import (
    HSTUPositionalEncoder,
)
from keras_rs.src.layers.sequence.stu_layer import STULayer


@keras_rs_export("keras_rs.layers.HSTUTransducer")
class HSTUTransducer(layers.Layer):
    """The main transducer for the HSTU model.

    This layer combines the positional encoder and a stack of STU layers to
    process a sequence of embeddings.

    Args:
        sequence_length: The maximum length of the input sequence.
        embedding_dim: The dimensionality of the input and output.
        num_heads: The number of attention heads in each STU layer.
        ff_dim: The dimensionality of the feed-forward network in each STU
            layer.
        num_stu_layers: The number of STU layers to stack.
        dropout_rate: The dropout rate to apply.
    """

    def __init__(
        self,
        sequence_length: int,
        embedding_dim: int,
        num_heads: int,
        ff_dim: int,
        num_stu_layers: int,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_stu_layers = num_stu_layers
        self.dropout_rate = dropout_rate

        self.pos_encoder = HSTUPositionalEncoder(
            sequence_length=sequence_length, embedding_dim=embedding_dim
        )
        self.stu_layers = [
            STULayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate,
                name=f"stu_layer_{i}",
            )
            for i in range(num_stu_layers)
        ]
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        x = self.pos_encoder(inputs)
        x = self.dropout(x, training=training)
        for stu_layer in self.stu_layers:
            x = stu_layer(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "num_stu_layers": self.num_stu_layers,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
