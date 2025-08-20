import numpy as np
import keras

from keras_rs.src.layers.sequence.hstu_positional_encoder import (
    HSTUPositionalEncoder,
)
from keras_rs.src.testing.test_case import TestCase


class TestHSTUPositionalEncoder(TestCase):
    def test_layer_adds_encoding(self):
        batch_size = 2
        sequence_length = 10
        embedding_dim = 16

        # Create dummy input data (zeros)
        input_data = np.zeros(
            (batch_size, sequence_length, embedding_dim), dtype="float32"
        )

        # Create layer instance
        layer = HSTUPositionalEncoder(
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
        )

        # Get layer output
        output_data = layer(input_data)

        # Assert output shape is correct
        self.assertEqual(output_data.shape, input_data.shape)

        # Assert that the output is not the same as the input (encoding was added)
        self.assertNotAllClose(output_data, input_data)

        # Assert that the first element in the sequence has a non-zero encoding
        self.assertNotAllClose(output_data[:, 0, :], 0)
