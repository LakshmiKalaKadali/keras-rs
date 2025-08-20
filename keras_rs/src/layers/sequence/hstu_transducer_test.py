import numpy as np

from keras_rs.src.layers.sequence.hstu_transducer import HSTUTransducer
from keras_rs.src.testing.test_case import TestCase


class TestHSTUTransducer(TestCase):
    def test_layer_output_shape(self):
        batch_size = 2
        sequence_length = 8
        embedding_dim = 16

        # Create dummy input data
        input_data = np.random.rand(
            batch_size, sequence_length, embedding_dim
        )

        # Create layer instance
        layer = HSTUTransducer(
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            num_heads=4,
            ff_dim=32,
            num_stu_layers=2,
        )

        # Get layer output
        output_data = layer(input_data)

        # Assert output shape is correct
        self.assertEqual(output_data.shape, input_data.shape)
