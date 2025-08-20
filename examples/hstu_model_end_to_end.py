"""
# HSTU: A Transformer-based model for sequential recommendation

This example demonstrates how to build and train a Hierarchical Sequential
Transformer Unit (HSTU) model for sequential recommendation tasks. The HSTU
model is a Transformer-based architecture designed to capture sequential
patterns in user interaction data.

We will build the following components:
1. `STULayer`: A Transformer block implementation.
2. `HSTUPositionalEncoder`: A layer to add positional information.
3. `HSTUTransducer`: A layer that stacks STU layers to form the core of the
   sequence model.
4. `HSTUModel`: The final Keras model that brings everything together.

Finally, we will train the model on synthetic data to demonstrate its usage.
"""

import keras
from keras import layers
import numpy as np
import os

# Set a backend for the example.
# We are using `tensorflow` here, but it could be 'jax' or 'torch'.
os.environ["KERAS_BACKEND"] = "tensorflow"

# To run this example, we need to make sure the local `keras_rs` package
# is in the python path.
import sys
sys.path.insert(0, os.getcwd())


# First, we need to import the custom layers we've built.
# Note: For these imports to work, you might need to run the api_gen.py
# script at the root of the repository to update the public API.
try:
    from keras_rs.layers.sequence import HSTUTransducer
except ImportError:
    print("Could not import HSTUTransducer.")
    print("Please run `python api_gen.py` from the root of the repository.")
    sys.exit(1)


# 1. Define the top-level HSTU Model
class HSTUModel(keras.Model):
    """A full sequential recommendation model using HSTU."""
    def __init__(
        self,
        vocab_size,
        sequence_length,
        embedding_dim,
        num_heads,
        ff_dim,
        num_stu_layers,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

        self.token_embedding = layers.Embedding(
            input_dim=vocab_size, output_dim=embedding_dim
        )
        self.transducer = HSTUTransducer(
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_stu_layers=num_stu_layers,
        )
        # A final dense layer to project the output to the vocabulary size
        self.output_layer = layers.Dense(vocab_size, activation="softmax")

    def call(self, inputs):
        # inputs shape: (batch_size, sequence_length)
        x = self.token_embedding(inputs)
        x = self.transducer(x)
        # We take the output of the last timestep to predict the next item.
        last_timestep_output = x[:, -1, :]
        return self.output_layer(last_timestep_output)


# 2. Set up parameters and generate synthetic data
VOCAB_SIZE = 100
SEQUENCE_LENGTH = 10
EMBEDDING_DIM = 32
NUM_HEADS = 4
FF_DIM = 64
NUM_STU_LAYERS = 2
BATCH_SIZE = 16
NUM_SAMPLES = 500

print("--- Generating synthetic data ---")
# Input sequences
x_train = np.random.randint(
    0, VOCAB_SIZE, size=(NUM_SAMPLES, SEQUENCE_LENGTH)
)
# Target labels (predicting the next item in the sequence)
y_train = np.random.randint(0, VOCAB_SIZE, size=(NUM_SAMPLES,))
print("Synthetic data generated.")


# 3. Instantiate, compile, and train the model
print("\n--- Building HSTU Model ---")
hstu_model = HSTUModel(
    vocab_size=VOCAB_SIZE,
    sequence_length=SEQUENCE_LENGTH,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    num_stu_layers=NUM_STU_LAYERS,
)

hstu_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\n--- Model Summary ---")
# Build the model by calling it on a sample batch
hstu_model.build(input_shape=(None, SEQUENCE_LENGTH))
hstu_model.summary()

print("\n--- Training Model ---")
hstu_model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=2, # Just a couple of epochs to demonstrate it works
)

print("\n--- HSTU Model End-to-End Example Complete ---")
