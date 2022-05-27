"""
Machine learning models to use on Pneumonia dataset.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, models, activations

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def vision_transformer(
    input_shape: tuple, patch_size: int, num_patches: int, projection_dim: int, transformer_layers: int,
    num_heads: int, mlp_heads_units: list, num_classes: int, is_binary: bool=True
    ):
    """Creates ViT (Vision Transformer) neural network.

    Args:
        - input_shape: shape ot the input images (height, width, num_channels)
        - patch_size: height/width of the single patch
        - num_patches: total number of patches to be extracted from the input image
        - projection_dim: dimensionality of the patches embedding projection (size of each attention head)
        - transformer_layers: number of Transformers layers
        - num_heads: number of attention heads
        - mlp_heads_units: units of final MLP head (list of two integres)
        - num_classes: number of classes (labels) to base prediction on
        - is_binary: whether the classification is binary or not

        *** in paper: embedding dimension = num_heads * projection_dim ***
    Returns:
        - model: TensorFlow functional model
    """
    inputs = layers.Input(shape=input_shape)
    patches = Patches(patch_size=patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches=num_patches, projection_dim=projection_dim)(patches)

    # Create Transformers layers
    #for layer in range(transformer_layers):
    for layer in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-4)(encoded_patches)
        x2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        x3 = layers.Add()([x2, encoded_patches])
        x4 = layers.LayerNormalization(epsilon=1e-4)(x3)
        x5 = layers.Dense(units=projection_dim*2, activation=activations.gelu)(x4)
        x6 = layers.Dropout(0.1)(x5)
        x7 = layers.Dense(units=projection_dim, activation=activations.gelu)(x6)
        x8 = layers.Dropout(0.1)(x7)
        encoded_patches = layers.Add()([x8, x3])

    # Projection/representation in latent space
    representation = layers.LayerNormalization(epsilon=1e-4)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    # MLP head
    x = layers.Dense(units=mlp_heads_units[0], activation=activations.gelu)(representation)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=mlp_heads_units[1], activation=activations.gelu)(x)
    x = layers.Dropout(0.5)(x)

    # Classification
    if is_binary == True:
        outputs = layers.Dense(units=1, activation="sigmoid")(x)
    elif is_binary == False:
        outputs = layers.Dense(units=num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model