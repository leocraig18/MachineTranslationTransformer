import os
import pathlib
import shutil
import random
import tensorflow as tf
from tensorflow import keras
from keras import layers

class PositionalEmbedding(layers.Layer):
  def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
    super().__init__(**kwargs)
    self.token_embeddings = layers.Embedding(
        input_dim=input_dim, output_dim=output_dim)
    self.position_embeddings = layers.Embedding(
        input_dim=sequence_length, output_dim=output_dim)
    self.sequence_length = sequence_length
    self.input_dim = input_dim
    self.output_dim = output_dim

  def call(self, inputs):
    length = tf.shape(inputs)[-1]
    positions = tf.range(start=0, limit=length, delta=1)
    embedded_tokens = self.token_embeddings(inputs)
    embedded_positions = self.position_embeddings(positions)
    return embedded_tokens + embedded_positions

  def compute_mask(self, inputs, mask=None):
    return tf.math.not_equal(inputs, 0)

  def get_config(self):
    config = super().get_config()
    config.update({
        "output_dim": self.output_dim,
        "sequence_length": self.sequence_length,
        "input_dim": self.input_dim,
    })
    return config

# Transformer Decoder:


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(
            inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config

class TransformerDecoder(layers.Layer):
  def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
    super().__init__(**kwargs)
    self.embed_dim = embed_dim
    self.dense_dim = dense_dim
    self.num_heads = num_heads
    self.attention_1 = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim)
    self.attention_2 = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim)
    self.dense_proj = keras.Sequential([
        layers.Dense(dense_dim, activation="relu"),
        layers.Dense(embed_dim),
    ])
    self.layernorm_1 = layers.LayerNormalization()
    self.layernorm_2 = layers.LayerNormalization()
    self.layernorm_3 = layers.LayerNormalization()
    self.supports_masking = True

  def get_config(self):
    config = super.get_config()
    config.update({
        "embed_dim": self.embed_dim,
        "num_heads": self.bnum_heads,
        "dense_dim": self.dense_dim
    })
    return config

  def get_causal_attention_mask(self, inputs):
    '''In this step two tensors are created, both i and j will start as an array containing values [0,1,2,3... len(sequence)-1 ]
    The difference is that i i reshaped to contain an extra axis making it a [sequence_length, 1] matrix.
    This results in j being expanded in order to perform the element wise comparison with i in the next step.
    The next variable, mask results from casting i against j with the condition i>=j.
    This results in a lower triangular matrix needed for masking during multi head attention in the decoder.
    The mask is then reshaped allowing it to be compatible with the batch dimension when computing attention scores.
'''
    input_shape = tf.shape(inputs)
    batch_size, sequence_length = input_shape[0], input_shape[1]
    i = tf.range(sequence_length)[:, tf.newaxis]
    j = tf.range(sequence_length)
    mask = tf.cast(i >= j, dtype="int32")
    mask = tf.reshape(mask, (1, sequence_length, sequence_length))
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1),
         tf.constant([1, 1], dtype=tf.int32)], axis=0
    )  # Results in a [8,1,1] tensor
    return tf.tile(mask, mult)  # [8,sequence_length, sequence_length] tensor

  def call(self, inputs, encoder_outputs, mask=None):
    causal_mask = self.get_causal_attention_mask(inputs)
    if mask is not None:
      padding_mask = tf.cast(
          mask[:, tf.newaxis, :], dtype='int32')
      padding_mask = tf.minimum(padding_mask, causal_mask)
    attention_output_1 = self.attention_1(
        query=inputs,
        value=inputs,
        key=inputs,
        attention_mask=padding_mask,
    )
    attention_output_1 = self.layernorm_1(inputs + attention_output_1)
    attention_output_2 = self.attention_2(
        query=attention_output_1,
        value=encoder_outputs,
        key=encoder_outputs,
        attention_mask=padding_mask,
    )
    attention_output_2 = self.attention_2(
        query=attention_output_1,
        value=encoder_outputs,
        key=encoder_outputs,
        attention_mask=padding_mask,
    )
    attention_output_2 = self.layernorm_2(
        attention_output_1 + attention_output_2
    )
    proj_output = self.dense_proj(attention_output_2)
    out = self.layernorm_3(attention_output_2 + proj_output)
    return out