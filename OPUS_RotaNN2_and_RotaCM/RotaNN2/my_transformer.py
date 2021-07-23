# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import tensorflow as tf
from tensorflow import keras


def create_padding_mask(x_mask):
    return x_mask[:, tf.newaxis, tf.newaxis, :]


def scaled_dot_product_attention(q, k, v, mask):

    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(keras.layers.Layer):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert self.d_model % self.num_heads == 0

        self.depth = self.d_model // self.num_heads

        self.WQ = keras.layers.Dense(self.d_model, activation='relu')
        self.WK = keras.layers.Dense(self.d_model, activation='relu')
        self.WV = keras.layers.Dense(self.d_model, activation='relu')

        self.dense = keras.layers.Dense(self.d_model, activation='relu')

    def split_heads(self, x, batch_size):

        x = tf.reshape(x,
                       (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        q = self.WQ(q)
        k = self.WK(k)
        v = self.WV(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention_outputs, attention_weights = \
            scaled_dot_product_attention(q, k, v, mask)

        scaled_attention_outputs = tf.transpose(
            scaled_attention_outputs, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention_outputs,
                                      (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights


class EncoderLayer(keras.layers.Layer):

    def __init__(self, d_model, num_heads, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.layer_norm1 = keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)

    def call(self, x, training, encoder_padding_mask):

        attn_output, attn_weights = self.mha(x, x, x, encoder_padding_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(x + attn_output)

        return out1, attn_weights


class EncoderModel(keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, rate=0.1):
        super(EncoderModel, self).__init__()

        self.num_layers = num_layers

        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, rate)
            for _ in range(self.num_layers)]

    def call(self, encoder_input, encoder_padding_mask, training):

        attention_weights = {}
        x = encoder_input
        for i in range(self.num_layers):
            x, attn = self.encoder_layers[i](x, training,
                                             encoder_padding_mask)
            attention_weights[
                'encoder_layer{}_att'.format(i+1)] = attn

        return x, attention_weights


class Transformer(keras.Model):
    def __init__(self, num_layers, d_model, num_heads, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder_model = EncoderModel(
            num_layers, d_model, num_heads, rate)

    def call(self, encoder_input, encoder_padding_mask, training):

        encoding_outputs, attention_weights = self.encoder_model(
            encoder_input, encoder_padding_mask, training)

        return encoding_outputs
