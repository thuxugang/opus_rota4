# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Bidirectional

class BiRNN_Rota(keras.Model):
    def __init__(self, num_layers, units, rate, 
                 rota_output):
        super(BiRNN_Rota, self).__init__()
        
        self.num_layers = num_layers
        
        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]
        
        self.dropout = keras.layers.Dropout(0.5)
        
        self.rota_layer = keras.layers.Dense(rota_output)
        
    def call(self, x, x_mask, training):
        
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))
        
        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)
        
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        
        rota_predictions = self.rota_layer(x)

        return rota_predictions


class MLP(keras.Model):
    def __init__(self, num_layers, rate):
        
        super(MLP, self).__init__()
        
        self.cov3d = keras.layers.Conv3D(1, 3, strides = 1, padding = 'valid', activation = 'elu')
        self.dnn1 = keras.layers.Dense(num_layers)
        self.dropout = keras.layers.Dropout(rate)
        
    def call(self, x, training):
        
        length = tf.shape(x)[1]
        x = tf.reshape(x, (1, length, 15, 15, 15, 4))
        x = tf.squeeze(x, 0)
        x = self.cov3d(x)
        x = tf.reshape(x, (length, 13*13*13))
        x = tf.expand_dims(x, 0)
        x = self.dnn1(x)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        
        return x    