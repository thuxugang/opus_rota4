# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 15:03:06 2020

@author: xugang
"""

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

class FirstBlock(keras.layers.Layer):

    def __init__(self, filter_num: int, kernel_size: int):
        
        super(FirstBlock, self).__init__()
        
        self.conv = keras.layers.Conv2D(filters=filter_num,
                                        kernel_size=kernel_size,
                                        padding="SAME")
        self.norm = tfa.layers.normalizations.InstanceNormalization()

    def call(self, inputs):
        
        inputs = self.conv(inputs)
        inputs = self.norm(inputs)
        inputs = tf.nn.elu(inputs)
        
        return inputs

class MyConv2d(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding, dilation_rate):
        
        super(MyConv2d, self).__init__()
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation_rate = [1, dilation_rate, dilation_rate,1]
    
    def build(self, input_shape):
        
        self.kernel = self.add_variable(
                name="my_kernel",
                shape=(self.kernel_size, self.kernel_size, self.filters, self.filters),
                initializer=tf.keras.initializers.GlorotUniform(),
                trainable=True,
                )
        self.bias = self.add_variable(
                name="my_bias",
                shape=(self.filters,),
                initializer=tf.zeros_initializer(),
                trainable=True,
        )
        self.built = True
        
    def call(self, inputs):
        return tf.compat.v1.nn.conv2d(inputs, filter=self.kernel, strides=[1,1,1,1], padding=self.padding,
                  dilations=self.dilation_rate) + self.bias
    
class DilatedResidualBlock(keras.layers.Layer):
    def __init__(self, filter_num: int, kernel_size: int, dilation: int, dropout: float):
        
        super(DilatedResidualBlock, self).__init__()

        self.conv1 = MyConv2d(filters=filter_num,
                              kernel_size=kernel_size,
                              padding="SAME",
                              dilation_rate=dilation)        
        self.norm1 = tfa.layers.normalizations.InstanceNormalization()
        
        self.dropout = keras.layers.Dropout(dropout)
        
        self.conv2 = MyConv2d(filters=filter_num,
                              kernel_size=kernel_size,
                              padding="SAME",
                              dilation_rate=dilation)           
        self.norm2 = tfa.layers.normalizations.InstanceNormalization()
        
    def call(self, inputs, training):
        
        shortcut = inputs
        
        inputs = self.conv1(inputs)
        inputs = self.norm1(inputs)
        inputs = tf.nn.elu(inputs)
        
        inputs = self.dropout(inputs, training=training)
        
        inputs = self.conv2(inputs)
        inputs = self.norm2(inputs)
        inputs = tf.nn.elu(inputs + shortcut)
        
        return inputs

def make_basic_block_layer(filter_num=64, num_layers=61, dropout=0.5):
    
    res_block = keras.Sequential()
    res_block.add(FirstBlock(filter_num=filter_num, kernel_size=5))
    
    dilation = 1
    for _ in range(num_layers):
        res_block.add(DilatedResidualBlock(filter_num=filter_num, kernel_size=3, dilation=dilation, dropout=dropout))
        dilation *= 2
        if dilation > 16:
            dilation = 1
            
    return res_block

class Attention(keras.layers.Layer):

    def __init__(self, attention_size=128):

        super(Attention, self).__init__()
        
        self.attention_size = attention_size

        self.v_t = keras.layers.Dense(attention_size, name='att_vt',activation='tanh')


    def build(self, input_shape):
        
        self.w = self.add_variable(
            name="att_w",
            initializer=tf.keras.initializers.GlorotUniform(),
            shape=(self.attention_size,),
            trainable=True)
        
        self.built = True

    def call(self, inputs):
        
        v = self.v_t(inputs)
        
        vu = tf.tensordot(v, self.w, axes=1)

        alpha = tf.nn.softmax(vu)
        
        output = tf.reduce_sum(inputs * tf.expand_dims(alpha,-1), 1)

        return output
    
class TRRosettaCNN_ATT(keras.Model):

    def __init__(self, filter_num=64, num_layers=61, dropout=0.5):
        
        super(TRRosettaCNN_ATT, self).__init__()

        self.filter_num = filter_num
        self.feature_layer = make_basic_block_layer(filter_num=filter_num,
                                                    num_layers=num_layers,
                                                    dropout=dropout)
        self.att = Attention()
        
    def call(self, x_trr, training):
        
        x_trr = self.feature_layer(x_trr, training=training)

        x_trr = tf.squeeze(x_trr, 0)
        
        x_trr = self.att(x_trr)
        x_trr = tf.expand_dims(x_trr, 0)
        
        return x_trr    
    
    
