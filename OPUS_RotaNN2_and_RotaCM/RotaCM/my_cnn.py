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
    
def make_basic_block_layer(filter_num=64, num_layers=41, dropout=0.5):
    
    res_block = keras.Sequential()
    res_block.add(FirstBlock(filter_num=filter_num, kernel_size=5))
    
    dilation = 1
    for _ in range(num_layers):
        res_block.add(DilatedResidualBlock(filter_num=filter_num, kernel_size=3, dilation=dilation, dropout=dropout))
        dilation *= 2
        if dilation > 16:
            dilation = 1
            
    return res_block

def make_basic_block_layer2(filter_num=256, num_layers=21, dropout=0.5):
    
    res_block = keras.Sequential()
    dilation = 1
    for _ in range(num_layers):
        res_block.add(DilatedResidualBlock(filter_num=filter_num, kernel_size=3, dilation=dilation, dropout=dropout))
        dilation *= 2
        if dilation > 16:
            dilation = 1
            
    return res_block

class TRRosettaCNN(keras.Model):

    def __init__(self, filter_num=64, num_layers=61, dropout=0.5):
        
        super(TRRosettaCNN, self).__init__()


        self.feature_layer = make_basic_block_layer(filter_num=filter_num,
                                                     num_layers=num_layers,
                                                     dropout=dropout)
        
        self.predict_theta = keras.layers.Conv2D(filters=25, kernel_size=1, padding='SAME')
        self.predict_phi = keras.layers.Conv2D(filters=13, kernel_size=1, padding='SAME')
        self.predict_dist = keras.layers.Conv2D(filters=37, kernel_size=1, padding='SAME')
        self.predict_omega = keras.layers.Conv2D(filters=25, kernel_size=1, padding='SAME')

    def call(self, x, training):
        
        x = self.feature_layer(x, training=training)
        
        logits_theta = self.predict_theta(x)

        logits_phi = self.predict_phi(x)

        sym_x = 0.5 * (x + tf.transpose(x, perm=[0, 2, 1, 3]))

        logits_dist = self.predict_dist(sym_x)

        logits_omega = self.predict_omega(sym_x)     
 
        return logits_theta, logits_phi, logits_dist, logits_omega
    
    
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
