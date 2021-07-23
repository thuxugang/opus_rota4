# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import tensorflow as tf

class FirstBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(FirstBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.downsample = tf.keras.layers.Conv2D(filters=filter_num,
                                                 kernel_size=(1, 1),
                                                 strides=stride)
            
    def call(self, inputs, training=None, **kwargs):
        
        residual = self.downsample(inputs)
        
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output
    
class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        
        residual = inputs
        
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output

def make_basic_block_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(FirstBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=stride))

    return res_block

class ResNet(tf.keras.Model):
    def __init__(self, layer_params=[3,3,3,3]):
        super(ResNet, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=8,
                                            kernel_size=(5, 5),
                                            padding="same")
        
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.layer1 = make_basic_block_layer(filter_num=16,
                                             blocks=layer_params[0])
        
        self.layer2 = make_basic_block_layer(filter_num=32,
                                             blocks=layer_params[1])
        
        self.layer3 = make_basic_block_layer(filter_num=16,
                                             blocks=layer_params[2])
        
        self.layer4 = make_basic_block_layer(filter_num=8,
                                             blocks=layer_params[3])

    def call(self, x, training):
        
        x = tf.expand_dims(x, -1)
        
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        
        cnn_output = tf.reduce_mean(x, axis=-1)
        
        return cnn_output