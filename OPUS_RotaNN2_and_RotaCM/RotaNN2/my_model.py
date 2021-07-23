# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import os
import tensorflow as tf
from RotaNN2.my_transformer import Transformer, create_padding_mask
from RotaNN2.my_rnn import BiRNN_Rota, MLP
from RotaNN2.my_cnn import ResNet
from RotaNN2.my_cnn2 import TRRosettaCNN_ATT

def clean_inputs(x, x_mask, dim_input):
    x_mask = tf.tile(x_mask[:,:,tf.newaxis], [1, 1, dim_input])
    x_clean = tf.where(tf.math.equal(x_mask, 0), x, x_mask-1)
    return x_clean

class Model(object):
    
    def __init__(self, params, name, att=True):
        
        self.params = params
        self.name = name
        
        self.transformer = Transformer(num_layers=self.params["transfomer_layers"],
                                       d_model=self.params["d_input"],
                                       num_heads=self.params["transfomer_num_heads"],
                                       rate=self.params["dropout_rate"])

        self.cnn = ResNet()
        self.mlp = MLP(num_layers=512,
                       rate=0.5)
        
        self.trr_cnn = TRRosettaCNN_ATT(filter_num=128, 
                                        num_layers=32, 
                                        dropout=0.5)

        self.birnn = BiRNN_Rota(num_layers=self.params["lstm_layers"],
                              units=self.params["lstm_units"],
                              rate=self.params["dropout_rate"],
                              rota_output=self.params["d_rota_output"])

    def inference(self, x, x_mask, x_trr, y, y_mask, training):

        encoder_padding_mask = create_padding_mask(x_mask)
        
        x_trr = self.trr_cnn(x_trr, training=training)
        
        x_1d = x[:,:,:41]
        assert x_1d.shape[-1] == 41
        
        x_3d_cnn = x[:,:,41:]
        assert x_3d_cnn.shape[-1] == 13500
        
        x_3d_cnn = self.mlp(x_3d_cnn)
        
        x = tf.concat([x_1d, x_3d_cnn, x_trr], -1)
        
        x = clean_inputs(x, x_mask, self.params["d_input"])
        
        transformer_out = self.transformer(x, encoder_padding_mask, training=training)
        cnn_out = self.cnn(x, training=training)
        x = tf.concat((x, cnn_out, transformer_out), -1)
        
        x = clean_inputs(x, x_mask, 3*self.params["d_input"])
        
        rota_predictions = \
            self.birnn(x, x_mask, training=training) 
            
        return rota_predictions
                
    def load_model(self):
        print ("load model:", self.name)
        self.transformer.load_weights(os.path.join(self.params["save_path"], self.name + '_trans_model_weight'))
        self.cnn.load_weights(os.path.join(self.params["save_path"], self.name + '_cnn_model_weight'))
        self.trr_cnn.load_weights(os.path.join(self.params["save_path"], self.name + '_trr_cnn_model_weight'))
        self.birnn.load_weights(os.path.join(self.params["save_path"], self.name + '_birnn_model_weight'))
        self.mlp.load_weights(os.path.join(self.params["save_path"], self.name + '_mlp_model_weight'))





