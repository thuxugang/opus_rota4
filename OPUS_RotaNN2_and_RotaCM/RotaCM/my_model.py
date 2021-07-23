# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import os
import tensorflow as tf
from RotaCM.my_cnn import TRRosettaCNN, MLP

class Model(object):
    
    def __init__(self, params, name):
        
        self.params = params
        self.name = name

        self.cnn = TRRosettaCNN(filter_num=self.params["filter_num"], 
                                num_layers=self.params["num_layers"], 
                                dropout=self.params["dropout"])

        self.mlp = MLP(num_layers=256,
                       rate=0.5)
        
    def inference(self, x1d, x2d, y=None, y_mask=None, training=False):

        """
        "theta",0,25
        "phi",25,38
        "dist",38,75
        "omega",75,100
        """
        length = x1d.shape[1]
        x_1d = x1d[:,:,:41]
        assert x_1d.shape == (1, length, 41)
        
        x_3d_cnn = x1d[:,:,41:]
        assert x_3d_cnn.shape == (1, length, 13500)
        
        x_3d_cnn = self.mlp(x_3d_cnn, training=training)

        x_1d = tf.concat([x_1d, x_3d_cnn], -1)
        assert x_1d.shape == (1, length, 41+256)
        
        x_1d = x_1d[0]
        x_1d = tf.concat([tf.tile(x_1d[:,None,:], [1,length,1]), 
                tf.tile(x_1d[None,:,:], [length,1,1])], axis=-1)
        x_1d = tf.expand_dims(x_1d, axis=0)
        
        x = tf.concat([x_1d, x2d],-1)
        assert x.shape == (1, length, length, 709)
       
        logits = {}
        logits_theta, logits_phi, logits_dist, logits_omega = self.cnn(x, training=training)
        
        logits["theta"]= logits_theta
        logits["phi"] = logits_phi
        logits["dist"] = logits_dist
        logits["omega"] = logits_omega
        
        return logits

    def load_model(self):
        print ("load model:", self.name)
        self.cnn.load_weights(os.path.join(self.params["save_path"], self.name + '_model_weight'))
        self.mlp.load_weights(os.path.join(self.params["save_path"], self.name + '_mlp_model_weight'))





