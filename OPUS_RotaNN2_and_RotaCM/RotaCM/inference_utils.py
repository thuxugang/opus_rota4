# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import os
import tensorflow as tf
import numpy as np

#=============================================================================    
def read_inputs(filenames, preparation_config):
    """
    (7pc + 19psp + 8ss + 3ss + 4pp)*2 + trr100 + csf15
    """
    assert len(filenames) == 1
    inputs1ds = []
    inputs2ds = []
    lengths = []
    for filename in filenames:
        
        inputs_1d = np.loadtxt((os.path.join(preparation_config["tmp_files_path"], filename + ".rotann2_1d_inputs")))
        length = inputs_1d.shape[0]

        csf15 = np.load(os.path.join(preparation_config["tmp_files_path"], filename + ".csf15.npy"))
        trr100 = np.load(os.path.join(preparation_config["tmp_files_path"], filename + ".trr100.npy"))
        
        inputs_dlpacker = np.load(os.path.join(preparation_config["tmp_files_path"], filename + ".3dcnn.npy"))
        inputs_1d = np.concatenate((inputs_1d, inputs_dlpacker[:,:13500]),-1)  
        
        assert inputs_1d.shape == (length, 41+13500)
        
        inputs_2d = np.concatenate((trr100, csf15),-1)
        assert inputs_2d.shape == (length, length, 115)
        
        inputs1ds.append(inputs_1d)
        inputs2ds.append(inputs_2d)
        lengths.append(length)
        
    return np.array(inputs1ds), np.array(inputs2ds), np.array(lengths)


class InputReader(object):

    def __init__(self, data_list, batch_size, preparation_config):

        self.data_list = data_list
        self.preparation_config = preparation_config
        self.dataset = tf.data.Dataset.from_tensor_slices(self.data_list).batch(batch_size)          
        
        print ("Data Size:", len(self.data_list)) 
    
    def read_file_from_disk(self, filenames_batch):
        
        filenames_batch = [bytes.decode(i) for i in filenames_batch.numpy()]
        inputs1d_batch, inputs2d_batch, inputs_len = \
            read_inputs(filenames_batch, self.preparation_config)
        
        inputs1d_batch = tf.convert_to_tensor(inputs1d_batch, dtype=tf.float32)
        inputs2d_batch = tf.convert_to_tensor(inputs2d_batch, dtype=tf.float32)
        inputs_len = tf.convert_to_tensor(inputs_len, dtype=tf.int32)
        
        return filenames_batch, inputs1d_batch, inputs2d_batch, inputs_len
            
#=============================================================================    
def get_ensemble_ouput(name, logits_predictions):
    
    if name == "TrRosetta":
            
        trrosetta_outputs = {}

        for key in ["theta", "phi", "dist", "omega"]:
            
            softmax_prediction = tf.nn.softmax(logits_predictions[0][key])
            tmp = [softmax_prediction.numpy()]
            
            if len(logits_predictions) > 1:
                for i in logits_predictions[1:]:
                    tmp.append(
                        tf.nn.softmax(i[key]).numpy())
            
            trrosetta_outputs[key] = np.mean(tmp, axis=0)

        return trrosetta_outputs
    
def output_results(filename, trrosetta_outputs, preparation_config, inputs_len):
    
    inputs_len = inputs_len.numpy()[0]
    filename = filename[0]
    
    assert trrosetta_outputs['dist'].shape == (1, inputs_len, inputs_len, 37)
    assert trrosetta_outputs['omega'].shape == (1, inputs_len, inputs_len, 25)
    assert trrosetta_outputs['theta'].shape == (1, inputs_len, inputs_len, 25)
    assert trrosetta_outputs['phi'].shape == (1, inputs_len, inputs_len, 13)

    np.savez_compressed(os.path.join(preparation_config["output_path"], filename + ".rotacm"), 
                        dist=trrosetta_outputs['dist'][0].astype(np.float32), 
                        omega=trrosetta_outputs['omega'][0].astype(np.float32), 
                        theta=trrosetta_outputs['theta'][0].astype(np.float32), 
                        phi=trrosetta_outputs['phi'][0].astype(np.float32))
    