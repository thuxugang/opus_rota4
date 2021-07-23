# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""
import warnings
import tensorflow as tf

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import time
from RotaCM.inference_utils import InputReader, get_ensemble_ouput, output_results
from RotaCM.inference_models import test_infer_step

def run_RotaCM(preparation_config):
    
    #==================================Model===================================
    start_time = time.time()
    print ("Run OPUS-RotaCM...")
    start_time = time.time()
    test_reader = InputReader(data_list=preparation_config["filenames"], 
                              batch_size=preparation_config["batch_size"],
                              preparation_config=preparation_config)
    
    for step, filenames_batch in enumerate(test_reader.dataset):

        filenames, x1d, x2d, inputs_len = test_reader.read_file_from_disk(filenames_batch)
        
        logits_predictions = test_infer_step(x1d, x2d)
            
        trrosetta_outputs = get_ensemble_ouput("TrRosetta", logits_predictions)            
            
        output_results(filenames, trrosetta_outputs, preparation_config, inputs_len)
        
    run_time = time.time() - start_time
    print('OPUS-RotaCM done..., time: %3.3f' % (run_time)) 
    #==================================Model===================================
    
    
    