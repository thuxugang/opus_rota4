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
from RotaNN2.inference_models import test_infer_step
from RotaNN2.inference_utils import InputReader, get_ensemble_ouput, output_results

def run_RotaNN2(preparation_config):

    #==================================Model===================================
    start_time = time.time()
    print ("Run OPUS-RotaNN2...")
    
    start_time = time.time()
    test_reader = InputReader(data_list=preparation_config["filenames"], 
                              batch_size=preparation_config["batch_size"],
                              preparation_config=preparation_config)
    
    total_lens = 0
    for step, filenames_batch in enumerate(test_reader.dataset):

        filenames, x, x_mask, x_trr, inputs_total_len = \
            test_reader.read_file_from_disk(filenames_batch)
        
        total_lens += inputs_total_len
        
        rota_predictions = test_infer_step(x, x_mask, x_trr)
            
        x1_outputs, x2_outputs, x3_outputs, x4_outputs = \
            get_ensemble_ouput("Rota", rota_predictions, x_mask, inputs_total_len)    
        
        assert len(filenames) == len(x1_outputs) == len(x2_outputs) == len(x3_outputs) == len(x4_outputs)
            
        output_results(filenames, x1_outputs, x2_outputs, x3_outputs, x4_outputs, preparation_config)
        
    run_time = time.time() - start_time
    print('OPUS-RotaNN2 done..., time: %3.3f' % (run_time)) 
    #==================================Model===================================
    
    
    