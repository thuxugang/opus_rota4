# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import os
import warnings
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import time
import multiprocessing

from RotaNN2.inference_utils import mk_fasta_pp, mk_ss, mk_csf15, mk_trr100, make_input
from RotaNN2.DLPacker_OPUS import utils, dlpacker
from RotaNN2.inference import run_RotaNN2
from RotaCM.inference import run_RotaCM

def preparation(multi_iter):
    
    file_path, filename, preparation_config = multi_iter
   
    fasta_filename = filename + '.fasta'
    pp_filename = filename + '.pp'
    if (not os.path.exists(os.path.join(preparation_config["tmp_files_path"], fasta_filename)) or
        not os.path.exists(os.path.join(preparation_config["tmp_files_path"], pp_filename))):
        mk_fasta_pp(file_path, filename, preparation_config)
        
    ss_filename = filename + '.ss'
    if not os.path.exists(os.path.join(preparation_config["tmp_files_path"], ss_filename)):
        mk_ss(file_path, filename, preparation_config)  

    csf15_filename = filename + '.csf15.npy'
    if not os.path.exists(os.path.join(preparation_config["tmp_files_path"], csf15_filename)):
        mk_csf15(file_path, filename, preparation_config)  

    trr100_filename = filename + '.trr100.npy'
    if not os.path.exists(os.path.join(preparation_config["tmp_files_path"], trr100_filename)):
        mk_trr100(file_path, filename, preparation_config)  

    make_input(filename, preparation_config)
        
if __name__ == '__main__':

    #============================Parameters====================================
    list_path = r"./data_bb"
    files_path = []
    f = open(list_path)
    for i in f.readlines():
        files_path.append(i.strip())
    f.close()
    
    preparation_config = {}
    preparation_config["batch_size"] = 1
    preparation_config["tmp_files_path"] = os.path.join(os.path.abspath('.'), "tmp_files")
    preparation_config["output_path"] = os.path.join(os.path.abspath('.'), "predictions")
    preparation_config["mkdssp_path"] = os.path.join(os.path.abspath('.'), "RotaNN2/mkdssp/mkdssp")
    
    num_cpu = 56
    
    #============================Parameters====================================
    
    
    #============================Preparation===================================
    print('Preparation start...')
    start_time = time.time()
    
    multi_iters = []
    filenames = []
    for file_path in files_path:
        filename = file_path.split('/')[-1].split('.')[0]
        multi_iters.append([file_path, filename, preparation_config])
        filenames.append(filename)
        
    pool = multiprocessing.Pool(num_cpu)
    pool.map(preparation, multi_iters)
    pool.close()
    pool.join()

    preparation_config["filenames"] = filenames
        
    run_time = time.time() - start_time
    print('Preparation done..., time: %3.3f' % (run_time))  
    #============================Preparation===================================

    #============================DLPacker(OPUS)===================================
    print('Run DLPacker(OPUS)...')
    start_time = time.time()
    
    model1 = utils.DLPModel(width = 128, nres = 6)
    model1.load_model(weights = "./RotaNN2/DLPacker_OPUS/models/1/3dcnn.h5")
    model2 = utils.DLPModel(width = 128, nres = 6)
    model2.load_model(weights = "./RotaNN2/DLPacker_OPUS/models/2/3dcnn.h5")
    model3 = utils.DLPModel(width = 128, nres = 6)
    model3.load_model(weights = "./RotaNN2/DLPacker_OPUS/models/3/3dcnn.h5")
    model4 = utils.DLPModel(width = 128, nres = 6)
    model4.load_model(weights = "./RotaNN2/DLPacker_OPUS/models/4/3dcnn.h5")
    model5 = utils.DLPModel(width = 128, nres = 6)
    model5.load_model(weights = "./RotaNN2/DLPacker_OPUS/models/5/3dcnn.h5")  
    model6 = utils.DLPModel(width = 128, nres = 6)
    model6.load_model(weights = "./RotaNN2/DLPacker_OPUS/models/6/3dcnn.h5")  
    model7 = utils.DLPModel(width = 128, nres = 6)
    model7.load_model(weights = "./RotaNN2/DLPacker_OPUS/models/7/3dcnn.h5") 

    models = [model1, model2, model3, model4, model5, model6, model7]
    for file_path in files_path:
        filename = file_path.split('/')[-1].split('.')[0]
        dlpacker_filename = filename + '.3dcnn.npy'
        if not os.path.exists(os.path.join(preparation_config["tmp_files_path"], dlpacker_filename)):
            seq_len = np.loadtxt((os.path.join(preparation_config["tmp_files_path"], filename + ".rotann2_1d_inputs"))).shape[0]
            dlp = dlpacker.DLPacker(file_path, models=models)
            dlp.reconstruct_protein(order='sequence', seq_len=seq_len, 
                                    output_path=os.path.join(preparation_config["tmp_files_path"], filename + '.3dcnn'))  
            
    run_time = time.time() - start_time
    print('DLPacker(OPUS) done..., time: %3.3f' % (run_time))  
    #============================DLPacker===================================
    
    #============================OPUS-RotaNN2===============================
    run_RotaNN2(preparation_config)
    #============================OPUS-RotaNN2===============================
    
    #============================OPUS-RotaCM================================
    run_RotaCM(preparation_config)
    #============================OPUS-RotaCM================================
    