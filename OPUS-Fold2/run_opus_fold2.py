# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:11:13 2016

@author: XuGang
"""

import os
from myclass import Residues, Myio
from buildprotein import RebuildStructure
from potential import SCTrrPotential, Potentials
import time
import tensorflow as tf
import numpy as np
import multiprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = ""

def run_script(multi_iter):
    
    bb_path, sctrr_cons_path, rota_init_path, outputs = multi_iter
    
    if not os.path.exists(outputs):
        init_rotamers, _ = Myio.readRotaNN(rota_init_path)
        num_rotamers = len(init_rotamers)
    
        # ############################## get main chain ##############################            
        atomsData_real = Myio.readPDB(bb_path)
        atomsData_mc = RebuildStructure.extractmc(atomsData_real)
        residuesData_mc = Residues.getResidueData(atomsData_mc) 
    
        assert num_rotamers == sum([i.num_dihedrals for i in residuesData_mc]) 
        num_atoms = sum([i.num_side_chain_atoms for i in residuesData_mc]) + 5*len(residuesData_mc)
        num_atoms_real = sum([i.num_atoms for i in residuesData_mc])
        
        geosData = RebuildStructure.getGeosData(residuesData_mc)
        
        residuesData_mc = RebuildStructure.rebuild_cb(residuesData_mc, geosData)
        # ############################## get main chain ##############################  
    
        init_atoms_matrix = np.zeros((num_atoms, 3)).astype(np.float32) 
        init_atoms_matrix  = RebuildStructure.make_atoms_matrix(residuesData_mc, init_atoms_matrix)
    
        sctrr_cons = SCTrrPotential.readTrrCons(sctrr_cons_path)
        SCTrr_matrix = SCTrrPotential.init_Trr_matrix(residuesData_mc, sctrr_cons, dihedral_index=0)

        init_rotamers = [tf.Variable(i) for i in init_rotamers]
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1.0,
            decay_steps=300,
            decay_rate=0.96,
            staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        tolerants = 20
        best_potential = 1e6
        best_torsions = None
        for epoch in range(500):
            with tf.GradientTape() as tape:
                start = time.time()
                for rotamer in init_rotamers:
                    if rotamer > 180:
                        rotamer.assign_sub(360)
                    elif rotamer < -180:
                        rotamer.assign_add(360) 
                        
                atoms_matrix, _ = RebuildStructure.rebuild(init_rotamers, residuesData_mc, geosData, init_atoms_matrix)
                assert init_atoms_matrix.shape[0] == len(atoms_matrix)
                
                loss = Potentials.get_potentials(SCTrr_matrix, atoms_matrix, init_rotamers)
                
                print ("Epoch:", epoch, loss.numpy())
    
                print (time.time() - start)
    
            gradients = tape.gradient(loss, init_rotamers, unconnected_gradients="zero")
            
            optimizer.apply_gradients(zip(gradients, init_rotamers))
    
            if loss.numpy() < best_potential:
                best_torsions = init_rotamers
                best_potential = loss.numpy()
                tolerants = 20
            else:
                tolerants -= 1
                
        Myio.outputRotaNN(best_torsions, residuesData_mc, outputs + ".rota4")
        
        atoms_matrix, atoms_matrix_name = RebuildStructure.rebuild(best_torsions, residuesData_mc, geosData, init_atoms_matrix)
        Myio.outputPDB(residuesData_mc, atoms_matrix, atoms_matrix_name, outputs + ".pdb")
        
if __name__ == '__main__':

    lists = []
    f = open(r'../datasets/list_cameo_hard61', 'r')
    # f = open(r'../datasets/list_caspfm56', 'r')
    # f = open(r'../datasets/list_casp14', 'r')
    for i in f.readlines():
        lists.append(i.strip())
    f.close()    

    multi_iters = []
    for filename in lists:
        
        bb_path = os.path.join("../datasets/bb_data", 
                               filename + ".native_bb")
        
        sctrr_cons_path = os.path.join("../OPUS_RotaNN2_and_RotaCM/predictions", 
                                     filename + ".rotacm.npz")
        
        rota_init_path = os.path.join("../OPUS_RotaNN2_and_RotaCM/predictions", 
                                      filename + ".rotann2")

        outputs = "./predictions/" + filename
        
        multi_iters.append([bb_path, sctrr_cons_path, rota_init_path, outputs])

    pool = multiprocessing.Pool(30)
    pool.map(run_script, multi_iters)
    pool.close()
    pool.join()  

    



        
    













    

                