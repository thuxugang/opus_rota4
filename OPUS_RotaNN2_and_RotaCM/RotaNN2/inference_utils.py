# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import os
import tensorflow as tf
import numpy as np
from RotaNN2.mkinputs import PDBreader, structure, vector, Geometry, getPhiPsi, mkCSF15, mkTRR100
    
ss8 = "CSTHGIEB"
ss8_dict = {}
for k,v in enumerate(ss8):
    ss8_dict[v] = k
    
def get_psp_dict():
    resname_to_psp_dict = {}
    resname_to_psp_dict['G'] = [1,4,7]
    resname_to_psp_dict['A'] = [1,3,7]
    resname_to_psp_dict['V'] = [1,7,12]
    resname_to_psp_dict['I'] = [1,3,7,12]
    resname_to_psp_dict['L'] = [1,5,7,12]
    resname_to_psp_dict['S'] = [1,2,5,7]
    resname_to_psp_dict['T'] = [1,7,15]
    resname_to_psp_dict['D'] = [1,5,7,11]
    resname_to_psp_dict['N'] = [1,5,7,14]
    resname_to_psp_dict['E'] = [1,6,7,11]
    resname_to_psp_dict['Q'] = [1,6,7,14]
    resname_to_psp_dict['K'] = [1,5,6,7,10]
    resname_to_psp_dict['R'] = [1,5,6,7,13]
    resname_to_psp_dict['C'] = [1,7,8]
    resname_to_psp_dict['M'] = [1,6,7,9]
    resname_to_psp_dict['F'] = [1,5,7,16]
    resname_to_psp_dict['Y'] = [1,2,5,7,16]
    resname_to_psp_dict['W'] = [1,5,7,18]
    resname_to_psp_dict['H'] = [1,5,7,17]
    resname_to_psp_dict['P'] = [7,19]
    return resname_to_psp_dict

def get_pc7_dict():
    resname_to_pc7_dict = {'A': [-0.350, -0.680, -0.677, -0.171, -0.170, 0.900, -0.476],
                'C': [-0.140, -0.329, -0.359, 0.508, -0.114, -0.652, 0.476],
                'D': [-0.213, -0.417, -0.281, -0.767, -0.900, -0.155, -0.635],
                'E': [-0.230, -0.241, -0.058, -0.696, -0.868, 0.900, -0.582],
                'F': [ 0.363, 0.373, 0.412, 0.646, -0.272, 0.155, 0.318],
                'G': [-0.900, -0.900, -0.900, -0.342, -0.179, -0.900, -0.900],
                'H': [ 0.384, 0.110, 0.138, -0.271, 0.195, -0.031, -0.106],
                'I': [ 0.900, -0.066, -0.009, 0.652, -0.186, 0.155, 0.688],
                'K': [-0.088, 0.066, 0.163, -0.889, 0.727, 0.279, -0.265],
                'L': [ 0.213, -0.066, -0.009, 0.596, -0.186, 0.714, -0.053],
                'M': [ 0.110, 0.066, 0.087, 0.337, -0.262, 0.652, -0.001],
                'N': [-0.213, -0.329, -0.243, -0.674, -0.075, -0.403, -0.529],
                'P': [ 0.247, -0.900, -0.294, 0.055, -0.010, -0.900, 0.106],
                'Q': [-0.230, -0.110, -0.020, -0.464, -0.276, 0.528, -0.371],
                'R': [ 0.105, 0.373, 0.466, -0.900, 0.900, 0.528, -0.371],
                'S': [-0.337, -0.637, -0.544, -0.364, -0.265, -0.466, -0.212],
                'T': [ 0.402, -0.417, -0.321, -0.199, -0.288, -0.403, 0.212],
                'V': [ 0.677, -0.285, -0.232, 0.331, -0.191, -0.031, 0.900],
                'W': [ 0.479, 0.900, 0.900, 0.900, -0.209, 0.279, 0.529],
                'Y': [ 0.363, 0.417, 0.541, 0.188, -0.274, -0.155, 0.476]}
    return resname_to_pc7_dict

resname_to_psp_dict = get_psp_dict()
resname_to_pc7_dict = get_pc7_dict()
    
def mk_fasta_pp(file_path, filename, preparation_config):
    
    fasta_path = os.path.join(preparation_config["tmp_files_path"], filename +'.fasta')
    pp_path = os.path.join(preparation_config["tmp_files_path"], filename +'.pp')
    
    atomsData = PDBreader.readPDB(file_path) 
    residuesData = structure.getResidueData(atomsData) 
    dihedralsData = getPhiPsi.getDihedrals(residuesData)
    
    fasta = "".join([i.resname for i in residuesData])
    assert len(fasta) == len(dihedralsData)
    
    f = open(fasta_path, 'w')
    f.writelines(">" + file_path.split('/')[-1] + "\n")
    f.writelines(fasta)
    f.close()   
    
    pps = []
    for i in dihedralsData:
        pps.append([np.sin(np.deg2rad(i.pp[0])), np.cos(np.deg2rad(i.pp[0])), 
                    np.sin(np.deg2rad(i.pp[1])), np.cos(np.deg2rad(i.pp[1]))])
    f.close()
    
    np.savetxt(pp_path, np.array(pps), fmt="%.4f")

def mk_ss(file_path, filename, preparation_config):
    
    ss_path = os.path.join(preparation_config["tmp_files_path"], filename +'.ss')
   
    cmd = preparation_config["mkdssp_path"] + ' ' + file_path
    print (cmd) 
    
    output = os.popen(cmd).read()
    ss = []
    for i in output.split("\n"):
        if i != "" and i[0] != '#':
            ss.append(i.strip().split()[2].strip())

    f = open(ss_path, 'w')
    f.writelines("".join(ss))
    f.close()

def mk_csf15(file_path, filename, preparation_config):
    
    csf15_path = os.path.join(preparation_config["tmp_files_path"], filename +'.csf15')
    
    atomsData = PDBreader.readPDB(file_path) 
    residuesData = structure.getResidueData(atomsData) 

    csf_cm15 = mkCSF15.get_contactlist(residuesData)
    np.save(csf15_path, csf_cm15.astype(np.float16))  
    
def mk_trr100(file_path, filename, preparation_config):

    trr100_path = os.path.join(preparation_config["tmp_files_path"], filename +'.trr100')
    
    with open(os.path.join(preparation_config["tmp_files_path"], filename +'.fasta'),'r') as r:
        results = [i.strip() for i in r.readlines()]
    
    seq = results[1]
    length = len(seq)
    
    atomsData = PDBreader.readPDB(file_path) 
    residuesData = structure.getResidueData(atomsData) 
    res_seq = [i.resname for i in residuesData]
    
    assert ''.join(seq) == ''.join(res_seq)
        
    dist_ref, omega_ref, theta_ref, phi_ref = \
        np.zeros((length, length)), np.zeros((length, length)), np.zeros((length, length)), np.zeros((length, length))
        
    for i in range(length):
        residue_a = residuesData[i]
        a_ca = residue_a.atoms["CA"].position
        a_n = residue_a.atoms["N"].position
        if "CB" in residue_a.atoms:
            a_cb = residue_a.atoms["CB"].position
        else:
            if residue_a.resname == 'G':
                res_name = 'A'
            else:
                res_name = residue_a.resname
            geo = Geometry.geometry(res_name)
            a_cb = vector.calculateCoordinates(
                residue_a.atoms["C"], residue_a.atoms["N"], residue_a.atoms["CA"], geo.CA_CB_length, geo.C_CA_CB_angle, geo.N_C_CA_CB_diangle)
            
        for j in range(length):
            if i == j:
                continue
            residue_b = residuesData[j]
            b_ca = residue_b.atoms["CA"].position
            if "CB" in residue_b.atoms:
                b_cb = residue_b.atoms["CB"].position
            else:
                if residue_b.resname == 'G':
                    res_name = 'A'
                else:
                    res_name = residue_b.resname
                geo = Geometry.geometry(res_name)
                b_cb = vector.calculateCoordinates(
                    residue_b.atoms["C"], residue_b.atoms["N"], residue_b.atoms["CA"], geo.CA_CB_length, geo.C_CA_CB_angle, geo.N_C_CA_CB_diangle)

            dist_ref[i][j] = np.linalg.norm(a_cb - b_cb)
            omega_ref[i][j] = np.deg2rad(vector.calc_dihedral(a_ca, a_cb, b_cb, b_ca))
            theta_ref[i][j] = np.deg2rad(vector.calc_dihedral(a_n, a_ca, a_cb, b_cb))
            phi_ref[i][j] = np.deg2rad(vector.calc_angle(a_ca, a_cb, b_cb))

    p_dist  = mkTRR100.mtx2bins(dist_ref,     2.0,  20.0, 37, mask=(dist_ref > 20))
    p_omega = mkTRR100.mtx2bins(omega_ref, -np.pi, np.pi, 25, mask=(p_dist[...,0]==1))
    p_theta = mkTRR100.mtx2bins(theta_ref, -np.pi, np.pi, 25, mask=(p_dist[...,0]==1))
    p_phi   = mkTRR100.mtx2bins(phi_ref,      0.0, np.pi, 13, mask=(p_dist[...,0]==1))
    feat    = np.concatenate([p_theta, p_phi, p_dist, p_omega],-1)
    
    assert feat.shape == (length, length, 100)
    
    np.save(trr100_path, feat.astype(np.int8))
    
def read_ss(path):

    with open(path,'r') as f:
        ss_result = [i for i in f.readlines()]
    ss_result = ss_result[0]
    ss_len = len(ss_result)
    
    ss8 = np.zeros(ss_len*8).reshape(ss_len, 8)
    for i in range(ss_len):
        ss8[i][ss8_dict[ss_result[i]]] = 1
        
    ss3 = np.zeros((ss_len, 3))
    ss3[:,0] = np.sum(ss8[:,:3],-1)
    ss3[:,1] = np.sum(ss8[:,3:6],-1)
    ss3[:,2] = np.sum(ss8[:,6:8],-1)
    
    return ss8, ss3
    
def make_input(filename, preparation_config):
    """
    7pc + 19psp + 8ss + 3ss + 4pp = 41
    """    
    
    fasta_path = os.path.join(preparation_config["tmp_files_path"], filename+'.fasta')
    with open(fasta_path,'r') as f:
        fasta_result = [i for i in f.readlines()]
    fasta = fasta_result[1]
    
    seq_len = len(fasta)
    
    pp_path = os.path.join(preparation_config["tmp_files_path"], filename+'.pp')
    ss_path = os.path.join(preparation_config["tmp_files_path"], filename+'.ss')
    input_path = os.path.join(preparation_config["tmp_files_path"], filename+'.rotann2_1d_inputs')
    
    ss8, ss3 = read_ss(ss_path)
    
    pc7 = np.zeros((seq_len, 7))
    for i in range(seq_len):
        pc7[i] = resname_to_pc7_dict[fasta[i]]
    
    psp = np.zeros((seq_len, 19))
    for i in range(seq_len):
        psp19 = resname_to_psp_dict[fasta[i]]
        for j in psp19:
            psp[i][j-1] = 1

    pp = np.loadtxt(pp_path)
    
    inputs_1d = np.concatenate((pc7, psp, ss8, ss3, pp),axis=1)
    assert inputs_1d.shape == (seq_len, 41)
    
    np.savetxt(input_path, inputs_1d, fmt="%.4f")

#=============================================================================    
def read_inputs(filenames, preparation_config):
    """
    7pc + 19psp + 8ss + 3ss + 4pp / trr100 + csf15
    """
    inputs_1ds = []
    inputs_2ds = []
    max_len = 0
    inputs_total_len = 0

    assert len(filenames) == 1
    for filename in filenames:
        
        inputs_1d = np.loadtxt((os.path.join(preparation_config["tmp_files_path"], filename + ".rotann2_1d_inputs")))
        length = inputs_1d.shape[0]

        csf15 = np.load(os.path.join(preparation_config["tmp_files_path"], filename + ".csf15.npy"))
        trr100 = np.load(os.path.join(preparation_config["tmp_files_path"], filename + ".trr100.npy"))

        inputs_dlpacker = np.load(os.path.join(preparation_config["tmp_files_path"], filename + ".3dcnn.npy"))
        inputs_1d = np.concatenate((inputs_1d, inputs_dlpacker[:,:13500]),-1)  
        
        assert inputs_1d.shape == (length, 41 + 13500)
        
        inputs_2d = np.concatenate((trr100, csf15),-1)
        assert inputs_2d.shape == (length, length, 115)
        
        inputs_total_len = length
        max_len = inputs_total_len
        
        inputs_1ds.append(inputs_1d)
        inputs_2ds.append(inputs_2d)
        
        inputs_1ds = np.array(inputs_1ds)
        inputs_2ds = np.array(inputs_2ds)
    
    inputs_mask = np.zeros(shape=(len(filenames), max_len))

    return inputs_1ds, inputs_mask, inputs_2ds, inputs_total_len


class InputReader(object):

    def __init__(self, data_list, batch_size, preparation_config):

        self.data_list = data_list
        self.preparation_config = preparation_config
        self.dataset = tf.data.Dataset.from_tensor_slices(self.data_list).batch(batch_size)          
        
        print ("Data Size:", len(self.data_list)) 
    
    def read_file_from_disk(self, filenames_batch):
        
        filenames_batch = [bytes.decode(i) for i in filenames_batch.numpy()]
        inputs_1ds_batch, inputs_masks_batch, inputs_2ds_batch, inputs_total_len = \
            read_inputs(filenames_batch, self.preparation_config)
        
        inputs_1ds_batch = tf.convert_to_tensor(inputs_1ds_batch, dtype=tf.float32)
        inputs_masks_batch= tf.convert_to_tensor(inputs_masks_batch, dtype=tf.float32)
        inputs_2ds_batch = tf.convert_to_tensor(inputs_2ds_batch, dtype=tf.float32)
        
        return filenames_batch, inputs_1ds_batch, inputs_masks_batch, inputs_2ds_batch, inputs_total_len
            
#=============================================================================    
def get_ensemble_ouput(name, predictions, x_mask, total_len):
    
    if name == "Rota":
        
        x1_predictions = []
        x2_predictions = []
        x3_predictions = []
        x4_predictions = []
        
        x1_outputs = []
        x2_outputs = []
        x3_outputs = []
        x4_outputs = []
        
        for i in predictions:
            
            i = i.numpy()
            
            x1_prediction = np.zeros((i.shape[0], i.shape[1], 1))
            x2_prediction = np.zeros((i.shape[0], i.shape[1], 1))
            x3_prediction = np.zeros((i.shape[0], i.shape[1], 1))
            x4_prediction = np.zeros((i.shape[0], i.shape[1], 1))

            x1_prediction[:,:,0] = np.rad2deg(np.arctan2(i[:,:,0], i[:,:,1]))
            x2_prediction[:,:,0] = np.rad2deg(np.arctan2(i[:,:,2], i[:,:,3]))
            x3_prediction[:,:,0] = np.rad2deg(np.arctan2(i[:,:,4], i[:,:,5]))
            x4_prediction[:,:,0] = np.rad2deg(np.arctan2(i[:,:,6], i[:,:,7]))
            
            x1_predictions.append(x1_prediction)
            x2_predictions.append(x2_prediction)
            x3_predictions.append(x3_prediction)
            x4_predictions.append(x4_prediction)
        
        x1_predictions = np.concatenate(x1_predictions, -1)
        x1_predictions = np.median(x1_predictions, -1)

        x2_predictions = np.concatenate(x2_predictions, -1)
        x2_predictions = np.median(x2_predictions, -1)
        
        x3_predictions = np.concatenate(x3_predictions, -1)
        x3_predictions = np.median(x3_predictions, -1)
        
        x4_predictions = np.concatenate(x4_predictions, -1)
        x4_predictions = np.median(x4_predictions, -1)
        
        x_mask = x_mask.numpy()
        max_length = x_mask.shape[1]
        for i in range(x_mask.shape[0]):
            indiv_length = int(max_length-np.sum(x_mask[i]))
            x1_outputs.append(x1_predictions[i][:indiv_length])
            x2_outputs.append(x2_predictions[i][:indiv_length])
            x3_outputs.append(x3_predictions[i][:indiv_length])
            x4_outputs.append(x4_predictions[i][:indiv_length])
        
        x1_outputs_concat = np.concatenate(x1_outputs, 0)
        x2_outputs_concat = np.concatenate(x2_outputs, 0)
        x3_outputs_concat = np.concatenate(x3_outputs, 0)
        x4_outputs_concat = np.concatenate(x4_outputs, 0)
        
        assert x1_outputs_concat.shape[0] == x2_outputs_concat.shape[0] == \
            x3_outputs_concat.shape[0] == x4_outputs_concat.shape[0] == total_len        
        
        return x1_outputs, x2_outputs, x3_outputs, x4_outputs
    
dihedral_nums ={"G":0,"A":0,"V":1,"I":2,"L":2,"S":1,"T":1,"D":2,"N":2,"E":3,"Q":3,"K":4,"R":4,"C":1,"M":3,"F":2,"Y":2,"W":2,"H":2,"P":2}
    
def output_results(filenames, x1_outputs, x2_outputs, x3_outputs, x4_outputs, preparation_config):
    
    for filename, x1_output, x2_output, x3_output, x4_output in \
        zip(filenames, x1_outputs, x2_outputs, x3_outputs, x4_outputs):
        
        fasta_path = os.path.join(preparation_config["tmp_files_path"], filename+'.fasta')

        with open(fasta_path, 'r') as r:
            fasta_content = [i.strip() for i in r.readlines()]
        fasta_seq = fasta_content[1]
        
        output_path = os.path.join(preparation_config["output_path"], filename+".rotann2")
        f = open(output_path, 'w')
        f.write("#\tX1\tX2\tX3\tX4\n")
        
        assert x1_output.shape[0] == x2_output.shape[0] == x3_output.shape[0] == x4_output.shape[0] == len(fasta_seq)

        for idx, (res_name, x1, x2, x3, x4) in \
            enumerate(zip(fasta_seq, x1_output, x2_output, x3_output, x4_output)):
            
            null_num = 4-dihedral_nums[res_name]
            if null_num == 4:
                x1 = x2 = x3 = x4 = 182
            elif null_num == 3:
                x2 = x3 = x4 = 182
            elif null_num == 2:
                x3 = x4 = 182
            elif null_num == 1:
                x4 = 182
                
            f.write('%i\t%3.2f\t%3.2f\t%3.2f\t%3.2f\n'%(idx+1, x1, x2, x3, x4))
            
        f.close()
    
    