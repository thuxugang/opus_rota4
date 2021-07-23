# -*- coding: utf-8 -*-
"""
Created on Wed May 11 08:43:50 2016

@author: Xu Gang
"""

import numpy as np
from RotaNN2.mkinputs import Geometry, vector

def transCoordinate(atom_ca_ref, atom_c_ref, atom_o_ref, atom_c):
    
    ref = atom_ca_ref
    c_ref_new = atom_c_ref - ref
    o_ref_new = atom_o_ref - ref
    c_new = atom_c - ref
  
    x_axis = c_ref_new/np.linalg.norm(c_ref_new)
    
    c_o = o_ref_new - c_ref_new
    
    y_axis = c_o - (x_axis.dot(c_o)/x_axis.dot(x_axis) * x_axis)
    y_axis = y_axis/np.linalg.norm(y_axis)

    z_axis = np.cross(x_axis,y_axis)
    
    rotation_matrix = np.array([x_axis[0],y_axis[0],z_axis[0],0,x_axis[1],y_axis[1],z_axis[1],0,x_axis[2],y_axis[2],z_axis[2],0,0,0,0,1]).reshape(4,4)

    new = np.array([c_new[0],c_new[1],c_new[2],1]).dot(rotation_matrix)

    return np.array([new[0],new[1],new[2]])

def get_cb(residue):

    if residue.resname != 'G':
        geo = Geometry.geometry(residue.resname)
        CB = vector.calculateCoordinates(
            residue.atoms["C"], residue.atoms["N"], residue.atoms["CA"], geo.CA_CB_length, geo.C_CA_CB_angle, geo.N_C_CA_CB_diangle)    
    else:
        CB = residue.atoms["CA"].position
    
    return CB

def get_contactlist(residuesData):

    length = len(residuesData)
    csf_cm15 = np.zeros((length, length, 15))
    
    for i in range(length):
        residue_a = residuesData[i]
        a_cb = get_cb(residue_a)
        for j in range(length):
            if i == j:
                continue
            residue_b = residuesData[j]
            b_cb = get_cb(residue_b)
            
            bb_distance = np.linalg.norm(a_cb - b_cb)
            
            if bb_distance < 10:     
                csf_cm15[i,j,:3] = transCoordinate(residue_a.atoms["CA"].position, residue_a.atoms["C"].position, residue_a.atoms["O"].position, \
                                                            residue_b.atoms["N"].position)            
                csf_cm15[i,j,3:6] = transCoordinate(residue_a.atoms["CA"].position, residue_a.atoms["C"].position, residue_a.atoms["O"].position, \
                                                            residue_b.atoms["CA"].position)
                csf_cm15[i,j,6:9] = transCoordinate(residue_a.atoms["CA"].position, residue_a.atoms["C"].position, residue_a.atoms["O"].position, \
                                                            residue_b.atoms["C"].position)    
                csf_cm15[i,j,9:12] = transCoordinate(residue_a.atoms["CA"].position, residue_a.atoms["C"].position, residue_a.atoms["O"].position, \
                                                            residue_b.atoms["O"].position)   
                if residue_b.resname != 'G':
                    csf_cm15[i,j,12:] = transCoordinate(residue_a.atoms["CA"].position, residue_a.atoms["C"].position, residue_a.atoms["O"].position, \
                                                        b_cb)
                        
    csf_cm15 = np.around(csf_cm15, decimals=2) 
    
    return csf_cm15    