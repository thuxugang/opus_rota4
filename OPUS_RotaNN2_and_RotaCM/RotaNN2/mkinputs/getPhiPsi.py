# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 09:13:16 2016

@author: XuGang
"""

from RotaNN2.mkinputs import vector
from RotaNN2.mkinputs import structure

class DihedralsInfo:
    def __init__(self, resid, resname, phi, psi):
        self.resname = structure.triResname(resname)
        self.resid = resid
        self.pp = [phi, psi]

def calculate_phipsi(residue_before, residue, residue_after):
    
    if(residue_before == None):
        phi = -60
    else:
        phi = vector.calc_dihedral(residue_before.atoms["C"].position, residue.atoms["N"].position, residue.atoms["CA"].position, residue.atoms["C"].position )
    
    if(residue_after == None):
        psi = 60
    else:
        psi = vector.calc_dihedral(residue.atoms["N"].position, residue.atoms["CA"].position, residue.atoms["C"].position, residue_after.atoms["N"].position )
        
    return phi,psi
        
def calculate_dihedral(residue_before, residue, residue_after):
    
    phi,psi = calculate_phipsi(residue_before, residue, residue_after)
    dihedral = DihedralsInfo(residue.resid, residue.resname, phi, psi)
    
    return dihedral

def getDihedrals(residuesData):
    
    dihedralsData = []
    
    seq = 0
    for residue in residuesData:
        #first
        if(seq == 0):
            dihedral = calculate_dihedral(None, residuesData[seq], residuesData[seq+1])
        #last
        elif(seq == len(residuesData)-1):
            dihedral = calculate_dihedral(residuesData[seq-1], residuesData[seq], None)
        else:
            dihedral = calculate_dihedral(residuesData[seq-1], residuesData[seq],residuesData[seq+1])
        
        dihedralsData.append(dihedral)
        seq = seq + 1
        
    return dihedralsData      


    

    

