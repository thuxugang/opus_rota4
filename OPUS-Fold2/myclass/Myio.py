# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:32:13 2015

@author: XuGang
"""

import numpy as np    
from myclass import Atoms, Residues

def readPDB(filename):
    f = open(filename,'r')
    atomsDatas = []
    for line in f.readlines():   
        line = line.strip()
        if (line == "" or line[:3] == "TER"):
            break
        else:
            if (line[:4] == 'ATOM' or line[:6] == 'HETATM'):
                atomid = line[6:11].strip()
                name1 = line[11:16].strip()
                resname = line[16:20].strip()
                        
                #B confomation        
                if(len(resname) == 4 and resname[0] != "A"):
                    continue
                
                resid = line[22:27].strip()

                x = line[30:38].strip()
                y = line[38:46].strip()
                z = line[46:54].strip()
                    
                if(name1[0] in ["N","O","C","S"]):
                    position = np.array([float(x), float(y), float(z)], dtype=np.float32)
                    atom = Atoms.Atom(atomid, name1, resname, resid, position)
                    atomsDatas.append(atom)
    f.close()
    return atomsDatas

def readRotaNN(rota_path):
    
    init_rotamers = []
    x1_index = []
    f = open(rota_path)
    for i in f.readlines():
        if i[0] == '#' or i.strip() == "":
            continue
        x1_4 = i.strip().split()[1:5]
        assert len(x1_4) == 4
        x1_index.append(len(init_rotamers))
        init_rotamers.extend([float(j) for j in x1_4 if float(j) != 182])
    f.close() 
    
    return init_rotamers, x1_index

def outputRotaNN(rotamers, residuesData, rota_path):
    
    count = 0
    f = open(rota_path, "w")
    for idx, residue in enumerate(residuesData):
        
        num_rotamers = residue.num_dihedrals
        rotamer = rotamers[count: count+num_rotamers]
        if  num_rotamers == 0:
            f.write(str(idx+1) + " " + str(182) + " " + str(182) + " " + str(182) + " " + str(182) + "\n")
        elif num_rotamers == 1:
            f.write(str(idx+1) + " " + str(rotamer[0].numpy()) + " " + str(182) + " " + str(182) + " " + str(182) + "\n")
        elif  num_rotamers == 2:
            f.write(str(idx+1) + " " + str(rotamer[0].numpy()) + " " + str(rotamer[1].numpy()) + " " + str(182) + " " + str(182) + "\n")
        elif  num_rotamers == 3:
            f.write(str(idx+1) + " " + str(rotamer[0].numpy()) + " " + str(rotamer[1].numpy()) + " " + str(rotamer[2].numpy()) + " " + str(182) + "\n")
        elif  num_rotamers == 4:
            f.write(str(idx+1) + " " + str(rotamer[0].numpy()) + " " + str(rotamer[1].numpy()) + " " + str(rotamer[2].numpy()) + " " + str(rotamer[3].numpy()) + "\n")
                                                                
        count += num_rotamers

    assert count == len(rotamers)
    
    f.close() 

def outputPDB(residuesData, atoms_matrix, atoms_matrix_name, pdb_path):
    
    atom_id = 1
    counter = 0
    f = open(pdb_path, 'w')
    for residue, atom_names in zip(residuesData, atoms_matrix_name):
        for idx, name1 in enumerate(atom_names):
            if residue.resname == "G" and name1 == "CB": 
                counter += 1
                continue
            atom_id2 = atom_id + idx
            string = 'ATOM  '
            id_len = len(list(str(atom_id2)))
            string = string + " "*(5-id_len) + str(atom_id2)
            string = string + " "*2
            name1_len = len(list(name1))
            string = string + name1 + " "*(3-name1_len)
            resname = Residues.triResname(residue.resname)
            resname_len = len(list(resname))
            string = string + " "*(4-resname_len) + resname
            string = string + " "*2
            resid = str(residue.resid)
            resid_len = len(list(resid))
            string = string + " "*(4-resid_len) + str(resid)
            string = string + " "*4
            x = format(atoms_matrix[counter][0],".3f")
            x_len = len(list(x))
            string = string + " "*(8-x_len) + x
            y = format(atoms_matrix[counter][1],".3f")
            y_len = len(list(y))
            string = string + " "*(8-y_len) + y
            z = format(atoms_matrix[counter][2],".3f")        
            z_len = len(list(z))
            string = string + " "*(8-z_len) + z  
            
            f.write(string)
            f.write("\n")
            
            counter += 1
        
        atom_id += residue.num_atoms
        
    assert len(atoms_matrix) == counter
    f.close()