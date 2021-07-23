# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:32:13 2015

@author: XuGang
"""

from RotaNN2.mkinputs import structure
import numpy as np    

def readPDB(filename):
    f = open(filename,'r')
    atomsDatas = []
    for line in f.readlines():   
        if (line == "" or line == "\n" or line[:3] == "TER"):
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
                    atom = structure.Atom(atomid, name1, resname, resid, position)
                    atomsDatas.append(atom)
    f.close()
    return atomsDatas