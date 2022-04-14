# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:41:48 2016

@author: XuGang
"""

import numpy as np
import os

"""
Format of .dihedral
idx RES_TYPE X1 X2 X3 X4

181 denote missing atom for dihedral calculation
182 denote the corresponding X doesn't exist

1 GLU 47.44 -176.19 -5.32 182
2 PHE -67.8 -70.0 182 182
3 ASP -69.65 -6.86 182 182
4 SER -59.33 182 182 182
5 PHE -57.7 -76.18 182 182
6 THR 56.79 182 182 182
7 SER -72.57 182 182 182
8 PRO -4.04 15.41 182 182
9 ASP -172.98 10.66 182 182
10 LEU -172.37 59.87 182 182
11 THR -61.87 182 182 182
12 ASN -71.47 -32.81 182 182
13 GLU -175.91 178.08 21.02 182
"""

if __name__ == "__main__":

    pdb_lists = []
    f = open(r'./list_casp14', 'r')
    for i in f.readlines():
        pdb_lists.append(i.strip())
    f.close()
    
    native_path = r"./native_structures"
    predict_path = r'./native_testsets/Rota4'    
    
    corrects = 0
    incompletes = 0
    totals = 0
    alls = 0
    maes = [[],[],[],[]]
    success = 0
    for filename in pdb_lists:

        results_native = {}
        f = open(os.path.join(native_path, filename + ".dihedrals"))
        for i in f.readlines():
            if i.strip() != "":
                key = i.strip().split()[0] + i.strip().split()[1]
                x1_4 = i.strip().split()[2:]
                assert len(x1_4) == 4
                results_native[key] = [float(j) for j in x1_4]
        f.close()  
        
        results_predict = {}
        f = open(os.path.join(predict_path, filename + ".dihedrals"))
        for i in f.readlines():
            if i.strip() != "":
                key = i.strip().split()[0] + i.strip().split()[1]
                x1_4 = i.strip().split()[2:]
                assert len(x1_4) == 4
                results_predict[key] = [float(j) for j in x1_4]
        f.close()      
        
        results_native_ = {}
        for i in results_native:
            if "MS" in i or "CS" in i:
                continue
            elif "ALA" in i or "GLY" in i:
                continue
            else:
                results_native_[i] = results_native[i]

        results_predict_ = {}
        for i in results_predict:
            if "ALA" in i or "GLY" in i:
                continue
            else:
                results_predict_[i] = results_predict[i]
                
        alls += len(results_native_)
        
        for key in results_native_:
            x1_4_n = results_native_[key]
            x1_4_p = results_predict_[key]
            
            if 181 in x1_4_n:
                incompletes += 1
            else:
                x1_4_n = [i for i in x1_4_n if i != 182]
                x1_4_p = [i for i in x1_4_p if i != 182]
                assert len(x1_4_n) == len(x1_4_p)
                
                if len(x1_4_n) == 0:
                    raise Exception("GLY, ALA error")
                    
                x1_4_n = np.array(x1_4_n)
                x1_4_p = np.array(x1_4_p)
                
                diff = x1_4_n - x1_4_p
                diff[np.where(diff<-180)] += 360
                diff[np.where(diff>180)] -= 360
                mae = diff       
                
                if key[-3:] in ["ASP", "PHE", "TYR"]:
                    if mae[1] > 90:
                        mae[1] -= 180
                    if mae[1] < -90: 
                        mae[1] += 180
                if key[-3:] in ["GLU"]:
                    if mae[2] > 90:
                        mae[2] -= 180
                    if mae[2] < -90: 
                        mae[2] += 180
                mae = np.abs(mae)       

                for i in range(len(mae)):
                    maes[i].append(mae[i])
                    
                if (mae<20).all() == True:
                    corrects += 1
                
                totals += 1
                
        success += 1
          
    print (success, alls, totals, incompletes, corrects, corrects/totals)
    print (len(maes[0]), len(maes[1]), len(maes[2]), len(maes[3]))
    print (np.mean(maes[0]), np.mean(maes[1]), np.mean(maes[2]), np.mean(maes[3]))
