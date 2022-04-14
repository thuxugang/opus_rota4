# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:41:48 2016

@author: XuGang
"""

import numpy as np
import os
from Bio.PDB import PDBParser, Selection, Superimposer, PDBIO, Atom, Residue, Structure

import warnings
warnings.filterwarnings("ignore")

def singleResname(AA):
    if(len(AA) == 1):
        return AA
    else:
        if(AA in ['GLY','AGLY']):
            return "G"
        elif(AA in ['ALA','AALA']):
            return "A"
        elif(AA in ['SER','ASER']):
            return "S"
        elif(AA in ['CYS','ACYS']):
            return "C"
        elif(AA in ['VAL','AVAL']):
            return "V"
        elif(AA in ['ILE','AILE']):
            return "I"
        elif(AA in ['LEU','ALEU']):
            return "L"
        elif(AA in ['THR','ATHR']):
            return "T"
        elif(AA in ['ARG','AARG']):
            return "R"
        elif(AA in ['LYS','ALYS']):
            return "K"
        elif(AA in ['ASP','AASP']):
            return "D"
        elif(AA in ['GLU','AGLU']):
            return "E"
        elif(AA in ['ASN','AASN']):
            return "N"
        elif(AA in ['GLN','AGLN']):
            return "Q"
        elif(AA in ['MET','AMET']):
            return "M"
        elif(AA in ['HIS','AHIS']):
            return "H"
        elif(AA in ['PRO','APRO']):
            return "P"
        elif(AA in ['PHE','APHE']):
            return "F"
        elif(AA in ['TYR','ATYR']):
            return "Y"
        elif(AA in ['TRP','ATRP']):
            return "W"
        else:
            print ("Unknown residue")
            
def get_order_atoms(resname, atom_dict):
    
    resname = singleResname(resname)
    results = []
    results.append(atom_dict["N"])
    results.append(atom_dict["C"])
    results.append(atom_dict["CA"])
    results.append(atom_dict["O"])
    results.append(atom_dict["CB"])
    
    if (resname == "V"):
        results.append(atom_dict["CG1"])

    elif (resname == "I"):
        results.append(atom_dict["CG1"])
        try:
            results.append(atom_dict["CD"])
        except:
            results.append(atom_dict["CD1"])
            
    elif (resname == "L"):
        results.append(atom_dict["CG"])
        results.append(atom_dict["CD1"])
        
    elif (resname == "S"):
        results.append(atom_dict["OG"])
        
    elif (resname == "T"):
        results.append(atom_dict["OG1"])
        
    elif (resname == "D"):
        results.append(atom_dict["CG"])
        results.append(atom_dict["OD1"])
        results.append(atom_dict["OD2"]) # 180
        
    elif (resname == "N"):
        results.append(atom_dict["CG"])
        results.append(atom_dict["OD1"])        

    elif (resname == "E"):
        results.append(atom_dict["CG"])
        results.append(atom_dict["CD"])         
        results.append(atom_dict["OE1"])         
        results.append(atom_dict["OE2"]) # 180
        
    elif (resname == "Q"):
        results.append(atom_dict["CG"])
        results.append(atom_dict["CD"])         
        results.append(atom_dict["OE1"])         

    elif (resname == "K"):
        results.append(atom_dict["CG"])
        results.append(atom_dict["CD"])         
        results.append(atom_dict["CE"])  
        results.append(atom_dict["NZ"])  

    elif (resname == "R"):
        results.append(atom_dict["CG"])
        results.append(atom_dict["CD"])         
        results.append(atom_dict["NE"])  
        results.append(atom_dict["CZ"]) 

    elif (resname == "C"):
        results.append(atom_dict["SG"])
        
    elif (resname == "M"):
        results.append(atom_dict["CG"])
        try:
            results.append(atom_dict["SD"])         
            results.append(atom_dict["CE"])  
        except:
            results.append(atom_dict["SE"])         
            results.append(atom_dict["CE"])          

    elif (resname == "F"):
        results.append(atom_dict["CG"])
        results.append(atom_dict["CD1"]) 
        results.append(atom_dict["CD2"]) # 180
        
    elif (resname == "Y"):
        results.append(atom_dict["CG"])
        results.append(atom_dict["CD1"])         
        results.append(atom_dict["CD2"]) # 180
        
    elif (resname == "W"):
        results.append(atom_dict["CG"])
        results.append(atom_dict["CD1"])         

    elif (resname == "H"):
        results.append(atom_dict["CG"])
        results.append(atom_dict["ND1"])  

    elif (resname == "P"):
        results.append(atom_dict["CG"])
        results.append(atom_dict["CD"]) 
        
    return results

def _remove_hydrogens(structure:Structure):
    # Removes all hydrogens.
    # This code is not suited to work with hydrogens
    for residue in Selection.unfold_entities(structure, 'R'):
        remove = []
        for atom in residue:
            if atom.element == 'H': remove.append(atom.get_id())
            if atom.name == 'OXT': remove.append(atom.get_id())
        for i in remove: residue.detach_child(i)

def _convert_mse(structure:Structure):
    # Changes MSE residues to MET
    for residue in Selection.unfold_entities(structure, 'R'):
        if residue.get_resname() == 'MSE':
            residue.resname = 'MET'
            for atom in residue:
                if atom.element == 'SE':
                    new_atom = Atom.Atom('SD',\
                                         atom.coord,\
                                         atom.bfactor,\
                                         atom.occupancy,\
                                         atom.altloc,\
                                         'SD  ',\
                                         atom.serial_number,\
                                         element='S')
                    residue.add(new_atom)
                    atom_to_remove = atom.get_id()
                    residue.detach_child(atom_to_remove)

def _remove_water(structure:Structure):
    # Removes all water molecules
    residues_to_remove = []
    for residue in Selection.unfold_entities(structure, 'R'):
        if residue.get_resname() == 'HOH':
            residues_to_remove.append(residue)
    for r in residues_to_remove:
        r.get_parent().detach_child(r.get_id())
    
if __name__ == "__main__":

    pdb_lists = []
    f = open(r'./list_casp14', 'r')
    for i in f.readlines():
        pdb_lists.append(i.strip())
    f.close()
    
    native_path = r"./native_structures"
    predict_path = r'./native_testsets/Rota4'    
    
    totals = 0
    sucess = 0
    rmsds1 = []
    for filename in pdb_lists:
        
        parser = PDBParser(PERMISSIVE = 1)
        
        native = parser.get_structure('structure', os.path.join(native_path, filename+".pdb"))
        pred = parser.get_structure('structure', os.path.join(predict_path, filename + ".pdb"))
           
        _remove_hydrogens(native) # we never use hydrogens
        _convert_mse(native)      # convers MSE to MET
        _remove_water(native)     # waters are not used anyway
            
        _remove_hydrogens(pred) # we never use hydrogens
        _convert_mse(pred)      # convers MSE to MET
        _remove_water(pred)     # waters are not used anyway
        
        count = 0
        residue_ps = [residue_p for residue_p in Selection.unfold_entities(pred, 'R')]
        for residue_n in Selection.unfold_entities(native, 'R'):
            for residue_p in residue_ps:
                if residue_p.get_id()[1] == residue_n.get_id()[1]:
                    assert residue_n.get_resname() == residue_p.get_resname()
                    if residue_n.get_resname() in ["GLY", "ALA"]: continue
                    totals += 1
                    try:
                        n = list(residue_n.get_atoms())
                        p = list(residue_p.get_atoms())

                        n_dict = {}
                        for atom in n:
                            n_dict[atom.name] = atom
                        p_dict = {}
                        for atom in p:
                            p_dict[atom.name] = atom                                

                        n = get_order_atoms(residue_n.get_resname(), n_dict)
                        p = get_order_atoms(residue_p.get_resname(), p_dict)
                        
                        n_name = "".join([i.name for i in n])
                        p_name = "".join([i.name for i in p])
                        assert n_name == p_name
                        
                        if residue_n.get_resname() in ["ASP", "GLU", "PHE", "TYR"]:
                            sup = Superimposer()
                            sup.set_atoms(n[:-1], p[:-1])
                            rms1 = sup.rms

                            sup = Superimposer()
                            sup.set_atoms(n[:-1], p[:-2]+p[-1:])
                            rms2 = sup.rms
                        
                            rmsds1.append(min(rms1, rms2))   
                            sucess += 1
                        else:
                            sup = Superimposer()
                            sup.set_atoms(n, p)
                            rmsds1.append(sup.rms) 
                            sucess += 1                                
                        
                    except:
                        pass
            count += 1
            
    print(np.mean(rmsds1), len(rmsds1), sucess, totals, sucess/totals)
