# -*- coding: utf-8 -*-
"""
Created on Sat May 30 07:14:18 2015

@author: XuGang
"""

from myclass import Residues


class Atom:
    def __init__(self, atomid, name1, resname, resid, position):
        self.atomid = atomid
        self.name1 = name1
        self.resname = Residues.singleResname(resname)
        self.resid = resid
        self.position = position
        
        if self.name1 in ['N','CA','C','O','CB']:
            self.ismainchain = True
        else:
            self.ismainchain = False 
