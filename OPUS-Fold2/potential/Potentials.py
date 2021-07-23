# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:46:08 2016

@author: XuGang
"""

from potential import SCTrrPotential

def get_potentials(SCTrr_matrix, atoms_matrix, init_rotamers):
    
    potentials = 0

    dist_potential, omega_potential, theta_potential, phi_potential = \
        SCTrrPotential.cal_TrrPotential(SCTrr_matrix, atoms_matrix)
    potentials += (5*dist_potential + 4*omega_potential + 4*theta_potential + 4*phi_potential)

    print (5*dist_potential.numpy(), 4*omega_potential.numpy(), 4*theta_potential.numpy(), 4*phi_potential.numpy())
    
    return potentials