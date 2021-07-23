'''This module is part of the PeptideBuilder library,
written by Matthew Z. Tien, Dariya K. Sydykova,
Austin G. Meyer, and Claus O. Wilke.

The Geometry module contains the default geometries of
all 20 amino acids. The main function to be used is the
geometry() function, which returns the default geometry
for the requested amino acid.

This file is provided to you under the GNU General Public
License, version 2.0 or later.'''

class Geo():
    '''Geometry base class'''
    def __repr__(self):
        repr = ""
        for var in dir(self):
            if var in self.__dict__: # exclude member functions, only print member variables
                repr += "%s = %s\n" % ( var, self.__dict__[var] )
        return repr


class GlyGeo(Geo):
    '''Geometry of Glycine'''
    def __init__(self):
        self.CA_N_length = 1.46
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 110.8914

        self.C_O_length = 1.23
        self.CA_C_O_angle = 120.5117
        self.N_CA_C_O_diangle = 180.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.residue_name= 'G'
    
class AlaGeo(Geo):
    '''Geometry of Alanin'''
    def __init__(self):
        self.CA_N_length = 1.46
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.068

        self.C_O_length = 1.23
        self.CA_C_O_angle = 120.5
        self.N_CA_C_O_diangle = -60.5

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277
    

        self.CA_CB_length = 1.53
        self.C_CA_CB_angle = 110.5
        self.N_C_CA_CB_diangle = -122.5

        self.residue_name= 'A'

class SerGeo(Geo):
    '''Geometry of Serine'''
    def __init__(self):
        self.CA_N_length = 1.46
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.2812

        self.C_O_length = 1.23
        self.CA_C_O_angle = 120.5
        self.N_CA_C_O_diangle = -60.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277
    

        self.CA_CB_length = 1.53
        self.C_CA_CB_angle = 110.5
        self.N_C_CA_CB_diangle = -122.5

        self.residue_name= 'S'

class CysGeo(Geo):                                        
    '''Geometry of Cystine'''
    def __init__(self):
        self.CA_N_length = 1.46
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 110.8856

        self.C_O_length = 1.23
        self.CA_C_O_angle = 120.5
        self.N_CA_C_O_diangle = -60.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277
    

        self.CA_CB_length = 1.53
        self.C_CA_CB_angle = 110.5
        self.N_C_CA_CB_diangle = -122.5

        self.residue_name= 'C'

class ValGeo(Geo):
    '''Geometry of Valine'''
    def __init__(self):
        self.CA_N_length = 1.46
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 109.7698

        self.C_O_length = 1.23
        self.CA_C_O_angle = 120.5686
        self.N_CA_C_O_diangle = -60.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 1.546
        self.C_CA_CB_angle = 111.5
        self.N_C_CA_CB_diangle = -122.5

        self.residue_name= 'V'

class IleGeo(Geo):
    '''Geometry of Isoleucine'''
    def __init__(self):
        self.CA_N_length = 1.46
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 109.7202

        self.C_O_length = 1.23
        self.CA_C_O_angle = 120.5403
        self.N_CA_C_O_diangle = -60.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 1.546
        self.C_CA_CB_angle = 111.5
        self.N_C_CA_CB_diangle = -122.5

        self.residue_name= 'I'

class LeuGeo(Geo):
    '''Geometry of Leucine'''
    def __init__(self):
        self.CA_N_length = 1.46
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 110.8652

        self.C_O_length = 1.23
        self.CA_C_O_angle = 120.4647
        self.N_CA_C_O_diangle = 120.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 1.53
        self.C_CA_CB_angle = 110.5
        self.N_C_CA_CB_diangle = -122.5

        self.residue_name= 'L'

class ThrGeo(Geo):
    '''Geometry of Threonine'''
    def __init__(self):
        self.CA_N_length = 1.46
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 110.7014

        self.C_O_length = 1.23
        self.CA_C_O_angle = 120.5359
        self.N_CA_C_O_diangle = 120.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 1.542
        self.C_CA_CB_angle = 111.5
        self.N_C_CA_CB_diangle = -122

        self.residue_name= 'T'

class ArgGeo(Geo):
    '''Geometry of Arginine'''
    def __init__(self):
        self.CA_N_length = 1.46
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 110.98

        self.C_O_length = 1.23
        self.CA_C_O_angle = 120.54
        self.N_CA_C_O_diangle = 120.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 1.53
        self.C_CA_CB_angle = 110.5
        self.N_C_CA_CB_diangle = -122.5

        self.residue_name= 'R'

class LysGeo(Geo):
    '''Geometry of Lysine'''
    def __init__(self):
        self.CA_N_length = 1.46
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.08

        self.C_O_length = 1.23
        self.CA_C_O_angle = 120.54
        self.N_CA_C_O_diangle = 120.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 1.53
        self.C_CA_CB_angle = 110.5
        self.N_C_CA_CB_diangle = -122.5

        self.residue_name= 'K'

class AspGeo(Geo):
    '''Geometry of Aspartic Acid'''
    def __init__(self):
        self.CA_N_length = 1.46
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.03

        self.C_O_length = 1.23
        self.CA_C_O_angle = 120.51
        self.N_CA_C_O_diangle = 120.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277
    
        self.CA_CB_length = 1.53
        self.C_CA_CB_angle = 110.5
        self.N_C_CA_CB_diangle = -122.5

        self.residue_name= 'D'

class AsnGeo(Geo):
    '''Geometry of Asparagine'''
    def __init__(self):
        self.CA_N_length = 1.46
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.5

        self.C_O_length = 1.23
        self.CA_C_O_angle = 120.4826
        self.N_CA_C_O_diangle = -60.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277
        
        self.CA_CB_length = 1.53
        self.C_CA_CB_angle = 110.5
        self.N_C_CA_CB_diangle = -122.5

        self.residue_name= 'N'

class GluGeo(Geo):
    '''Geometry of Glutamic Acid'''
    def __init__(self):
        self.CA_N_length = 1.46
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.1703

        self.C_O_length = 1.23
        self.CA_C_O_angle = 120.511
        self.N_CA_C_O_diangle = 120.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 1.53
        self.C_CA_CB_angle = 110.5
        self.N_C_CA_CB_diangle = -122.5

        self.residue_name= 'E'

class GlnGeo(Geo):                                
    '''Geometry of Glutamine'''
    def __init__(self):
        self.CA_N_length = 1.46
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.0849

        self.C_O_length = 1.23
        self.CA_C_O_angle = 120.5029
        self.N_CA_C_O_diangle = 120.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 1.53
        self.C_CA_CB_angle = 110.5
        self.N_C_CA_CB_diangle = -122.5

        self.residue_name= 'Q'

class MetGeo(Geo):                                    
    '''Geometry of Methionine'''
    def __init__(self):
        self.CA_N_length = 1.46
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 110.9416

        self.C_O_length = 1.23
        self.CA_C_O_angle = 120.4816
        self.N_CA_C_O_diangle = 120.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 1.53
        self.C_CA_CB_angle = 110.5
        self.N_C_CA_CB_diangle = -122.5

        self.residue_name= 'M'

class HisGeo(Geo):                               
    '''Geometry of Histidine'''
    def __init__(self):
        self.CA_N_length = 1.46
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 111.0859

        self.C_O_length = 1.23
        self.CA_C_O_angle = 120.4732
        self.N_CA_C_O_diangle = 120.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 1.53
        self.C_CA_CB_angle = 110.5
        self.N_C_CA_CB_diangle = -122.5

        self.residue_name= 'H'

class ProGeo(Geo):
    '''Geometry of Proline'''
    def __init__(self):
        self.CA_N_length = 1.46
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 112.7499

        self.C_O_length = 1.23
        self.CA_C_O_angle = 120.2945
        self.N_CA_C_O_diangle = -45.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 1.53
        self.C_CA_CB_angle = 103.2
        self.N_C_CA_CB_diangle = -120

        self.residue_name= 'P'
    
class PheGeo(Geo):
    '''Geometry of Phenylalanine'''
    def __init__(self):
        self.CA_N_length = 1.46
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 110.7528

        self.C_O_length = 1.23
        self.CA_C_O_angle = 120.5316
        self.N_CA_C_O_diangle = 120.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 1.53
        self.C_CA_CB_angle = 110.5
        self.N_C_CA_CB_diangle = -122.5

        self.residue_name= 'F'

class TyrGeo(Geo):                                             
    '''Geometry of Tyrosine'''
    def __init__(self):
        self.CA_N_length = 1.46
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 110.9288

        self.C_O_length = 1.23
        self.CA_C_O_angle = 120.5434
        self.N_CA_C_O_diangle = 120.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 1.53
        self.C_CA_CB_angle = 110.5
        self.N_C_CA_CB_diangle = -122.5

        self.residue_name= 'Y'

class TrpGeo(Geo):                              
    '''Geometry of Tryptophan'''
    def __init__(self):
        self.CA_N_length = 1.46
        self.CA_C_length = 1.52
        self.N_CA_C_angle = 110.8914

        self.C_O_length = 1.23
        self.CA_C_O_angle = 120.5117
        self.N_CA_C_O_diangle = 120.0

        self.phi = -120
        self.psi_im1 = 140
        self.omega = 180.0
        self.peptide_bond = 1.33
        self.CA_C_N_angle = 116.642992978143
        self.C_N_CA_angle = 121.382215820277

        self.CA_CB_length = 1.53
        self.C_CA_CB_angle = 110.5
        self.N_C_CA_CB_diangle = -122.5

        self.residue_name= 'W'

def geometry(AA):
    '''Generates the geometry of the requested amino acid.
    The amino acid needs to be specified by its single-letter
    code. If an invalid code is specified, the function
    returns the geometry of Glycine.'''
    if(AA=='G'):
        return GlyGeo()
    elif(AA=='A'):
        return AlaGeo()
    elif(AA=='S'):
        return SerGeo()
    elif(AA=='C'):
        return CysGeo()
    elif(AA=='V'):
        return ValGeo()
    elif(AA=='I'):
        return IleGeo()
    elif(AA=='L'):
        return LeuGeo()
    elif(AA=='T'):
        return ThrGeo()
    elif(AA=='R'):
        return ArgGeo()
    elif(AA=='K'):
        return LysGeo()
    elif(AA=='D'):
        return AspGeo()
    elif(AA=='E'):
        return GluGeo()
    elif(AA=='N'):
        return AsnGeo()
    elif(AA=='Q'):
        return GlnGeo()
    elif(AA=='M'):
        return MetGeo()
    elif(AA=='H'):
        return HisGeo()
    elif(AA=='P'):
        return ProGeo()
    elif(AA=='F'):
        return PheGeo()
    elif(AA=='Y'):
        return TyrGeo()
    elif(AA=='W'):
        return TrpGeo()
    else:
        print ("Geometry.geometry() wrong")

