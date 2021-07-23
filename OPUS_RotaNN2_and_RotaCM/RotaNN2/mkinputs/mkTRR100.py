# load libraries
import numpy as np

alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
states = len(alpha_1)
alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
           'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']

aa_1_N = {a:n for n,a in enumerate(alpha_1)}
aa_3_N = {a:n for n,a in enumerate(alpha_3)}
aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}

def AA_to_N(x):
  # ["ARND"] -> [[0,1,2,3]]
  x = np.array(x);
  if x.ndim == 0: x = x[None]
  return [[aa_1_N.get(a, states-1) for a in y] for y in x]

def N_to_AA(x):
  # [[0,1,2,3]] -> ["ARND"]
  x = np.array(x);
  if x.ndim == 1: x = x[None]
  return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]

def parse_PDB(x, atoms=['N','CA','C'], chain=None):
    '''
    input:  x = PDB filename
            atoms = atoms to extract (optional)
    output: (length, atoms, coords=(x,y,z)), sequence
    '''
    xyz,seq,min_resn,max_resn = {},{},np.inf,-np.inf
    for line in open(x,"rb"):
      line = line.decode("utf-8","ignore").rstrip()
    
      if line[:6] == "HETATM" and line[17:17+3] == "MSE":
        line = line.replace("HETATM","ATOM  ")
        line = line.replace("MSE","MET")
    
      if line[:4] == "ATOM":
        ch = line[21:22]
        if ch == chain or chain is None:
          atom = line[12:12+4].strip()
          resi = line[17:17+3]
          resn = line[22:22+5].strip()
          x,y,z = [float(line[i:(i+8)]) for i in [30,38,46]]
    
          if resn[-1].isalpha(): resa,resn = resn[-1],int(resn[:-1])-1
          else: resa,resn = "",int(resn)-1
          if resn < min_resn: min_resn = resn
          if resn > max_resn: max_resn = resn
          if resn not in xyz: xyz[resn] = {}
          if resa not in xyz[resn]: xyz[resn][resa] = {}
          if resn not in seq: seq[resn] = {}
          if resa not in seq[resn]: seq[resn][resa] = resi
    
          if atom not in xyz[resn][resa]:
            xyz[resn][resa][atom] = np.array([x,y,z])
    
    # convert to numpy arrays, fill in missing values
    seq_,xyz_ = [],[]
    for resn in range(min_resn,max_resn+1):
      if resn in seq:
        for k in sorted(seq[resn]): seq_.append(aa_3_N.get(seq[resn][k],20))
      else: seq_.append(20)
      if resn in xyz:
        for k in sorted(xyz[resn]):
          for atom in atoms:
            if atom in xyz[resn][k]: xyz_.append(xyz[resn][k][atom])
            else: xyz_.append(np.full(3,np.nan))
      else:
        for atom in atoms: xyz_.append(np.full(3,np.nan))
    return np.array(xyz_).reshape(-1,len(atoms),3), np.array(seq_)

def extend(a,b,c, L,A,D):
    '''
    input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
    output: 4th coord
    '''
    N = lambda x: x/np.sqrt(np.square(x).sum(-1,keepdims=True))
    bc = N(b-c)
    n = N(np.cross(b-a, bc))
    m = [bc,np.cross(n,bc),n]
    d = [L*np.cos(A), L*np.sin(A)*np.cos(D), -L*np.sin(A)*np.sin(D)]
    return c + sum([m*d for m,d in zip(m,d)])

def to_len(a,b):
    '''given coordinates a-b, return length or distance'''
    return np.sqrt(np.sum(np.square(a-b),axis=-1))

def to_ang(a,b,c):
    '''given coordinates a-b-c, return angle'''
    D = lambda x,y: np.sum(x*y,axis=-1)
    N = lambda x: x/np.sqrt(np.square(x).sum(-1,keepdims=True) + 1e-8)
    return np.arccos(D(N(b-a),N(b-c)))

def to_dih(a,b,c,d):
    '''given coordinates a-b-c-d, return dihedral'''
    D = lambda x,y: np.sum(x*y,axis=-1)
    N = lambda x: x/np.sqrt(np.square(x).sum(-1,keepdims=True) + 1e-8)
    bc = N(b-c)
    n1 = np.cross(N(a-b),bc)
    n2 = np.cross(bc,N(c-d))
    return np.arctan2(D(np.cross(n1,bc),n2),D(n1,n2))

def mtx2bins(x_ref, start, end, nbins, mask):
    bins = np.linspace(start, end, nbins)
    x_true = np.digitize(x_ref, bins).astype(np.uint8)
    x_true[mask] = 0
    return np.eye(nbins+1)[x_true][...,:-1]
    
def prep_input(pdb, chain=None, mask_gaps=False):
    '''Parse PDB file and return features compatible with TrRosetta'''
    ncac, seq = parse_PDB(pdb,["N","CA","C"], chain=chain)
    
    # mask gap regions
    if mask_gaps:
        mask = seq != 20
        ncac, seq = ncac[mask], seq[mask]
    
    N,CA,C = ncac[:,0], ncac[:,1], ncac[:,2]
    CB = extend(C, N, CA, 1.522, 1.927, -2.143)
    
    dist_ref  = to_len(CB[:,None], CB[None,:])
    omega_ref = to_dih(CA[:,None], CB[:,None], CB[None,:], CA[None,:])
    theta_ref = to_dih( N[:,None], CA[:,None], CB[:,None], CB[None,:])
    phi_ref   = to_ang(CA[:,None], CB[:,None], CB[None,:])
    
    p_dist  = mtx2bins(dist_ref,     2.0,  20.0, 37, mask=(dist_ref > 20))
    p_omega = mtx2bins(omega_ref, -np.pi, np.pi, 25, mask=(p_dist[...,0]==1))
    p_theta = mtx2bins(theta_ref, -np.pi, np.pi, 25, mask=(p_dist[...,0]==1))
    p_phi   = mtx2bins(phi_ref,      0.0, np.pi, 13, mask=(p_dist[...,0]==1))
    feat    = np.concatenate([p_theta, p_phi, p_dist, p_omega],-1)
    # return {"seq":N_to_AA(seq), "feat":feat, "dist_ref":dist_ref}
    return dist_ref, omega_ref, theta_ref, phi_ref, feat

def split_feat(feat):
    out = {}
    for k,i,j in [["theta",0,25],["phi",25,38],["dist",38,75],["omega",75,100]]:
        out[k] = feat[...,i:j]
    return out

def get_dist_acc(pred, true, true_mask=None,sep=5,eps=1e-8):
    ## compute accuracy of CB features ##
    pred,true = [x[...,39:51].sum(-1) for x in[pred,true]]
    if true_mask is not None:
        mask = true_mask[:,:,None] * true_mask[:,None,:]
    else: mask = np.ones_like(pred)
    i,j = np.triu_indices(pred.shape[-1],k=sep)
    P,T,M = pred[...,i,j], true[...,i,j], mask[...,i,j]
    ## give equal weighting to positive and negative predictions
    pos = (T*P*M).sum(-1)/((M*T).sum(-1)+eps)
    neg = ((1-T)*(1-P)*M).sum(-1)/((M*(1-T)).sum(-1)+eps)
    return 2.0*(pos*neg)/(pos+neg+eps)    