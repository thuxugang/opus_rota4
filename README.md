# OPUS-Rota4

Accurate protein side-chain modeling is crucial for protein folding and protein design. In the past decades, many successful methods have been proposed to address this issue. However, most of them depend on the discrete samples from the rotamer library, which may have limitations on their accuracies and usages. In this study, we report an open-source toolkit for protein side-chain modeling, named OPUS-Rota4. It consists of three modules: OPUS-RotaNN2, which predicts protein side-chain dihedral angles; OPUS-RotaCM, which measures the distance and orientation information between the side chain of different residue pairs; and OPUS-Fold2, which applies the constraints derived from the first two modules to guide side-chain modeling. In summary, OPUS-Rota4 adopts the dihedral angles predicted by OPUS-RotaNN2 as its initial states, and uses OPUS-Fold2 to refine the side-chain conformation with the constraints derived from OPUS-RotaCM. In this case, we convert the protein side-chain modeling problem into a side-chain contact map prediction problem. OPUS-Fold2 is written in Python and TensorFlow2.4, which is user-friendly to include other differentiable energy terms into its side-chain modeling procedure. In other words, OPUS-Rota4 provides a platform in which the protein side-chain conformation can be dynamically adjusted under the influence of other processes, such as protein-protein interaction. We apply OPUS-Rota4 on 15 FM predictions submitted by Alphafold2 on CASP14, the results show that the side chains modeled by OPUS-Rota4 are closer to their native counterparts than the side chains predicted by Alphafold2.


## Usage

### Dependency

```
Python 3.7
TensorFlow 2.4
```

The standalone version of OPUS-Rota4 and the test sets we used can be downloaded directly from [Here](https://ma-lab.rice.edu/dist/opus-rota4.zip).

```
$DOWNLOAD_DIR/                             # Total: ~ 4 GB
    datasets/                              
        # af2_bb_data (AlphaFold2 predicted backbones for CASP14(15))
        # bb_data (Native backbones for CAMEO(60), CASP14(15) and CASPFM(56))
        # data (Native structures for all datasets)
        
    OPUS_RotaNN2_and_RotaCM/
        RotaCM/
          # Codes and pre-trained models for OPUS-RotaCM
        RotaNN2/
          # Codes and pre-trained models for OPUS_RotaNN2
          DLPacker_OPUS/
            # Codes and pre-trained models for DLPacker(OPUS)
          mkinputs/
            # Codes for calculating input features

    OPUS-Fold2/
        # Codes for OPUS-Fold2
```


## Useful Tools

### OPUS-RotaNN

[OPUS-RotaNN](https://github.com/thuxugang/opus_rota3)

### OPUS-X

[OPUS-X](https://github.com/thuxugang/opus_x)

### DLPacker

[DLPacker](https://github.com/nekitmm/DLPacker)


## Run OPUS_RotaNN2_and_RotaCM

Use `run_opus_rota4.py` to generate the results of OPUS-RotaNN2 (\*.rotann2) and OPUS-RotaCM (\*.rotacm.npz).


## Run OPUS-Fold2

Use `run_opus_fold2.py` to generate the optimized results of OPUS-Rota4 (\*.rota4) and (\*.pdb).
