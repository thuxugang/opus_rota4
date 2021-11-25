# OPUS-Rota4

Accurate protein side-chain modeling is crucial for protein folding and protein design. In the past decades, many successful methods have been proposed to address this issue. However, most of them depend on the discrete samples from the rotamer library, which may have limitations on their accuracies and usages. In this study, we report an open-source toolkit for protein side-chain modeling, named OPUS-Rota4. It consists of three modules: OPUS-RotaNN2, which predicts protein side-chain dihedral angles; OPUS-RotaCM, which measures the distance and orientation information between the side chain of different residue pairs; and OPUS-Fold2, which applies the constraints derived from the first two modules to guide side-chain modeling. In summary, OPUS-Rota4 adopts the dihedral angles predicted by OPUS-RotaNN2 as its initial states, and uses OPUS-Fold2 to refine the side-chain conformation with the constraints derived from OPUS-RotaCM. In this case, we convert the protein side-chain modeling problem into a side-chain contact map prediction problem. OPUS-Fold2 is written in Python and TensorFlow2.4, which is user-friendly to include other differentiable energy terms into its side-chain modeling procedure. In other words, OPUS-Rota4 provides a platform in which the protein side-chain conformation can be dynamically adjusted under the influence of other processes, such as protein-protein interaction. We apply OPUS-Rota4 on 15 FM predictions submitted by Alphafold2 on CASP14, the results show that the side chains modeled by OPUS-Rota4 are closer to their native counterparts than the side chains predicted by Alphafold2.

## Framework of OPUS-Rota4

<img src="./images/figure1.png"/>

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

## Results

#### The performance of different side-chain modeling methods on three native backbone test sets measured by all residues

#### CAMEO(60)
| |MAE(χ1)|MAE(χ2)|MAE(χ3)|MAE(χ4)|ACC|
|:----:|:----:|:----:|:----:|:----:|:----:|
|FASPR |	29.15 |	42.36 |	57.01| 	57.93 |	49.10%|
|SCWRL4 |	29.01 |	42.88 |	57.25 |	57.17 |	49.48%|
|OSCAR-star|	27.29 |	41.97 |	56.08 |	57.66 |	49.91%|
|DLPacker|	24.11 |	39.60 |	63.84 |	68.10 |	52.19%|
|OPUS-RotaNN2|	21.61 |	31.13 |	49.79 	|47.78| 	55.61%|
|OPUS-Rota4|	**21.34** |	**31.13** |	**49.79** |	**47.78** |	**57.35%**|

#### CASPFM(56)
| |MAE(χ1)|MAE(χ2)|MAE(χ3)|MAE(χ4)|ACC|
|:----:|:----:|:----:|:----:|:----:|:----:|
|FASPR |	26.63 |	39.75 |	53.40 |	54.81 |	53.11%|
|SCWRL4 |	27.09| 	40.44 |	52.67 |	54.61 |	53.17%|
|OSCAR-star	|24.53| 37.43|	50.51 |	52.99 |	54.92%|
|DLPacker|	21.35| 	37.79 |	61.05 |	66.78| 	55.26%|
|OPUS-RotaNN2|	18.85 |	28.50 |	44.88 	|44.87 |	58.17%|
|OPUS-Rota4|	**18.46** |	**28.50** |	**44.88** |	**44.87** |	**60.42%**|

#### CASP14(15)
| |MAE(χ1)|MAE(χ2)|MAE(χ3)|MAE(χ4)|ACC|
|:----:|:----:|:----:|:----:|:----:|:----:|
|FASPR| 	35.80 |	48.72 	|56.59 |	45.19 |	36.34%|
|SCWRL4 |	35.27 |	48.13 |	58.37 |	48.15 |	36.57%|
|OSCAR-star|	34.45 |	48.10| 	56.70 |	42.28 	|36.76%|
|DLPacker|	30.99 	|48.21 |	65.14 |	70.83| 	40.05%|
|OPUS-RotaNN2	|**28.21** 	|40.14 |	51.93 |	40.76 |	41.16%|
|OPUS-Rota4|	28.33 |	**40.14** |	**51.93** 	|**40.76**| 	**43.38%**|

#### The RMSD results of different side-chain modeling methods on non-native backbone test set  CASP14-AF2 (15)

|	|RMSD(All)	|P-value	|RMSD(Core)|	P-value|
|:----:|:----:|:----:|:----:|:----:|
|AlphaFold2	|0.588 |	1.3E-12	|0.472 |	5.5E-04|
|FASPR| 	0.574 |	2.5E-09|	0.484 |	1.4E-05|
|SCWRL4 |	0.585 |	3.9E-14|	0.489 |	1.0E-05|
|OSCAR-star|	0.569 |	5.9E-08	|0.483 |	3.0E-05|
|DLPacker|	0.576 |	1.1E-13|	0.449| 	5.9E-04|
|OPUS-Rota4|	**0.535** |	-|**0.407** |	-|


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

## Accessibility
This project is freely available for academic usage.

## Reference 
```bibtex
@article{xu2021opus2,
  title={OPUS-Rota4: A Gradient-Based Protein Side-Chain Modeling Framework Assisted by Deep Learning-Based Predictors},
  author={Xu, Gang and Wang, Qinghua and Ma, Jianpeng},
  journal={Briefings in Bioinformatics},
  year={2021},
  publisher={Oxford University Press}
}
```
