# STFreqGNN

## Paper
[**A UNIFIED SPATIOTEMPORAL FREQUENCY GRAPH NEURAL NETWORK FOR FMRI-BASED BRAIN FUNCTIONAL CONNECTIVITY ANALYSIS**](https://ieeexplore.ieee.org/abstract/document/10890520) \
Yulang Huang, Zhiyuan Ding and others \
presented at *ICASSP 2025* \


## Concept
![The framework of STAGIN](framework.png)

## Dataset
Example structure of the dataset directory tree.
```
data
├─── dataset1
│    ├─── raw
│    │    ├─── sub_0.pt
│    │    ├─── sub_1.pt
│    │    ├─── ...
│    │    └─── sub_xx.pt
│    │
│    └─── prcessed
│    
├─── dataset2
├─── ...
└─── datasetX
```

## Environment Creation
Run the commands to create conda environment.
```shell
conda create -n stfreqgnn python=3.11
conda activate stfreqgnn
pip install -r requirements.txt
```


## Training Models
Run the main script to perform experiments.
```shell
bash run/FTD_train.sh
```

