# BiHyPE: Binary-based Hybrid Positional Encoding for Time-series Analysis

[![Python 3.6 (Legacy)](https://img.shields.io/badge/Python-3.6-yellow?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch >=1.4.0](https://img.shields.io/badge/PyTorch-%3E%3D1.4.0%2C_%3C%3D1.10.2-red?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

## Baseline Overview
This repository provides an implementation/reproduction of the baseline **[Anomaly Transformer]** based on the original paper:
* **Paper Title:** *[Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy]*
* **Authors:** *[Xu, Jiehui, et al]*
* **Source / Links:** [https://arxiv.org/abs/2110.02642] | [https://github.com/thuml/Anomaly-Transformer]
* **Role in Research:** Serves as a benchmark Transformer-based baseline to evaluate against our proposed method [**BiHyPE**].

## Implementation Details

1. Install Python 3.6, PyTorch >= 1.4.0. 
2. You can obtain five benchmarks from [Google Drive](https://drive.google.com/drive/folders/1PKHJIi2--liygyXCXiGmgj_RBbDtz6DD?usp=sharing). **All the datasets are well pre-processed by the authors of Anomaly Transformer**. You can download this data, then create the folder of dataset in the source code (valid dataset directory example: dataset/MSL, dataset/PSM, dataset/SMAP, dataset/SWaT, dataset/SMD).
3. Train and evaluate. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results as follows:
```bash
bash ./scripts/SMD.sh
bash ./scripts/MSL.sh
bash ./scripts/SMAP.sh
bash ./scripts/PSM.sh
bash ./scripts/SWaT.sh
```

Especially, you can reproduce across all datasets as follows:
```bash
bash ./scripts/Start.sh
```
## Experimental Results on Anomaly Transformer Baseline

The performance and anomaly score distribution of the Anomaly Transformer baseline were evaluated across 5 benchmark datasets:

| Dataset | Figure of corresponding dataset |
| :--- | :--- |
| **MSL** | <img src="images/MSL_result.jpg" width="100%" alt="MSL Result"> |
| **PSM** | <img src="images/PSM_result.jpg" width="100%" alt="PSM Result"> |
| **SMAP** | <img src="images/SMAP_result.jpg" width="100%" alt="SMAP Result"> |
| **SWaT** | <img src="images/SWaT_result.jpg" width="100%" alt="SWaT Result"> |
| **SMD** | <img src="images/SMD_result.jpg" width="100%" alt="SMD Result"> |
