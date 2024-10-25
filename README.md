# Towards Hardware-Efficient Modulation Recognition using Low-Rate Spiking Neural Networks

This repository contains the code for the paper "Towards Hardware-Efficient Modulation Recognition using Low-Rate Spiking Neural Networks," by [Sai Sanjeet](https://scholar.google.com/citations?user=vMHDxGIAAAAJ) and [Bibhu Datta Sahoo](https://scholar.google.com/citations?user=AuzF4ScAAAAJ).

If you have any questions, please feel free to reach out to the authors. email: syerragu@buffalo.edu

## Abstract

Real-time modulation recognition is crucial for modern communication systems in various cognitive radio tasks. While prior works have employed deep learning techniques to address this challenge, few are feasible for real-time applications. Spiking Neural Networks (SNNs) present a promising alternative to conventional deep learning approaches, enabling low-power hardware implementations. However, existing SNN-based modulation recognition methods often lag behind traditional techniques or necessitate high sample rate implementations. This work introduces an SNN architecture that utilizes a low-resolution quantizer in the receiver and operates at a lower rate than the quantizer, resulting in significant area and power savings when integrated into a system. We experimentally determine the optimal quantizer resolution and the ratio of quantizer-to-SNN rate. The optimized network achieves an average classification accuracy of 68.45\% on the RadioML2018.01A dataset, utilizing a 4-bit quantizer and running at a rate 16 times lower than the quantizer. This performance is comparable to conventional neural networks and surpasses that of previous spiking-based methods, especially at low signal-to-noise ratio (SNR) conditions.

## Requirements

The code is implemented in Python 3.10 and PyTorch with the following dependencies:

- numpy 1.26.3
- torch 2.3.1
- h5py 3.6.0
- rich 13.7.1 (for pretty-printing tables)

## Datasets

The datasets used are the RadioML2018.01A dataset and the RadioML2016.10A dataset, available at https://www.deepsig.ai/datasets. The `DATAFILE` variable in `main_spiking_rml16.py` and `main_spiking_rml18.py` should be set to the path of the corresponding dataset files.

## Usage

The design hyperparameter sweep mentioned in the paper is implemented in `rml18_hyperparam_search.py`. The optimal hyperparameters are then used in `results_rml18_seed_loop.py` and `results_rml16_seed_loop.py` to obtain the results reported in the paper.
