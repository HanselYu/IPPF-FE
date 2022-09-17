# IPPF-FE：An integrated peptide and protein function prediction framework based on fused features and ensemble models
# We are designing user-friendly website for biologists!
## 1. Introduction
This repository contains source data and code for paper "An integrated peptide and protein function prediction framework based on fused features and ensemble models".
IPPF-FE is a python implementation of the model.

## 2. Installation
```
python=3.6.9
```
You could configure enviroment by running this:
```
pip install -r requirment.txt
```
Notice:
1. You need install pretrained language modoel *ProtT5-XL-UniRef50*, the link is provided on [ProtT5-XL-U50](https://github.com/agemagician/ProtTrans#models).
2. You also could *pip install torch 1.10.1+cu113* by manual method, the link is provided as on [Pytorch](https://download.pytorch.org/whl/torch/).

## 3. Requirments
In order to run successfully, the embedding of *ProtT5-XL-UniRef50* requires GPU. We utilized an NVIDIA GeForce RTX 3080 with 10018MiB to embed peptide or protein sequences to 1024-dimensional vector. And Hand-crafted features could be implemented on personal computer.
Other hardware equipments are not necessary.

## 4. Usage
For each dataset, you could run corresponding *.py* file, train model and external test are all implemented. We took Antibacterial peptides dataset as an example.

### Train model
```
python Pantibacterial.py Train -
```
### External test
```
python Pantibacterial.py Test test.fasta
```
### Optional input variables
```
-predicted type    e.g. Pantibacterial.py, Phemolytic.py, Pbiofilm_inhibitory.py, PDPP_IV.py, PT3SEs.py, Pcsq_resolution.py, Pcsq_rfree.py, Pgpl.py, Pcyclinp.py
-Train or Test    e.g. Train, Test
-input file    e.g. test.fasta
```
## 5. References

## 6. Contact
If you have any question, you could contact **han.yu@siat.ac.cn**.
