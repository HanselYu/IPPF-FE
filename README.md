# IPPF-FE：An integrated peptide and protein function prediction framework based on fused features and ensemble models
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
You also need install pretrained language modoel *ProtT5-XL-UniRef50*, the link is provided on [ProtT5-XL-U50](https://github.com/agemagician/ProtTrans#models).
## 3. Requirments
In order to run successfully, the embedding of *ProtT5-XL-UniRef50* requires GPU. We utilized an NVIDIA GeForce RTX 3080 with 10018MiB to embed peptide or protein sequences to 1024-dimensional vector. And Hand-crafted features could be implemented on personal computer.
Other hardware equipments are not necessary.

## 4. Usage
For each dataset, you could run corresponding *.py* file, cross-validation and independent test are all implemented.

## 5. References

## 6. Contact
If you have any question, you could contact **han.yu@siat.ac.cn**.
