# Balancing Accuracy and Efficiency: A Comparative Study of Knowledge Distillation and Post-Training Quantization Techniques

## Abstract
This repository contains the project for DSC180 Q2, aimed at optimizing compressed models through a combination of knowledge distillation, pre-training pruning, and post-training quantization, specifically GPFQ algorithm, applied on the CIFAR10 dataset.

## Repo Overview

### `knowledge_distillation.ipynb`
This jupyter notebook Contains the main code to train both the teacher and student models. Please Adjust the second cell in each notebook to accommodate the availability of GPU resources.

### `utils.py`
This code for distillation and training

### `networks.py`
This code for setting up the network architectures (input).

### `checkpoints_teacher`
This directory stores the trained teacher networks along with their performance metrics.
- **`results_teacher.csv`**: Contains results of the teacher network. Each record includes Dropout Input, Dropout Hidden, Weight Decay, LR Decay, Momentum, Learning Rate, Test Accuracy, and Training Time (s).

### `checkpoints_student`
This directory stores the trained student networks along with their performance metrics.
- **`results_student.csv`**: Contains results of the student. Each record includes Alpha,Temperature,Dropout Input,Dropout Hidden,Weight Decay,LR Decay,Momentum,Learning Rate,Pruning Factor,Zero Parameters,Test Accuracy,Training Time (s).

## Acknowlwdedgement
This project references the code provided by https://github.com/shriramsb/Distilling-the-Knowledge-in-a-Neural-Network which is an implementation of a part of the paper "Distilling the Knowledge in a Neural Network" (https://arxiv.org/abs/1503.02531)