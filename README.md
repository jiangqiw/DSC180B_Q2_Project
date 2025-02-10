# Balancing Accuracy and Efficiency: A Comparative Study of Knowledge Distillation and Post-Training Quantization Techniques

## Abstract
This repository contains the project for DSC180 Q2, aimed at optimizing compressed models through a combination of knowledge distillation, pre-training pruning, and post-training quantization, specifically GPFQ algorithm, applied on the CIFAR10 dataset.

## Repo Overview

The source code is included in the SRC folder, with the following structure:
```
src/
└── GPFQ/
    ├── quantize_neural_net.py
    ├── step_algorithm.py
└── models/
    ├── pruning.py
    ├── quantized_resnet18.py
    ├── resnetv2.py
    ├── student.py
    └── teachers.py
└── shampoo_optimizer/
    ├── matrix_functions.py
    ├── shampoo_utils.py
    └── shampoo.py
└── utilities/
    ├── data_utils.py
    ├── model_utils.py
    ├── utils.py
└── train_student_dkd.py # Script to train with the DKD method
└── train_student_DML.py # Script to train with DML
└── train_student_QAT.py # Script to train with QAT and mixup
└── train_student.py # Script to train with no additional methods
└── train_student_mixup.py # Script to train with mixup
```

Notebooks used for experimentation can be found in the notebooks folder, but these should not be treated as functional to run scripts without modifications.

The models folder should include trained student models, as well as the teacher models used for training students.

The model_checkpoints folder will save model checkpoints created during training.

## Acknowlwdedgement
This project references the code provided by https://github.com/shriramsb/Distilling-the-Knowledge-in-a-Neural-Network which is an implementation of a part of the paper "Distilling the Knowledge in a Neural Network" (https://arxiv.org/abs/1503.02531)S