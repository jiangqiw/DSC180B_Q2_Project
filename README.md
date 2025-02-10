# Balancing Accuracy and Efficiency: A Comparative Study of Knowledge Distillation and Post-Training Quantization Techniques

## Abstract
This repository contains the project for DSC180 Q2, aimed at optimizing compressed models through a combination of knowledge distillation, pre-training pruning, and post-training quantization, specifically GPFQ algorithm, applied on the CIFAR10 dataset.

## Repo Overview

The source code is included in the SRC folder, with the following structure:
```
src/
└── GPFQ/ # Includes code to run GPFQ
    ├── quantize_neural_net.py
    ├── step_algorithm.py
└── models/
    ├── pruning.py # Model pruning helper code
    ├── quantized_resnet18.py # Torch quantizable variation of Resnet18
    ├── resnetv2.py # Alternative Resnet variant
    ├── student.py # Student model class
    └── teachers.py # Teacher model classes
└── shampoo_optimizer/ # Additional optimizer option
    ├── matrix_functions.py
    ├── shampoo_utils.py
    └── shampoo.py
└── utilities/
    ├── data_utils.py # Utilities to load data
    ├── model_utils.py # Utilities for models
    ├── utils.py # Utilities for training
└── train_student_dkd.py # Script to train with the DKD method
└── train_student_DML.py # Script to train with DML
└── train_student_QAT.py # Script to train with QAT and mixup
└── train_student.py # Script to train with no additional methods
└── train_student_mixup.py # Script to train with mixup
```

Notebooks used for experimentation can be found in the notebooks folder, but these should not be treated as functional to run scripts without modifications.

The models folder should include trained student models, as well as the teacher models used for training students.

The model_checkpoints folder will save model checkpoints created during training.

## Setup Instructions
TODO: Dockerize

To get started with this project, follow these steps to set up your environment:

1. Clone the repository:
   git clone [<repository_url>](https://github.com/jiangqiw/DSC180B_Q2_Project.git)

   cd DSC180B_Q2_Project

2. Create a virtual environment:

   python -m venv venv

3. Activate the virtual environment:
   - On Windows:
     venv\Scripts\activate
   - On macOS/Linux:
     source venv/bin/activate

4. Install the required dependencies:
   pip install -r requirements.txt

Whenever you start working on the project, make sure to **activate the virtual environment**:

- On Windows:
  venv\Scripts\activate

- On macOS/Linux:
  source venv/bin/activate

You will also need to download the weights used to train models from external sources. To use Resnet-50 as the teacher, you can download the pytorch_model.bin file from this link: https://huggingface.co/edadaltocg/resnet50_cifar10/tree/main or train your own. This file should be named resnet50_cifar10_pretrained.bin for the scripts to work correctly.

## Training models

To train a student model the scripts in the src folder can be ran. The student model checkpoints throughout training will be saved to the model_checkpoints/ folder, and the final model will be saved in the models/ folder.

## Acknowlwdedgement
This project references the code provided by https://github.com/shriramsb/Distilling-the-Knowledge-in-a-Neural-Network which is an implementation of a part of the paper "Distilling the Knowledge in a Neural Network" (https://arxiv.org/abs/1503.02531)S