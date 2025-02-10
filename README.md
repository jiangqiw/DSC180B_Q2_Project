# Balancing Accuracy and Efficiency: A Comparative Study of Knowledge Distillation and Post-Training Quantization Techniques

## Abstract
This repository contains the project for DSC180 Q2, aimed at optimizing compressed models through a combination of knowledge distillation, pre-training pruning, and post-training quantization, specifically GPFQ algorithm, applied on the CIFAR10 dataset.

Model compression techniques like **Knowledge Distillation (KD)**, **Pruning**, and **Post-Training Quantization (PTQ)** are crucial for deploying deep learning models on resource-constrained devices. This project explores the efficiency-accuracy trade-offs of these methods on the CIFAR-10 dataset using the GPFQ algorithm and other model compression techniques jointly.

---

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

- **Notebooks/**: Used for experimentation. Not intended for use outside of this purpose.
- **models/**: Stores trained student models and teacher models.
- **model_checkpoints/**: Saves model checkpoints during training.


---

## Environment Instructions
> **Note:** A Docker setup is planned (TODO: Dockerize). For now, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jiangqiw/DSC180B_Q2_Project.git
   cd DSC180B_Q2_Project
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - On **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - On **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Downloading Pretrained Weights

To use **ResNet-50** as the teacher model:

1. Download `pytorch_model.bin` from [Hugging Face](https://huggingface.co/edadaltocg/resnet50_cifar10/tree/main).
2. Rename the file to:
   ```
   resnet50_cifar10_pretrained.bin
   ```
3. Place it in the `models/` directory.

Alternatively, you can train your own teacher model and rename it to match this name, as long as it is a Resnet-50 architechture.

---

## Training models

You can train student models using the provided scripts in the `src/` folder. Checkpoints are saved in the `model_checkpoints/` folder, and final models are saved in the `models/` folder.

Example:
```bash
python src/train_student.py --temperatures 4 --learning_rates 0.0005 --learning_rate_decays 0.95 --weight_decays 0.0001 --momentums 0.9 --dropout_probabilities 0.0 0.0 --num_epochs 200 --print_every 100
```
### Arguements:
- `--temperatures`: Temperature values for distillation (default: 4)
- `--learning_rates`: Learning rates for training (default: 0.0005)
- `--learning_rate_decays`: Learning rate decay factors (default: 0.95)
- `--weight_decays`: Weight decay values for regularization (default: 0.0001)
- `--momentums`: Momentum values for optimizers (default: 0.9)
- `--dropout_probabilities`: Dropout probabilities for input and hidden layers (default: 0.0 0.0)
- `--num_epochs`: Number of epochs to train (default: 200)
- `--print_every`: Frequency of logging training progress (default: 100)

---

## Acknowledgements
This project references the code provided by https://github.com/shriramsb/Distilling-the-Knowledge-in-a-Neural-Network which is an implementation of a part of the paper "Distilling the Knowledge in a Neural Network" (https://arxiv.org/abs/1503.02531).