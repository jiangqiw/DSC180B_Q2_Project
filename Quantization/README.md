# GPFQ: Post-Training Quantization On Pre-trained Model with Hyper-parameter Analysis
## Overview
This is the repository for DSC180 Q1 project, for conducting post-training quantization, specifically GPFQ algorithm, on CIFAR10 dataset.

## Guide on Installing Dependencies

The repo would require python version larger than `3.8.0` is installed in the user's 
machine.

To install the necessary dependency, one can first start a virtual environment
by doing the following: 
```
python3.9 -m venv .venv
source .venv/bin/activate
```
Then can install file using the `requirements.txt` file
```
pip3 install -r requirements.txt
```
This should install all the required dependencies of this project. 

## Repo Overview

This repository contains code for replicating GPFQ quantization algorithm and notebook for visualizing train and test data (CIFAR10). There are several directories, each serving a specific purpose. Below is an overview of each directory to help you navigate the codebase effectively.

### `image`

This is the directory holding the figures for quantization result among different parameters. 

### `Logs`

This is the directory holding the logs data for recording the quantization result. 

- **`init_log.py`**
    This script initializes and creates the log file used to record various quantization metrics and results. The log file format includes several important entries that facilitate detailed analysis and comparison across experiments. Here are some important entry in the log file:
    - `Original Top5 Accuracy`: The Top-5 accuracy before quantization.
    - `Quantized Top5 Accuracy`: The Top-5 accuracy post quantization.
    - `MLP_Alphabet_Scalar`, `CNN_Alphabet_Scalar`: Scalars used for alphabet quantization in MLP and CNN layers.
    - `Regularizer`, `Lambda`: Regularization type and lambda value used in quantization to sparsify weights.

- **`Quantization_Log_Scalar.csv`**
    This CSV file contains the results of various quantized models, which are parameterized by different scalar values ranging from 1.0 to 2.0, with an incremental step size of 0.05.
  
- **`Quantization_Log_Bit.csv`**
    This CSV file contains the results of various quantized models, which are parameterized by different bit size ranging from 1 to 8, with an incremental step size of 1.
  
### `pretrained_cifar10`

This is the directory holding a pretrained resnet-18 model for CIFAR10. 

- **`resnet18_cifar10.pt`**
  This file contains the saved state dictionary of a ResNet-18 model that has been trained or is ready to be used on the CIFAR-10 dataset.

### `quantized_models`

This is the directory holding the quantized model.

### `src`

This is the directory holding the additional resources

- **`visualize_dataset.ipynb`**
    This is the notebook containing working code for visualizing CIFAR10 dataset.

- **`data_loaders.py`**
    This is the script for loading data to the GPFQ algorithm

- **`main.py`**
    This is the main function for GPFQ, in charge of taking hyper-parameters including number of bits for quantization and scalars.

- **`quantize_neural_net.py`**
    Contains function for running the quantization algorithm by quantizing layer by layer.

- **`step_algorithm.py`**
    Step algorithm for quantizing each layer

- **`utils.py`**
    Scripts containing helper functions

### `requirements.txt`

This is the file holding the required package version for the environment.


## Commands to Run

To run GPFQ on a specific dataset and pre-trained model, navigate to the `src` directory and execute the following command:

```bash
python main.py -ds dataset_name -model model_name -b width -bs batch_size -s scalar
```

**Parameters Explanation**

- -ds dataset_name: Specifies the name of the dataset to use.

Example: MNIST, CIFAR10.

- -model model_name: Defines the pre-trained model to apply GPFQ.

    Example: ResNet, LeNet.

- -b width: Sets the bit-width for quantization.

    Example: 8, 16.

- -bs batch_size: Specifies the batch size for processing the dataset.

    Example: 32, 64.

- -s scalar: A scaling factor applied during the quantization process.

    Example: 0.1, 1.0.


Here is one example of the command:

```
python main.py -ds CIFAR10 -model resnet18 -b 4 -bs 256 -s 1.16
```
This is quantize the ResNet-18 using CIFAR10 data with bit = 4, batch_size = 256, scalar = 1.16.
