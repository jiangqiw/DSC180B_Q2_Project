import torch.nn as nn
import requests
import numpy as np
import io
import torch
import os
from torchvision.models.resnet import BasicBlock as tBasicBlock
from torchvision.models.resnet import Bottleneck as tBottleneck 
from torchvision.models.resnet import ResNet as tResNet
from torchvision.models.googlenet import BasicConv2d, Inception, InceptionAux
from torchvision.models.efficientnet import Conv2dNormActivation, SqueezeExcitation, MBConv 
from torchvision.models.mobilenetv2 import InvertedResidual

SUPPORTED_LAYER_TYPE = {nn.Linear, nn.Conv2d}
SUPPORTED_BLOCK_TYPE = {nn.Sequential, 
                        tBottleneck, tBasicBlock, tResNet,
                        BasicConv2d, Inception, InceptionAux,
                        Conv2dNormActivation, SqueezeExcitation, MBConv,
                        InvertedResidual
                        }


# Helper Function to Modify ResNet for CIFAR-10

def modify_resnet(model: nn.Module, num_classes: int) -> nn.Module:
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

def get_weights(bit_variant):
  response = requests.get(f'https://storage.googleapis.com/bit_models/{bit_variant}.npz')
  response.raise_for_status()
  return np.load(io.BytesIO(response.content))

def count_parameters(model):
    """
    Counts the total number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The model whose parameters need to be counted.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum((p.data != 0).sum().item() for p in model.parameters() if p.requires_grad)


def count_zero_parameters(model):
    """
    Counts the number of trainable parameters that are exactly zero in a PyTorch model.

    Args:
        model (torch.nn.Module): The model whose zero parameters need to be counted.

    Returns:
        int: Total number of trainable parameters that are exactly zero.
    """
    return sum((p.data == 0).sum().item() for p in model.parameters() if p.requires_grad)
  
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
    
def extract_layers(model, layer_list, supported_block_type=SUPPORTED_BLOCK_TYPE, supported_layer_type=SUPPORTED_LAYER_TYPE):
    '''
    Recursively obtain layers of given network
    
    Parameters
    -----------
    model: nn.Module
        The nueral network to extrat all MLP and CNN layers
    layer_list: list
        list containing all supported layers
    '''
    for layer in model.children():
        if type(layer) in supported_block_type:
            # if sequential layer, apply recursively to layers in sequential layer
            extract_layers(layer, layer_list, supported_block_type, supported_layer_type)
        if not list(layer.children()) and type(layer) in supported_layer_type:
            # if leaf node, add it to list
            layer_list.append(layer) 


def eval_sparsity(model):
    '''
    Compute the propotion of 0 in a network.
    
    Parameters
    ----------
    model: nn.Module
        The module to evaluate sparsity
    
    Returns
    -------
    A float capturing the proption of 0 in all the considered params.
    '''
    layers = []
    extract_layers(model, layers)
    supported_layers = [l for l in layers if type(l) in SUPPORTED_LAYER_TYPE]
    total_param = 0
    num_of_zero = 0
    
    for l in supported_layers:
        if l.weight is not None:
            total_param += l.weight.numel()
            num_of_zero += l.weight.eq(0).sum().item()
        if l.bias is not None:
            total_param += l.bias.numel()
            num_of_zero += l.bias.eq(0).sum().item()
    return np.around(num_of_zero / total_param, 4)
  
def fusion_layers_inplace(model, device):
  '''
  Let a convolutional layer fuse with its subsequent batch normalization layer  
  
  Parameters
  -----------
  model: nn.Module
      The nueral network to extrat all CNN and BN layers
  '''
  model_layers = []
  extract_layers(model, model_layers, supported_layer_type = [nn.Conv2d, nn.BatchNorm2d])

  if len(model_layers) < 2:
      return 
  
  for i in range(len(model_layers)-1):
      curr_layer, next_layer = model_layers[i], model_layers[i+1]

      if isinstance(curr_layer, nn.Conv2d) and isinstance(next_layer, nn.BatchNorm2d):
          cnn_layer, bn_layer = curr_layer, next_layer
          # update the weight and bias of the CNN layer 
          bn_scaled_weight = bn_layer.weight.data / torch.sqrt(bn_layer.running_var + bn_layer.eps)
          bn_scaled_bias = bn_layer.bias.data - bn_layer.weight.data * bn_layer.running_mean / torch.sqrt(bn_layer.running_var + bn_layer.eps)
          cnn_layer.weight.data = cnn_layer.weight.data * bn_scaled_weight[:, None, None, None]
          # update the parameters in the BN layer 
          bn_layer.running_var = torch.ones(bn_layer.num_features, device=device)
          bn_layer.running_mean = torch.zeros(bn_layer.num_features, device=device)
          bn_layer.weight.data = torch.ones(bn_layer.num_features, device=device)
          bn_layer.eps = 0.

          if cnn_layer.bias is None:
              bn_layer.bias.data = bn_scaled_bias
          else:
              cnn_layer.bias.data = cnn_layer.bias.data * bn_scaled_weight + bn_scaled_bias 
              bn_layer.bias.data = torch.zeros(bn_layer.num_features, device=device)
              
def enforce_sparsity(model):
    """Reapply pruning masks to ensure pruned weights remain zero."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Handle floating-point layers
                if hasattr(module, 'weight_mask'):
                    module.weight.data *= module.weight_mask
                if hasattr(module, 'bias_mask') and module.bias is not None:
                    module.bias.data *= module.bias_mask

            elif isinstance(module, (nn.quantized.Conv2d, nn.quantized.Linear)):
                # Handle quantized layers
                if hasattr(module, 'weight_mask'):
                    # Dequantize, apply mask, and re-quantize
                    float_weights = module.weight().dequantize()
                    float_weights *= module.weight_mask
                    q_scale = module.weight().q_scale()
                    q_zero_point = module.weight().q_zero_point()
                    quantized_weights = torch.quantize_per_tensor(float_weights, scale=q_scale, zero_point=q_zero_point, dtype=torch.qint8)
                    module._weight_bias()[0].data = quantized_weights