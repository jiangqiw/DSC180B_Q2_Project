import torch
import torch.nn as nn
import torchvision.models as models

import torch.quantization
from torch.quantization import QuantStub, DeQuantStub

from quantized_resnet18 import QuantizedResNet18

class TeacherNetwork(nn.Module):
    def __init__(self):
        super(TeacherNetwork, self).__init__()
        # Load a pre-trained ResNet-18 and adjust the final layer for 10 classes (CIFAR-10)
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # CIFAR-10 has 10 classes
    
    def forward(self, x):
        return self.model(x)

def apply_pruning_to_layer(layer, amount):
    """ Prune 'amount' fraction of weights and biases in the layer by setting them to zero based on magnitude. """
    with torch.no_grad():
        # Prune weights
        weight_threshold = torch.quantile(torch.abs(layer.weight.data), amount)
        weight_mask = torch.abs(layer.weight.data) > weight_threshold
        layer.weight.data *= weight_mask  # Zero out small weights
        layer.register_buffer('weight_mask', weight_mask.float())
        # Prune biases, if they exist
        if layer.bias is not None:
            bias_threshold = torch.quantile(torch.abs(layer.bias.data), amount)
            bias_mask = torch.abs(layer.bias.data) > bias_threshold
            layer.bias.data *= bias_mask  # Zero out small biases

def prune_network(model, prune_amount):
    """ Apply pruning to each convolutional and linear layer in the model. """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            apply_pruning_to_layer(module, amount=prune_amount)
            
def apply_gradient_masking(model):
    """ Modify gradients to ensure pruned weights remain zero. """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_mask'):
                module.weight.grad *= module.weight_mask
            if hasattr(module, 'bias_mask') and module.bias is not None:
                module.bias.grad *= module.bias_mask
                
            
def apply_gradient_masking_quantized(model):
    """
    Modify gradients to ensure pruned weights remain zero in a quantized model.
    This ensures gradients for pruned weights are masked during backpropagation.
    """
    for module in model.modules():
        # Check for quantized layers
        if isinstance(module, (nn.quantized.Conv2d, nn.quantized.Linear)):
            # Ensure the module has a weight_mask attribute
            if hasattr(module, 'weight_mask') and module.weight is not None:
                # Dequantize the weight gradients, apply the mask, and re-quantize
                float_grad = module.weight.grad.dequantize()
                float_grad *= module.weight_mask
                q_scale = module.weight.grad.q_scale()
                q_zero_point = module.weight.grad.q_zero_point()
                module.weight.grad = torch.quantize_per_tensor(float_grad, q_scale, q_zero_point, dtype=torch.qint8)
            # Mask the bias gradients if present
            if hasattr(module, 'bias_mask') and module.bias is not None:
                module.bias.grad *= module.bias_mask

    
class StudentNetwork(nn.Module):
    def __init__(self, prune_amount, teacher_net, q=False, fuse=False, qat=False):
        super(StudentNetwork, self).__init__()
        self.model = self.clone_model(teacher_net)
        if q:
            temp_state_dict = self.model.model.state_dict()
            self.model.model = QuantizedResNet18()
            self.model.model.load_state_dict(temp_state_dict)
        if fuse:
            self.model.model.eval()
            self.model.model.fuse_model(is_qat=qat)
            self.model.model.train()
            
        # Prune the network weights with user-defined pruning amount
        prune_network(self.model, prune_amount=prune_amount)
        self.q = q
        if self.q:
          self.quant = QuantStub()
          self.dequant = DeQuantStub()

    def forward(self, x):
        if self.q:
            x = self.quant(x)
        x = self.model(x)
        if self.q:
            x = self.dequant(x)
        return x

    def clone_model(self, model):
        """Deep copy of the model to ensure the original model is not modified."""
        # Create a new instance of the same class
        cloned_model = type(model)()  # Assuming the model has a default constructor
        # Copy the weights and buffers
        cloned_model.load_state_dict(model.state_dict())
        return cloned_model
    