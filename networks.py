import torch
import torch.nn as nn
import torchvision.models as models

import torch.quantization
from torch.quantization import QuantStub, DeQuantStub

from quantized_resnet18 import QuantizedResNet18

from resnetv2 import ResNetV2

from utils import get_weights

from torchvision.transforms import Resize
from torchvision.transforms.functional import to_pil_image, to_tensor

import timm

class TeacherNetwork(nn.Module):
    def __init__(self):
        super(TeacherNetwork, self).__init__()
        # Load a pre-trained ResNet-18 and adjust the final layer for 10 classes (CIFAR-10)
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # CIFAR-10 has 10 classes
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.maxpool = nn.Identity()
    
    def forward(self, x):
        return self.model(x)

class TeacherNetwork50(nn.Module):
    def __init__(self):
        super(TeacherNetwork50, self).__init__()
        # Load a pre-trained ResNet-18 and adjust the final layer for 10 classes (CIFAR-10)
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # CIFAR-100 has 100 classes
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.maxpool = nn.Identity()
    
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

class TeacherNetworkBiT(nn.Module):
    def __init__(self, size='R50x1'):
        super(TeacherNetworkBiT, self).__init__()
        if size=='R50x1':
            weights_cifar10 = get_weights('BiT-M-R50x1-CIFAR10')
            self.model = ResNetV2(ResNetV2.BLOCK_UNITS['r50'], width_factor=1, head_size=10)  # NOTE: No new head.
            self.model.load_from(weights_cifar10)
        else:
            self.model = ResNetV2(ResNetV2.BLOCK_UNITS['r101'], width_factor=1, head_size=10)  # NOTE: No new head.
            checkpoint = torch.load('BiT-M-R101x1_step5000.tar')
            state_dict = checkpoint['model']
            new_state_dict = {}

            for key in state_dict.keys():
                new_key = key.replace("module.", "")  # Remove "module." prefix
                new_state_dict[new_key] = state_dict[key]

            self.model.load_state_dict(new_state_dict)
        self.resize = Resize((128, 128))  # Define the resize transform
        
class TeacherNetworkViT(nn.Module):
    def __init__(self):
        super(TeacherNetworkViT, self).__init__()
        self.model = timm.create_model("vit_base_patch16_384", pretrained=True)
        self.model.head = nn.Linear(net.head.in_features, 10)
        checkpoint = torch.load('vit_cifar10_finetune.t7')
        self.model.load_state_dict(checkpoint['model'])
        self.resize = Resize(384)
        
    def forward(self, x):
        x = self.resize(x)
        return self.model(x)

    
    def forward(self, x):
        # Check if x is a batch
        if len(x.shape) == 4:  # Batch of tensors (B, C, H, W)
            # Upscale for BiT
            x = torch.nn.functional.interpolate(
                x, size=(128, 128), mode="bilinear", align_corners=False
            )
        else:
            raise ValueError("Input must be a 4D tensor with shape (B, C, H, W)")

        # Pass the resized tensor batch to the model
        return self.model(x)

class EnsembleModel(nn.Module):
    def __init__(self, model1, model2):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        # Get logits from both models
        logits1 = self.model1(x)
        logits2 = self.model2(x)
        
        # Average the logits
        averaged_logits = (logits1 + logits2) / 2
        return averaged_logits
    
class StudentNetwork(nn.Module):
    def __init__(self, pruning_factor,  teacher_net, q=False, fuse=False, qat=False, dif_arch = False):
        super(StudentNetwork, self).__init__()
        if dif_arch:
            self.model = self.clone_model(TeacherNetwork())
        else:
            self.model = self.clone_model(teacher_net)
        if q:
            temp_state_dict = self.model.model.state_dict()
            self.model.model = QuantizedResNet18()
            self.model.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.model.model.maxpool = nn.Identity()
            self.model.model.load_state_dict(temp_state_dict)
        if fuse:
            self.model.model.eval()
            self.model.model.fuse_model(is_qat=qat)
            self.model.model.train()
            
        if q:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
        else:
            self.quant = nn.Identity()
            self.dequant = nn.Identity()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

    def clone_model(self, model):
        """Deep copy of the model to ensure the original model is not modified."""
        # Create a new instance of the same class
        cloned_model = type(model)()  # Assuming the model has a default constructor
        # Copy the weights and buffers
        cloned_model.load_state_dict(model.state_dict())
        return cloned_model
    