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

class TeacherNetworkUntrained(nn.Module):
    def __init__(self):
        super(TeacherNetworkUntrained, self).__init__()  # Corrected the use of super()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # Adjust for CIFAR-10
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
    
    def forward(self, x):
        return self.model(x)

class TeacherNetworkBiT(nn.Module):
    def __init__(self):
        super(TeacherNetworkBiT, self).__init__()
        weights_cifar10 = get_weights('BiT-M-R50x1-CIFAR10')
        self.model = ResNetV2(ResNetV2.BLOCK_UNITS['r50'], width_factor=1, head_size=10)  # NOTE: No new head.
        self.model.load_from(weights_cifar10)
        self.resize = Resize((128, 128))  # Define the resize transform

    
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
            self.model = self.clone_model(TeacherNetworkUntrained())
            #self.model = self.clone_model(teacher_net)
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
    