import torch
import torch.nn as nn
import torchvision.models as models

import torch.quantization

from .resnetv2 import ResNetV2

from utilities.model_utils import get_weights, modify_resnet

from torchvision.transforms import Resize

import timm
import detectors


class TeacherNetworkR18(nn.Module):
    def __init__(self, dataset='cifar10'):
        super(TeacherNetworkR18, self).__init__()
        # Load a pre-trained ResNet-18 and adjust the final layer for 10 classes (CIFAR-10)

        if dataset=='cifar10':
            self.model = modify_resnet(models.resnet18(pretrained=False), 10)    
        elif dataset=='cifar100':
            self.model = modify_resnet(models.resnet18(pretrained=False), 100)
        else:
            raise RuntimeError('Dataset not implemented')

    
    def forward(self, x):
        return self.model(x)
    
    def load_weights(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint)

class TeacherNetworkR50(nn.Module):
    def __init__(self, dataset='cifar10', checkpoint_path=None):
        super(TeacherNetworkR50, self).__init__()
        self.dataset = dataset
        if dataset=='cifar10':
            self.model = modify_resnet(models.resnet50(pretrained=False), 10)
        elif dataset=='cifar100':
            self.model = modify_resnet(models.resnet50(pretrained=False), 100)
        else:
            raise RuntimeError('Dataset not implemented')
        if checkpoint_path:
            self.load_weights(checkpoint_path)

    def forward(self, x):
        return self.model(x)

    def load_weights(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint)
    
    def get_num_classes(self):
        return 10 if self.dataset=='cifar10' else 100
    
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
    
class TeacherNetworkViT(nn.Module):
    def __init__(self):
        super(TeacherNetworkViT, self).__init__()
        self.model = timm.create_model("vit_base_patch16_384", pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, 10)
        checkpoint = torch.load('vit_cifar10_finetune.t7')
        self.model.load_state_dict(checkpoint['model'])
        self.resize = Resize(384)
        
    def forward(self, x):
        x = self.resize(x)
        return self.model(x)

class EnsembleModel(nn.Module):
    def __init__(self, *models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        logits = [model(x) for model in self.models]
        return torch.mean(torch.stack(logits), dim=0)