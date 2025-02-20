from copy import deepcopy
import torch.nn as nn
import torchvision
from torch.quantization import QuantStub, DeQuantStub
from copy import deepcopy
from .teachers import TeacherNetworkR18
from utilities.model_utils import modify_resnet
from .quantized_resnet18 import QuantizedResNet18


class StudentNetwork(nn.Module):
    def __init__(self, quantized=False, fuse=False, dataset='cifar10'):
        super(StudentNetwork, self).__init__()
        if quantized:
            self.model = QuantizedResNet18()
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
        else:
            self.model = torchvision.models.resnet18()
            self.quant = nn.Identity()
            self.dequant = nn.Identity()
            
        if fuse:
            self.model.eval()
            self.model.fuse_model(is_qat=quantized)
            self.model.train()
        if dataset not in ['cifar10', 'cifar100']:
            raise NotImplementedError('Dataset not implemented')
        if dataset=='cifar10':
            self.model = modify_resnet(self.model, 10)
        elif dataset=='cifar100':
            self.model = modify_resnet(self.model, 100)


    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

    def clone_model(self, model: nn.Module) -> nn.Module:
        return deepcopy(model)
