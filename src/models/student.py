from copy import deepcopy
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
from copy import deepcopy
from teachers import TeacherNetworkR18
from model_utils import modify_resnet
from quantized_resnet18 import QuantizedResNet18


class StudentNetwork(nn.Module):
    def __init__(self,  teacher_net, q=False, fuse=False, qat=False, dif_arch = False):
        super(StudentNetwork, self).__init__()
        if dif_arch:
            self.model = self.clone_model(TeacherNetworkR18())
        else:
            self.model = self.clone_model(teacher_net)
        if q:
            temp_state_dict = self.model.model.state_dict()
            self.model.model = QuantizedResNet18()
            self.model = modify_resnet(self.model, 10)
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

    def clone_model(self, model: nn.Module) -> nn.Module:
        return deepcopy(model)
