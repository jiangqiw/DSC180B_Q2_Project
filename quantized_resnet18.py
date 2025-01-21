import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models.quantization.utils import _fuse_modules
from typing import Optional

# Define a custom BasicBlock with nn.quantized.FloatFunctional for addition
class QuantizedBasicBlock(BasicBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_func = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Use FloatFunctional for addition
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add_func.add(out, identity)
        out = self.relu(out)

        return out
    
    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        _fuse_modules(self, [["conv1", "bn1", "relu"], ["conv2", "bn2"]], is_qat, inplace=True)
        if self.downsample:
            _fuse_modules(self.downsample, ["0", "1"], is_qat, inplace=True)

# Define a custom ResNet18 using the modified QuantizedBasicBlock
class QuantizedResNet18(ResNet):
    def __init__(self):
        super().__init__(
            block=QuantizedBasicBlock,
            layers=[2, 2, 2, 2],
        )
        self.fc = nn.Linear(self.fc.in_features, 10)
        
    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        r"""Fuse conv/bn/relu modules in resnet models

        Fuse conv+bn+relu/ Conv+relu/conv+Bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """
        _fuse_modules(self, ["conv1", "bn1", "relu"], is_qat, inplace=True)
        for m in self.modules():
            if type(m) is QuantizedBasicBlock:
                m.fuse_model(is_qat)