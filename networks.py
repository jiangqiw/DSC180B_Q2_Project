import torch
import torch.nn as nn
import torchvision.models as models

class TeacherNetwork(nn.Module):
    def __init__(self):
        super(TeacherNetwork, self).__init__()
        # Load a pre-trained ResNet-152 and adjust the final layer for 10 classes (CIFAR-10)
        self.model = models.resnet152(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # CIFAR-10 has 10 classes
    
    def forward(self, x):
        return self.model(x)

def apply_pruning_to_layer(layer, amount):
    """ Prune 'amount' fraction of weights in the layer by setting them to zero based on magnitude. """
    with torch.no_grad():
        threshold = torch.quantile(torch.abs(layer.weight.data), amount)
        mask = torch.abs(layer.weight.data) > threshold
        layer.weight.data *= mask  # Zero out small weights

def prune_network(model, prune_amount):
    """ Apply pruning to each convolutional and linear layer in the model. """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            apply_pruning_to_layer(module, amount=prune_amount)

class StudentNetwork(nn.Module):
    def __init__(self, prune_amount):
        super(StudentNetwork, self).__init__()
        # Load a pre-trained ResNet-152 and adjust for CIFAR-10
        self.model = models.resnet152(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

        # Prune the network weights with user-defined pruning amount
        prune_network(self.model, prune_amount=prune_amount)

    def forward(self, x):
        return self.model(x)

# Note: When adjusting the number of filters or features in layers, ensure the compatibility of layers in terms of input and output sizes.