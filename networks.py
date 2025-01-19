import torch
import torch.nn as nn
import torchvision.models as models

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

class StudentNetwork(nn.Module):
    def __init__(self, prune_amount, teacher_net):
        super(StudentNetwork, self).__init__()
        self.model = self.clone_model(teacher_net)

        # Prune the network weights with user-defined pruning amount
        prune_network(self.model, prune_amount=prune_amount)

    def forward(self, x):
        return self.model(x)

    def clone_model(self, model):
        """Deep copy of the model to ensure the original model is not modified."""
        # Create a new instance of the same class
        cloned_model = type(model)()  # Assuming the model has a default constructor
        # Copy the weights and buffers
        cloned_model.load_state_dict(model.state_dict())
        return cloned_model