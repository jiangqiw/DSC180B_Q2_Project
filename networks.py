import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.utils.prune as prune

class TeacherNetwork(nn.Module):
    def __init__(self):
        super(TeacherNetwork, self).__init__()
        # Load a pre-trained ResNet-18 and adjust the final layer for 10 classes (CIFAR-10)
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        return self.model(x)

def apply_pruning_with_masks(model, prune_amount):
    """Apply global pruning and save masks for all prunable layers."""
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Apply pruning to weights
            prune.l1_unstructured(module, name='weight', amount=prune_amount)
            masks[f"{name}.weight"] = module.weight_mask.clone()
            prune.remove(module, 'weight')  # Permanently remove pruning mask

            # Apply pruning to biases if they exist
            if module.bias is not None:
                prune.l1_unstructured(module, name='bias', amount=prune_amount)
                masks[f"{name}.bias"] = module.bias_mask.clone()
                prune.remove(module, 'bias')  # Permanently remove pruning mask
    return masks

class StudentNetwork(nn.Module):
    def __init__(self, prune_amount, teacher_net):
        super(StudentNetwork, self).__init__()
        self.model = self.clone_model(teacher_net)
        self.masks = self.apply_global_pruning(prune_amount)

    def apply_global_pruning(self, prune_amount):
        """Apply global pruning and save masks."""
        return apply_pruning_with_masks(self.model, prune_amount)

    def forward(self, x):
        return self.model(x)

    def clone_model(self, model):
        """Deep copy of the model to ensure the original model is not modified."""
        cloned_model = type(model)()
        cloned_model.load_state_dict(model.state_dict())
        return cloned_model