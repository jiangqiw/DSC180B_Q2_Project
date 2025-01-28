import torch
import torch.nn as nn


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
                
def create_pruning_masks_quantized(model, pruning_percentage=0.2):
    """
    Create or update binary masks for pruned weights in a quantized model.
    If masks already exist, combine the new masks with the existing ones
    using an element-wise AND operation.

    Parameters:
        model (nn.Module): The quantized model to apply masking.
        pruning_percentage (float): Fraction of weights to prune (0.0 to 1.0).
    """
    print(pruning_percentage)
    for module in model.modules():
        # Check for quantized layers
        if isinstance(module, (nn.quantized.Conv2d, nn.quantized.Linear)):
            # Create or update weight_mask
            if module.weight is not None:
                # Dequantize weight tensor for mask calculation
                float_weights = module.weight.dequantize()
                
                # Calculate threshold for pruning
                threshold = torch.quantile(float_weights.abs(), pruning_percentage)
                
                # Generate new mask: 1 for weights >= threshold, 0 otherwise
                new_mask = (float_weights.abs() >= threshold).float()
                
                if hasattr(module, 'weight_mask'):
                    # Combine existing mask with new mask using logical AND
                    combined_mask = getattr(module, 'weight_mask') * new_mask
                    module.register_buffer('weight_mask', combined_mask)
                else:
                    # Initialize with the new mask if no mask exists
                    module.register_buffer('weight_mask', new_mask)
                

    print("Updated pruning masks for quantized model.")

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
