import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from networks import apply_gradient_masking, apply_gradient_masking_quantized
import numpy as np
import os

class InterruptException(Exception):
    pass


def trainStep(network, criterion, optimizer, X, y):
	"""
	One training step of the network: forward prop + backprop + update parameters
	Return: (loss, accuracy) of current batch
	"""
	optimizer.zero_grad()
	outputs = network(X)
	loss = criterion(outputs, y)
	loss.backward()
	optimizer.step()
	accuracy = float(torch.sum(torch.argmax(outputs, dim=1) == y).item()) / y.shape[0]
	return loss, accuracy

def getLossAccuracyOnDataset(network, dataset_loader, fast_device, criterion=None):
	"""
	Returns (loss, accuracy) of network on given dataset
	"""
	network.is_training = False
	accuracy = 0.0
	loss = 0.0
	dataset_size = 0
	for j, D in enumerate(dataset_loader, 0):
		X, y = D
		X = X.to(fast_device)
		y = y.to(fast_device)
		with torch.no_grad():
			pred = network(X)
			if criterion is not None:
				loss += criterion(pred, y) * y.shape[0]
			accuracy += torch.sum(torch.argmax(pred, dim=1) == y).item()
		dataset_size += y.shape[0]
	loss, accuracy = loss / dataset_size, accuracy / dataset_size
	network.is_training = True
	return loss, accuracy

def trainTeacherOnHparam(teacher_net, hparam, num_epochs, 
							train_loader, val_loader, 
							print_every=0, 
							fast_device=torch.device('cuda:0')):
	"""
	Trains teacher on given hyperparameters for given number of epochs; Pass val_loader=None when not required to validate for every epoch 
	Return: List of training loss, accuracy for each update calculated only on the batch; List of validation loss, accuracy for each epoch
	"""
	train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
	teacher_net.dropout_input = hparam['dropout_input']
	teacher_net.dropout_hidden = hparam['dropout_hidden']
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(teacher_net.parameters(), lr=hparam['lr'], momentum=hparam['momentum'], weight_decay=hparam['weight_decay'])
	lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

	# Early stopping variables
	best_val_loss = float('inf')
	best_val_acc = 0.0
	epochs_without_improvement = 0
	patience = 5
	best_model_state = None

	for epoch in range(num_epochs):
		lr_scheduler.step()
		teacher_net.train()
		train_loss, train_acc = 0.0, 0.0

		for i, data in enumerate(train_loader, 0):
			X, y = data
			X, y = X.to(fast_device), y.to(fast_device)
			optimizer.zero_grad()
			outputs = teacher_net(X)
			loss = criterion(outputs, y)
			loss.backward()
			optimizer.step()

			train_loss += loss.item() * X.size(0)
			train_acc += (outputs.argmax(1) == y).sum().item()

			if print_every > 0 and i % print_every == print_every - 1:
				print('[%d, %5d/%5d] train loss: %.3f train accuracy: %.3f' %
					  (epoch + 1, i + 1, len(train_loader), loss.item(), (outputs.argmax(1) == y).float().mean().item()))

		train_loss /= len(train_loader.dataset)
		train_acc /= len(train_loader.dataset)
		train_loss_list.append(train_loss)
		train_acc_list.append(train_acc)

		# Validation
		if val_loader is not None:
			teacher_net.eval()
			val_loss, val_acc = getLossAccuracyOnDataset(teacher_net, val_loader, fast_device, criterion)
			val_loss_list.append(val_loss)
			val_acc_list.append(val_acc)
			print('Epoch: %d, Train Loss: %.4f, Train Acc: %.4f, Val Loss: %.4f, Val Acc: %.4f' %
				  (epoch + 1, train_loss, train_acc, val_loss, val_acc))

			# Check for early stopping
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				best_val_acc = val_acc
				epochs_without_improvement = 0
				best_model_state = teacher_net.state_dict()
			else:
				epochs_without_improvement += 1

			if epochs_without_improvement >= patience:
				print(f"Early stopping triggered after {epoch + 1} epochs.")
				break

	# Fine-tuning
	if best_model_state is not None:
		print("Fine-tuning the model.")
		teacher_net.load_state_dict(best_model_state)
		optimizer = optim.SGD(teacher_net.parameters(), lr=hparam['lr'] * 0.1, momentum=hparam['momentum'], weight_decay=hparam['weight_decay'])
		for epoch in range(5):  # Fine-tune for 5 epochs
			teacher_net.train()
			for i, data in enumerate(train_loader, 0):
				X, y = data
				X, y = X.to(fast_device), y.to(fast_device)
				optimizer.zero_grad()
				outputs = teacher_net(X)
				loss = criterion(outputs, y)
				loss.backward()
				optimizer.step()

			# Optional: Validation during fine-tuning
			if val_loader is not None:
				val_loss, val_acc = getLossAccuracyOnDataset(teacher_net, val_loader, fast_device, criterion)
				print('Fine-tuning Epoch: %d, Val Loss: %.4f, Val Acc: %.4f' %
					  (epoch + 1, val_loss, val_acc))

	return {'train_loss': train_loss_list, 
			'train_acc': train_acc_list, 
			'val_loss': val_loss_list, 
			'val_acc': val_acc_list}

def studentTrainStep(teacher_net, student_net, studentLossFn, optimizer, X, y, T, alpha):
	"""
	One training step of student network: forward prop + backprop + update parameters
	Return: (loss, accuracy) of current batch
	"""
	optimizer.zero_grad()
	teacher_pred = None
	if (alpha > 0):
		with torch.no_grad():
			teacher_pred = teacher_net(X)
	student_pred = student_net(X)
	loss = studentLossFn(teacher_pred, student_pred, y, T, alpha)
	loss.backward()
	apply_gradient_masking_quantized(student_net)
	optimizer.step()
	enforce_sparsity(student_net)
	accuracy = float(torch.sum(torch.argmax(student_pred, dim=1) == y).item()) / y.shape[0]
	return loss, accuracy

def trainStudentOnHparam(teacher_net, student_net, hparam, num_epochs, 
						train_loader, val_loader, 
						print_every=0, 
						fast_device=torch.device('cuda:0'),
      					quant=False):
	"""
	Trains teacher on given hyperparameters for given number of epochs; Pass val_loader=None when not required to validate for every epoch
	Return: List of training loss, accuracy for each update calculated only on the batch; List of validation loss, accuracy for each epoch
	"""
	train_loss_list, train_acc_list, val_acc_list = [], [], []
	T = hparam['T']
	alpha = hparam['alpha']
	student_net.dropout_input = hparam['dropout_input']
	student_net.dropout_hidden = hparam['dropout_hidden']
	optimizer = optim.SGD(student_net.parameters(), lr=hparam['lr'], momentum=hparam['momentum'], weight_decay=hparam['weight_decay'])
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=hparam['lr_decay'])

	def studentLossFn(teacher_pred, student_pred, y, T, alpha):
		"""
		Loss function for student network: Loss = alpha * (distillation loss with soft-target) + (1 - alpha) * (cross-entropy loss with true label)
		Return: loss
		"""
		if (alpha > 0):
			loss = F.kl_div(F.log_softmax(student_pred / T, dim=1), F.softmax(teacher_pred / T, dim=1), reduction='batchmean') * (T ** 2) * alpha + F.cross_entropy(student_pred, y) * (1 - alpha)
		else:
			loss = F.cross_entropy(student_pred, y)
		return loss

	for epoch in range(num_epochs):
		lr_scheduler.step()
		if epoch == 0:
			if val_loader is not None:
				_, val_acc = getLossAccuracyOnDataset(student_net, val_loader, fast_device)
				val_acc_list.append(val_acc)
				print('epoch: %d validation accuracy: %.3f' %(epoch, val_acc))
		for i, data in enumerate(train_loader, 0):
			X, y = data
			X, y = X.to(fast_device), y.to(fast_device)
			loss, acc = studentTrainStep(teacher_net, student_net, studentLossFn, optimizer, X, y, T, alpha)
			train_loss_list.append(loss)
			train_acc_list.append(acc)
		
			if print_every > 0 and i % print_every == print_every - 1:
				print('[%d, %5d/%5d] train loss: %.3f train accuracy: %.3f' %
					  (epoch + 1, i + 1, len(train_loader), loss, acc))
	
		if val_loader is not None:
			_, val_acc = getLossAccuracyOnDataset(student_net, val_loader, fast_device)
			val_acc_list.append(val_acc)
			print('epoch: %d validation accuracy: %.3f' %(epoch + 1, val_acc))
		if quant:
			if epoch > 3:
				# Freeze quantizer parameters
				student_net.apply(torch.ao.quantization.disable_observer)
			if epoch > 2:
				# Freeze batch norm mean and variance estimates
				student_net.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
	return {'train_loss': train_loss_list, 
			'train_acc': train_acc_list, 
			'val_acc': val_acc_list}

def hparamToString(hparam):
	"""
	Convert hparam dictionary to string with deterministic order of attribute of hparam in output string
	"""
	hparam_str = ''
	for k, v in sorted(hparam.items()):
		hparam_str += k + '=' + str(v) + ', '
	return hparam_str[:-2]

def hparamDictToTuple(hparam):
	"""
	Convert hparam dictionary to tuple with deterministic order of attribute of hparam in output tuple
	"""
	hparam_tuple = [v for k, v in sorted(hparam.items())]
	return tuple(hparam_tuple)

def getTrainMetricPerEpoch(train_metric, updates_per_epoch):
	"""
	Smooth the training metric calculated for each batch of training set by averaging over batches in an epoch
	Input: List of training metric calculated for each batch
	Output: List of training matric averaged over each epoch
	"""
	train_metric_per_epoch = []
	temp_sum = 0.0
	for i in range(len(train_metric)):
		temp_sum += train_metric[i]
		if (i % updates_per_epoch == updates_per_epoch - 1):
			train_metric_per_epoch.append(temp_sum / updates_per_epoch)
			temp_sum = 0.0

	return train_metric_per_epoch

def fusion_layers_inplace(model, device):
    '''
    Let a convolutional layer fuse with its subsequent batch normalization layer  
    
    Parameters
    -----------
    model: nn.Module
        The nueral network to extrat all CNN and BN layers
    '''
    model_layers = []
    extract_layers(model, model_layers, supported_layer_type = [nn.Conv2d, nn.BatchNorm2d])

    if len(model_layers) < 2:
        return 
    
    for i in range(len(model_layers)-1):
        curr_layer, next_layer = model_layers[i], model_layers[i+1]

        if isinstance(curr_layer, nn.Conv2d) and isinstance(next_layer, nn.BatchNorm2d):
            cnn_layer, bn_layer = curr_layer, next_layer
            # update the weight and bias of the CNN layer 
            bn_scaled_weight = bn_layer.weight.data / torch.sqrt(bn_layer.running_var + bn_layer.eps)
            bn_scaled_bias = bn_layer.bias.data - bn_layer.weight.data * bn_layer.running_mean / torch.sqrt(bn_layer.running_var + bn_layer.eps)
            cnn_layer.weight.data = cnn_layer.weight.data * bn_scaled_weight[:, None, None, None]
            # update the parameters in the BN layer 
            bn_layer.running_var = torch.ones(bn_layer.num_features, device=device)
            bn_layer.running_mean = torch.zeros(bn_layer.num_features, device=device)
            bn_layer.weight.data = torch.ones(bn_layer.num_features, device=device)
            bn_layer.eps = 0.

            if cnn_layer.bias is None:
                bn_layer.bias.data = bn_scaled_bias
            else:
                cnn_layer.bias.data = cnn_layer.bias.data * bn_scaled_weight + bn_scaled_bias 
                bn_layer.bias.data = torch.zeros(bn_layer.num_features, device=device)
                
                
from torchvision.models.resnet import BasicBlock as tBasicBlock
from torchvision.models.resnet import Bottleneck as tBottleneck 
from torchvision.models.resnet import ResNet as tResNet
from torchvision.models.googlenet import BasicConv2d, Inception, InceptionAux
from torchvision.models.efficientnet import Conv2dNormActivation, SqueezeExcitation, MBConv 
from torchvision.models.mobilenetv2 import InvertedResidual

SUPPORTED_LAYER_TYPE = {nn.Linear, nn.Conv2d}
SUPPORTED_BLOCK_TYPE = {nn.Sequential, 
                        tBottleneck, tBasicBlock, tResNet,
                        BasicConv2d, Inception, InceptionAux,
                        Conv2dNormActivation, SqueezeExcitation, MBConv,
                        InvertedResidual
                        }
def extract_layers(model, layer_list, supported_block_type=SUPPORTED_BLOCK_TYPE, supported_layer_type=SUPPORTED_LAYER_TYPE):
    '''
    Recursively obtain layers of given network
    
    Parameters
    -----------
    model: nn.Module
        The nueral network to extrat all MLP and CNN layers
    layer_list: list
        list containing all supported layers
    '''
    for layer in model.children():
        if type(layer) in supported_block_type:
            # if sequential layer, apply recursively to layers in sequential layer
            extract_layers(layer, layer_list, supported_block_type, supported_layer_type)
        if not list(layer.children()) and type(layer) in supported_layer_type:
            # if leaf node, add it to list
            layer_list.append(layer) 


def eval_sparsity(model):
    '''
    Compute the propotion of 0 in a network.
    
    Parameters
    ----------
    model: nn.Module
        The module to evaluate sparsity
    
    Returns
    -------
    A float capturing the proption of 0 in all the considered params.
    '''
    layers = []
    extract_layers(model, layers)
    supported_layers = [l for l in layers if type(l) in SUPPORTED_LAYER_TYPE]
    total_param = 0
    num_of_zero = 0
    
    for l in supported_layers:
        if l.weight is not None:
            total_param += l.weight.numel()
            num_of_zero += l.weight.eq(0).sum().item()
        if l.bias is not None:
            total_param += l.bias.numel()
            num_of_zero += l.bias.eq(0).sum().item()
    return np.around(num_of_zero / total_param, 4)

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
    
def enforce_sparsity(model):
    """Reapply pruning masks to ensure pruned weights remain zero."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Handle floating-point layers
                if hasattr(module, 'weight_mask'):
                    module.weight.data *= module.weight_mask
                if hasattr(module, 'bias_mask') and module.bias is not None:
                    module.bias.data *= module.bias_mask

            elif isinstance(module, (nn.quantized.Conv2d, nn.quantized.Linear)):
                # Handle quantized layers
                if hasattr(module, 'weight_mask'):
                    # Dequantize, apply mask, and re-quantize
                    float_weights = module.weight().dequantize()
                    float_weights *= module.weight_mask
                    q_scale = module.weight().q_scale()
                    q_zero_point = module.weight().q_zero_point()
                    quantized_weights = torch.quantize_per_tensor(float_weights, scale=q_scale, zero_point=q_zero_point, dtype=torch.qint8)
                    module._weight_bias()[0].data = quantized_weights