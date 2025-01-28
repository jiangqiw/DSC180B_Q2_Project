import torch
import torch.ao.quantization
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.intrinsic
import torch.optim as optim
import numpy as np
import os
import requests
import io

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
    optimizer.step()
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
            if epoch%5==0:
                create_pruning_masks_quantized(student_net, (epoch/5)/10)
                student_net.apply(torch.ao.quantization.enable_observer)
                student_net.apply(torch.nn.intrinsic.qat.update_bn_stats)
            if epoch % 3:
                # Freeze quantizer parameters
                student_net.apply(torch.ao.quantization.disable_observer)
            if epoch % 2:
                # Freeze batch norm mean and variance estimates
                student_net.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    return {'train_loss': train_loss_list, 
            'train_acc': train_acc_list, 
            'val_acc': val_acc_list}
def studentTrainStepMixup(
    teacher_net,
    student_net,
    studentLossFn,
    optimizer,
    X,
    y,
    T,
    alpha,
    mixup_alpha=1.0,
    update_teacher=False,
    teacher_optimizer=None,
    teacherLossFn=None
):
    """
    One training step of the student network with mixup-based function matching.

    Args:
        teacher_net (nn.Module): Teacher model.
        student_net (nn.Module): Student model.
        studentLossFn (function): Loss function for the student.
        optimizer (Optimizer): Optimizer for the student network.
        X (torch.Tensor): Input images (batch_size, channels, height, width).
        y (torch.Tensor): Ground-truth labels (batch_size,).
        T (float): Temperature for distillation.
        alpha (float): Weight for distillation loss.
        mixup_alpha (float): Mixup hyperparameter. If 0, no mixup is applied.
        update_teacher (bool): If True, update the teacher network.
        teacher_optimizer (Optimizer): Optimizer for the teacher network.
        teacherLossFn (function): Loss function for updating the teacher.

    Returns:
        tuple: Loss and accuracy for the current batch.
    """
    optimizer.zero_grad()
    teacher_net.eval()
    # Convert labels to one-hot encoding
    y_onehot = F.one_hot(y, num_classes=10).float()

    # Compute teacher logits
    with torch.no_grad():
        teacher_logits = teacher_net(X)


    # Apply aggressive mixup
    mixed_X, mixed_teacher_logits, mixed_labels, lam = mixup_function_matching(
        X, teacher_logits, y_onehot, alpha=mixup_alpha
    )
    

    # Forward pass through the student network
    student_logits = student_net(mixed_X)

    # Compute loss with mixup for the student network
    student_loss = studentLossFn(mixed_teacher_logits, student_logits, mixed_labels, T, alpha)

    # Backpropagation and optimization for the student
    student_loss.backward()
    optimizer.step()

    # Optionally update the teacher network
    if update_teacher and teacher_optimizer is not None and teacherLossFn is not None:
        teacher_optimizer.zero_grad()
                
        teacher_net.train()
        teacher_logits = teacher_net(X)


        # Apply aggressive mixup
        mixed_X, mixed_teacher_logits, mixed_labels, lam = mixup_function_matching(
            X, teacher_logits, y_onehot, alpha=mixup_alpha
        )
        with torch.no_grad():
            student_logits = student_net(mixed_X)
        # Compute the KL divergence between teacher logits and mixed teacher logits
        teacher_loss = teacherLossFn(mixed_teacher_logits, student_logits, mixed_labels, T, alpha)
        # Backpropagation and optimization for the teacher
        teacher_loss.backward()
        teacher_optimizer.step()

    # Calculate accuracy (based on original labels)
    accuracy = (torch.argmax(student_logits, dim=1) == y).float().mean().item()

    return student_loss.item(), accuracy

def trainStudentOnHparamMixup(teacher_net, student_net, hparam, num_epochs, 
                         train_loader, val_loader, 
                         print_every=0, 
                         fast_device=torch.device('cuda:0'),
                         quant=False, mixup_alpha=1.0, checkpoint_save_path = 'checkpoints_student_QAT/', resume_checkpoint = False, optimizer_choice = 'adam'):
    """
    Train the student network using aggressive mixup-based function matching.

    Args:
        teacher_net (nn.Module): Teacher model.
        student_net (nn.Module): Student model.
        hparam (dict): Hyperparameters for training.
        num_epochs (int): Number of epochs.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        print_every (int): Print progress every specified number of batches.
        fast_device (torch.device): Device to run the models on.
        quant (bool): Whether to apply quantization-aware training.
        mixup_alpha (float): Mixup hyperparameter.

    Returns:
        dict: Training and validation metrics (loss and accuracy).
    """
    train_loss_list, train_acc_list, val_acc_list, val_loss_list = [], [], [], []
    T = hparam['T']
    alpha = hparam['alpha']
    student_net.dropout_input = hparam['dropout_input']
    student_net.dropout_hidden = hparam['dropout_hidden']
    
    if optimizer_choice == 'adam':
        optimizer = optim.Adam(student_net.parameters(), lr=hparam['lr'], weight_decay=hparam['weight_decay'], eps=.01)
    else:
        optimizer = optim.SGD(student_net.parameters(), lr = hparam['lr'], momentum=hparam['momentum'], weight_decay=hparam['weight_decay'])
    teacher_optimizer = optim.SGD(teacher_net.parameters(), lr=.0001, momentum=hparam['momentum'], weight_decay=hparam['weight_decay'])

    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint)
        student_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    #optimizer = optim.SGD(student_net.parameters(), lr=hparam['lr'], momentum=hparam['momentum'], weight_decay=hparam['weight_decay'])

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.0)

    def studentLossFn(mixed_teacher_logits, student_logits, mixed_labels, T, alpha):
        """
        Compute the student loss using mixup-enhanced teacher logits and labels.

        Args:
            mixed_teacher_logits (torch.Tensor): Mixed teacher logits.
            student_logits (torch.Tensor): Student predictions.
            mixed_labels (torch.Tensor): Mixed one-hot labels.
            T (float): Temperature for distillation.
            alpha (float): Weight for distillation loss.
            lam (float): Mixup coefficient.

        Returns:
            torch.Tensor: Loss value.
        """
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(mixed_teacher_logits / T, dim=1),
            reduction='batchmean'
        ) * (T ** 2) * alpha + F.cross_entropy(student_logits, mixed_labels) * (1 - alpha)
        
        return distillation_loss
    
    def teacherLossFn(mixed_teacher_logits, student_logits, mixed_labels, T, alpha):
        """
        Compute the student loss using mixup-enhanced teacher logits and labels.

        Args:
            mixed_teacher_logits (torch.Tensor): Mixed teacher logits.
            student_logits (torch.Tensor): Student predictions.
            mixed_labels (torch.Tensor): Mixed one-hot labels.
            T (float): Temperature for distillation.
            alpha (float): Weight for distillation loss.
            lam (float): Mixup coefficient.

        Returns:
            torch.Tensor: Loss value.
        """
        distillation_loss = F.kl_div(
            F.log_softmax(mixed_teacher_logits / T, dim=1),
            F.softmax(student_logits / T, dim=1),
            reduction='batchmean'
        ) * (T ** 2) * alpha

        return distillation_loss
    
    if resume_checkpoint:
        for epoch in range(start_epoch, num_epochs):
            for i, (X, y) in enumerate(train_loader):
                X, y = X.to(fast_device), y.to(fast_device)

                # Perform one training step with mixup
                if False:
                    loss, acc = studentTrainStepMixup(teacher_net, student_net, studentLossFn, optimizer, X, y, T, alpha, mixup_alpha=mixup_alpha, update_teacher=True, teacher_optimizer=teacher_optimizer, teacherLossFn=teacherLossFn)
                else:
                    loss, acc = studentTrainStepMixup(teacher_net, student_net, studentLossFn, optimizer, X, y, T, alpha, mixup_alpha=mixup_alpha)

                train_loss_list.append(loss)
                train_acc_list.append(acc)
                lr_scheduler.step()


                if print_every > 0 and i % print_every == print_every - 1:
                    print(f'[Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)}] Train Loss: {loss:.3f}, Train Accuracy: {acc:.3f}')

            # Validate after each epoch
            if val_loader is not None:
                val_loss, val_acc = getLossAccuracyOnDataset(student_net, val_loader, fast_device)
                val_acc_list.append(val_acc)
                val_loss_list.append(val_loss)
                print(f'Epoch {epoch + 1} Validation Accuracy: {val_acc:.3f}')
                
            if (epoch + 1) % 25 == 0 or epoch + 1 == num_epochs:
                checkpoint_path = checkpoint_save_path + hparamToString(hparam) + f'_checkpoint_epoch_{epoch + 1}.tar'
                torch.save({
                    'model_state_dict': student_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'train_loss': train_loss_list,
                    'train_acc': train_acc_list,
                    'val_loss': val_loss_list,
                    'val_acc': val_acc_list
                }, checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch + 1}: {checkpoint_path}")

            # Apply quantization steps if enabled
            if quant:

                if epoch > 8:
                    student_net.apply(torch.ao.quantization.disable_observer)
                if epoch > 6:
                    student_net.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    else:
        for epoch in range(num_epochs):
            for i, (X, y) in enumerate(train_loader):
                X, y = X.to(fast_device), y.to(fast_device)

                # Perform one training step with mixup
                
                # Need to unhard-code these
                if False:
                    loss, acc = studentTrainStepMixup(teacher_net, student_net, studentLossFn, optimizer, X, y, T, alpha, mixup_alpha=mixup_alpha, update_teacher=True, teacher_optimizer=teacher_optimizer, teacherLossFn=teacherLossFn)
                else:
                    loss, acc = studentTrainStepMixup(teacher_net, student_net, studentLossFn, optimizer, X, y, T, alpha, mixup_alpha=mixup_alpha)

                train_loss_list.append(loss)
                train_acc_list.append(acc)
                
                lr_scheduler.step()


                if print_every > 0 and i % print_every == print_every - 1:
                    print(f'[Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)}] Train Loss: {loss:.3f}, Train Accuracy: {acc:.3f}')
            # Validate after each epoch
            if val_loader is not None:
                val_loss, val_acc = getLossAccuracyOnDataset(student_net, val_loader, fast_device)
                val_acc_list.append(val_acc)
                val_loss_list.append(val_loss)
                print(f'Epoch {epoch + 1} Validation Accuracy: {val_acc:.3f}')
                
            if (epoch + 1) % 25 == 0 or epoch + 1 == num_epochs:
                checkpoint_path = checkpoint_save_path + hparamToString(hparam) + f'_checkpoint_epoch_{epoch + 1}.tar'
                torch.save({
                    'model_state_dict': student_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'train_loss': train_loss_list,
                    'train_acc': train_acc_list,
                    'val_loss': val_loss_list,
                    'val_acc': val_acc_list
                }, checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch + 1}: {checkpoint_path}")

            # Apply quantization steps if enabled
            if quant:
                if epoch > 8:
                    student_net.apply(torch.ao.quantization.disable_observer)
                if epoch > 6:
                    student_net.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    return {
        'train_loss': train_loss_list,
        'train_acc': train_acc_list,
        'val_acc': val_acc_list,
        'val_loss': val_loss_list
    }
    
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
                    
def mixup_function_matching(inputs, teacher_logits, labels, alpha=1.0):
    """
    Apply mixup for function matching with teacher logits and one-hot labels.

    Args:
        inputs (torch.Tensor): Input images (batch_size, channels, height, width).
        teacher_logits (torch.Tensor): Teacher model outputs (batch_size, num_classes).
        labels (torch.Tensor): One-hot encoded labels (batch_size, num_classes).
        alpha (float): Mixup parameter.

    Returns:
        tuple: Mixed inputs, mixed teacher logits, mixed labels, and mixup coefficient lambda.
    """
    if alpha > 0:
        lam = np.random.uniform(0, 1)
    else:
        lam = 1.0

    batch_size = inputs.size(0)
    index = torch.randperm(batch_size)

    # Mix inputs
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]

    # Mix teacher logits and labels
    mixed_teacher_logits = lam * teacher_logits + (1 - lam) * teacher_logits[index, :]
    mixed_labels = lam * labels + (1 - lam) * labels[index, :]

    return mixed_inputs, mixed_teacher_logits, mixed_labels, lam

def get_weights(bit_variant):
  response = requests.get(f'https://storage.googleapis.com/bit_models/{bit_variant}.npz')
  response.raise_for_status()
  return np.load(io.BytesIO(response.content))

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
