import torch
import torch.ao.quantization
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.intrinsic
import torch.optim as optim
import numpy as np

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

def student_train_step(teacher_net, student_net, studentLossFn, optimizer, X, y, T, alpha):
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

def train_student_on_hparam(
    teacher_net,
    student_net,
    hparam,
    num_epochs,
    train_loader,
    val_loader=None,
    print_every=0,
    fast_device=torch.device('cuda:0'),
    quant=False,
    checkpoint_save_path=None,
    resume_checkpoint=False,
    optimizer_choice='adam'
):
    train_loss_list, train_acc_list, val_acc_list, val_loss_list = [], [], [], []
    T, alpha = hparam['T'], hparam['alpha']

    student_net.dropout_input = hparam['dropout_input']
    student_net.dropout_hidden = hparam['dropout_hidden']

    optimizer = (optim.Adam(student_net.parameters(), lr=hparam['lr'], weight_decay=hparam['weight_decay'], eps=0.01)
                 if optimizer_choice == 'adam'
                 else optim.SGD(student_net.parameters(), lr=hparam['lr'], momentum=hparam['momentum'], weight_decay=hparam['weight_decay']))

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.0)

    def student_loss_fn(teacher_logits, student_logits, labels, T, alpha):
        return (
            F.kl_div(F.log_softmax(student_logits / T, dim=1), F.softmax(teacher_logits / T, dim=1), reduction='batchmean') * (T ** 2) * alpha +
            F.cross_entropy(student_logits, labels) * (1 - alpha)
        )

    start_epoch = 0
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint)
        student_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    for epoch in range(start_epoch, num_epochs):
        student_net.train()
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(fast_device), y.to(fast_device)

            loss, acc = student_train_step(
                teacher_net, student_net, student_loss_fn, optimizer, X, y, T, alpha
            )

            train_loss_list.append(loss)
            train_acc_list.append(acc)
            lr_scheduler.step()

            if print_every > 0 and i % print_every == print_every - 1:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)}] Loss: {loss:.3f}, Acc: {acc:.3f}')

        if val_loader:
            val_loss, val_acc = getLossAccuracyOnDataset(student_net, val_loader, fast_device)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
            print(f'Epoch {epoch + 1} Validation Acc: {val_acc:.3f}')

        if (epoch + 1) % 25 == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = f"{checkpoint_save_path}{hparamToString(hparam)}_epoch_{epoch + 1}.tar"
            torch.save({
                'model_state_dict': student_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'train_loss': train_loss_list,
                'train_acc': train_acc_list,
                'val_loss': val_loss_list,
                'val_acc': val_acc_list
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

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


def student_train_step_mixup(
    teacher_net,
    student_net,
    student_loss_fn,
    optimizer,
    X,
    y,
    T,
    alpha,
    mixup_alpha=1.0,
    update_teacher=False,
    teacher_optimizer=None,
    teacher_loss_fn=None
):
    optimizer.zero_grad()
    teacher_net.eval()

    y_onehot = F.one_hot(y, num_classes=10).float()

    with torch.no_grad():
        teacher_logits = teacher_net(X)

    mixed_X, mixed_teacher_logits, mixed_labels, _ = mixup_function_matching(
        X, teacher_logits, y_onehot, alpha=mixup_alpha
    )

    student_logits = student_net(mixed_X)
    student_loss = student_loss_fn(mixed_teacher_logits, student_logits, mixed_labels, T, alpha)

    student_loss.backward()
    optimizer.step()

    if update_teacher and teacher_optimizer and teacher_loss_fn:
        teacher_optimizer.zero_grad()
        teacher_net.train()

        teacher_logits = teacher_net(X)
        mixed_X, mixed_teacher_logits, mixed_labels, _ = mixup_function_matching(
            X, teacher_logits, y_onehot, alpha=mixup_alpha
        )

        with torch.no_grad():
            student_logits = student_net(mixed_X)

        teacher_loss = teacher_loss_fn(mixed_teacher_logits, student_logits, mixed_labels, T, alpha)
        teacher_loss.backward()
        teacher_optimizer.step()

    accuracy = (torch.argmax(student_logits, dim=1) == y).float().mean().item()

    return student_loss.item(), accuracy

def train_student_on_hparam_mixup(
    teacher_net,
    student_net,
    hparam,
    num_epochs,
    train_loader,
    val_loader=None,
    print_every=0,
    fast_device=torch.device('cuda:0'),
    quant=False,
    mixup_alpha=1.0,
    checkpoint_save_path='checkpoints_student_QAT/',
    resume_checkpoint=False,
    optimizer_choice='adam'
):
    train_loss_list, train_acc_list, val_acc_list, val_loss_list = [], [], [], []
    T, alpha = hparam['T'], hparam['alpha']

    student_net.dropout_input = hparam['dropout_input']
    student_net.dropout_hidden = hparam['dropout_hidden']

    optimizer = (optim.Adam(student_net.parameters(), lr=hparam['lr'], weight_decay=hparam['weight_decay'], eps=0.01)
                 if optimizer_choice == 'adam'
                 else optim.SGD(student_net.parameters(), lr=hparam['lr'], momentum=hparam['momentum'], weight_decay=hparam['weight_decay']))

    teacher_optimizer = optim.SGD(teacher_net.parameters(), lr=0.0001, momentum=hparam['momentum'], weight_decay=hparam['weight_decay'])
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.0)

    def student_loss_fn(mixed_teacher_logits, student_logits, mixed_labels, T, alpha):
        return (
            F.kl_div(F.log_softmax(student_logits / T, dim=1), F.softmax(mixed_teacher_logits / T, dim=1), reduction='batchmean') * (T ** 2) * alpha +
            F.cross_entropy(student_logits, mixed_labels) * (1 - alpha)
        )

    def teacher_loss_fn(mixed_teacher_logits, student_logits, mixed_labels, T, alpha):
        return F.kl_div(
            F.log_softmax(mixed_teacher_logits / T, dim=1),
            F.softmax(student_logits / T, dim=1),
            reduction='batchmean'
        ) * (T ** 2) * alpha

    start_epoch = 0
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint)
        student_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    for epoch in range(start_epoch, num_epochs):
        student_net.train()
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(fast_device), y.to(fast_device)

            loss, acc = student_train_step_mixup(
                teacher_net, student_net, student_loss_fn, optimizer, X, y, T, alpha,
                mixup_alpha=mixup_alpha, update_teacher=False, teacher_optimizer=teacher_optimizer, teacher_loss_fn=teacher_loss_fn
            )

            train_loss_list.append(loss)
            train_acc_list.append(acc)
            lr_scheduler.step()

            if print_every > 0 and i % print_every == print_every - 1:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)}] Loss: {loss:.3f}, Acc: {acc:.3f}')

        if val_loader:
            val_loss, val_acc = getLossAccuracyOnDataset(student_net, val_loader, fast_device)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
            print(f'Epoch {epoch + 1} Validation Acc: {val_acc:.3f}')

        if (epoch + 1) % 25 == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = f"{checkpoint_save_path}{hparamToString(hparam)}_epoch_{epoch + 1}.tar"
            torch.save({
                'model_state_dict': student_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'train_loss': train_loss_list,
                'train_acc': train_acc_list,
                'val_loss': val_loss_list,
                'val_acc': val_acc_list
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

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


def dkdLoss(teacher_logits, student_logits, target, alpha=1.0, beta=8.0, temperature=4.0):
    """
    Compute the Decoupled Knowledge Distillation (DKD) loss following mdistiller's approach.
    """
    # Compute soft probabilities
    pred_student = F.log_softmax(student_logits / temperature, dim=1)
    pred_teacher = F.softmax(teacher_logits / temperature, dim=1)
    
    # Target mask
    target_mask = F.one_hot(target, num_classes=student_logits.size(1)).bool()
    
    # Target class knowledge distillation (TCKD)
    tckd_loss = F.kl_div(pred_student[target_mask], pred_teacher[target_mask], reduction='batchmean')
    
    # Non-target class knowledge distillation (NCKD)
    nontarget_mask = ~target_mask
    nckd_loss = F.kl_div(pred_student[nontarget_mask], pred_teacher[nontarget_mask], reduction='batchmean')
    
    dkd_loss = alpha * tckd_loss + beta * nckd_loss
    
    return dkd_loss

def studentTrainStepDKD(teacher_net, student_net, optimizer, X, y, alpha=1.0, beta=8.0, temperature=4.0):
    """
    One training step using Decoupled Knowledge Distillation
    """
    optimizer.zero_grad()
    
    with torch.no_grad():
        teacher_logits = teacher_net(X)
    student_logits = student_net(X)
    
    loss = dkdLoss(teacher_logits, student_logits, y, alpha, beta, temperature)
    
    loss.backward()
    optimizer.step()
    
    accuracy = (student_logits.argmax(dim=1) == y).float().mean().item()
    
    return loss.item(), accuracy

def trainStudentWithDKD(teacher_net, student_net, hparam, num_epochs, 
                       train_loader, val_loader, 
                       print_every=0, 
                       fast_device=torch.device('cuda:0'),
                       quant=False, checkpoint_save_path = 'checkpoints_student_DKD/', a=1.0, b=8.0):
    """
    Train student network using Decoupled Knowledge Distillation
    """
    train_loss_list, train_acc_list, val_acc_list, val_loss_list = [], [], [], []
    
    # Set up student network parameters
    student_net.dropout_input = hparam.get('dropout_input', 0.0)
    student_net.dropout_hidden = hparam.get('dropout_hidden', 0.0)
    
    # Initialize optimizer and scheduler
    optimizer = optim.SGD(student_net.parameters(), 
                         lr=hparam['lr'],
                         momentum=hparam['momentum'],
                         weight_decay=hparam['weight_decay'])
    
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # DKD hyperparameters
    alpha = hparam.get('alpha', a)
    beta = hparam.get('beta', b)
    temperature = hparam.get('temperature', 4.0)
    
    for epoch in range(num_epochs):
        # Training phase
        student_net.train()
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(fast_device), y.to(fast_device)
            
            # Training step with DKD
            loss, acc = studentTrainStepDKD(
                teacher_net=teacher_net,
                student_net=student_net,
                optimizer=optimizer,
                X=X, y=y,
                alpha=a,
                beta=b,
                temperature=temperature
            )
            
            train_loss_list.append(loss)
            train_acc_list.append(acc)
            
            if print_every > 0 and i % print_every == print_every - 1:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)}] '
                      f'Loss: {loss:.3f}, Accuracy: {acc:.3f}')
                
        if (epoch + 1) % 25 == 0 or epoch + 1 == num_epochs:
            checkpoint_path = checkpoint_save_path + hparamToString(hparam) +  f'_{alpha}_{beta}' + f'_checkpoint_epoch_{epoch + 1}.tar'
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
        
        # Validation phase
        if val_loader is not None:
            student_net.eval()
            val_loss, val_acc = getLossAccuracyOnDataset(student_net, val_loader, fast_device)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
            print(f'Epoch {epoch + 1} Validation Accuracy: {val_acc:.3f} Validation Loss: {val_loss:.3f}')
        
        # Update learning rate
        lr_scheduler.step()
        
        # Apply quantization steps if enabled
        if quant:
            if epoch > 8:
                student_net.apply(torch.ao.quantization.disable_observer)
            if epoch > 6:
                student_net.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    
    return {
        'train_loss': train_loss_list,
        'train_acc': train_acc_list,
        'val_acc': val_acc_list
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

def trainStudentOnHparamAT(teacher_net, student_net, hparam, num_epochs, train_loader, val_loader,
                           print_every=0, fast_device=torch.device('cuda:0'), quant=False, 
                           checkpoint_save_path='checkpoints_student_AT/', resume_checkpoint=False, 
                           optimizer_choice='adam'):
    """
    Train the student network using attention-based knowledge distillation.

    Args:
        teacher_net (torch.nn.Module): Teacher model.
        student_net (torch.nn.Module): Student model.
        hparam (dict): Hyperparameters for training.
        num_epochs (int): Number of epochs.
        train_loader (torch.utils.data.DataLoader): Dataloader for training.
        val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing.
        print_every (int): Print progress every specified number of batches.
        fast_device (torch.device): Device to run the models on.
        quant (bool): Whether to apply quantization-aware training.
        checkpoint_save_path (str): Path to save training checkpoints.
        resume_checkpoint (str/bool): Path to resume training from a checkpoint, or False to start fresh.
        optimizer_choice (str): Choice of optimizer ('adam' or 'sgd').

    Returns:
        dict: Training and validation metrics (loss and accuracy).
    """
    teacher_net.to(fast_device)
    student_net.to(fast_device)
    teacher_net.eval()

    if optimizer_choice == 'adam':
        optimizer = optim.Adam(student_net.parameters(), lr=hparam['lr'], weight_decay=hparam['weight_decay'])
    else:
        optimizer = optim.SGD(student_net.parameters(), lr=hparam['lr'], momentum=hparam['momentum'], weight_decay=hparam['weight_decay'])

    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint)
        student_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    at_loss_fn = ATLoss()  # Assuming ATLoss is defined and imported
    criterion = nn.CrossEntropyLoss()

    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []

    for epoch in range(start_epoch, num_epochs):
        student_net.train()
        total_loss, total_correct, total_images = 0, 0, 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(fast_device), labels.to(fast_device)
            optimizer.zero_grad()

            student_outputs = student_net(inputs)
            with torch.no_grad():
                teacher_outputs = teacher_net(inputs)

            distillation_loss = at_loss_fn(teacher_outputs, student_outputs)
            classification_loss = criterion(student_outputs, labels)
            loss = distillation_loss + classification_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(student_outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_images += labels.size(0)

            if print_every > 0 and i % print_every == print_every - 1:
                print(f'Epoch {epoch + 1}, Batch {i + 1}: Loss = {loss.item():.4f}')

        train_loss_list.append(total_loss / total_images)
        train_acc_list.append(total_correct / total_images)

        # Validation phase
        student_net.eval()
        val_loss, val_acc = getLossAccuracyOnDataset(student_net, val_loader, fast_device, criterion)
        student_net.train()
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        print(f'Epoch {epoch + 1}: Validation Loss = {val_loss:.4f}, Validation Accuracy = {val_acc:.2f}%')

        lr_scheduler.step()

        # Save checkpoints
        if (epoch + 1) % 25 == 0 or epoch + 1 == num_epochs:
            checkpoint_path = f"{checkpoint_save_path}checkpoint_epoch_{epoch + 1}.pth"
            torch.save({
                'model_state_dict': student_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss_list,
                'train_acc': train_acc_list,
                'val_loss': val_loss_list,
                'val_acc': val_acc_list
            }, checkpoint_path)
            print(f"Checkpoint saved at: {checkpoint_path}")

        # Apply quantization steps if enabled
        if quant:
            if epoch > 8:
                student_net.apply(torch.ao.quantization.disable_observer)
            if epoch > 6:
                student_net.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    return {
        'train_loss': train_loss_list,
        'train_acc': train_acc_list,
        'val_loss': val_loss_list,
        'val_acc': val_acc_list
    }

class ATLoss(nn.Module):
    """
    Module for calculating AT Loss

    :param norm_type (int): Norm to be used in calculating loss
    """

    def __init__(self, norm_type=2):
        super(ATLoss, self).__init__()
        self.p = norm_type

    def forward(self, teacher_output, student_output):
        """
        Forward function

        :param teacher_output (torch.FloatTensor): Prediction made by the teacher model
        :param student_output (torch.FloatTensor): Prediction made by the student model
        """

        A_t = teacher_output  # [1:]
        A_s = student_output  # [1:]

        loss = 0.0
        for (layerT, layerS) in zip(A_t, A_s):

            xT = self.single_at_loss(layerT)
            xS = self.single_at_loss(layerS)
            loss += (xS - xT).pow(self.p).mean()

        return loss

    def single_at_loss(self, activation):
        """
        Function for calculating single attention loss
        """
        return F.normalize(activation.pow(self.p).view(activation.size(0), -1))


def trainStudentsDML(student_models, hparam, num_epochs, train_loader, val_loader, 
                     optimizers, device=torch.device('cuda:0'), checkpoint_save_path='checkpoints_dml/'):
    """
    Train a cohort of student networks using Deep Mutual Learning (DML).

    Args:
        student_models (list of torch.nn.Module): List of student models.
        hparam (dict): Hyperparameters for training.
        num_epochs (int): Number of epochs.
        train_loader (torch.utils.data.DataLoader): DataLoader for training.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation/testing.
        optimizers (list of torch.optim.Optimizer): List of optimizers for each student model.
        device (torch.device): Device to run the models on.
        checkpoint_save_path (str): Path to save training checkpoints.
    """
    for model in student_models:
        model.to(device).train()

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = [0] * len(student_models)
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)

            # Zero the parameter gradients for all optimizers
            for optimizer in optimizers:
                optimizer.zero_grad()

            # Forward pass for all students
            outputs = [model(data) for model in student_models]

            # Calculate loss for each student
            losses = []
            for idx, (student_output, optimizer) in enumerate(zip(outputs, optimizers)):
                # Cross-entropy loss for correct labels
                loss = criterion(student_output, labels)
                
                # DML Loss: each student learns from each other
                for other_output in outputs:
                    if student_output is not other_output:
                        loss += F.mse_loss(F.softmax(student_output, dim=1), F.softmax(other_output, dim=1))
                
                losses.append(loss)
                total_loss[idx] += loss.item()
                loss.backward()
                optimizer.step()

        if epoch % 1 == 0:  # Logging interval
            print(f'Epoch {epoch + 1}, Losses: {[l.item() for l in losses]}')

        # Validation logic here if required
        # Save models at checkpoints if necessary

    # Save final models
    for idx, model in enumerate(student_models):
        torch.save(model.state_dict(), f"{checkpoint_save_path}student_{idx}_final.pth")

    print("Training completed.")

def reproducibilitySeed(use_gpu):
    """
    Ensure reproducibility of results; Seeds to 0
    """
    torch_init_seed = 0
    torch.manual_seed(torch_init_seed)
    numpy_init_seed = 0
    np.random.seed(numpy_init_seed)
    if use_gpu:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False