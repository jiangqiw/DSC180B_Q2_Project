import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

def enforce_pruning_masks(model, masks, fast_device=torch.device('cuda:0')):
    """
    Apply the saved masks to enforce pruning after every optimizer step.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.mul_(masks[name].to(fast_device))

def studentTrainStep(teacher_net, student_net, studentLossFn, optimizer, X, y, T, alpha):
    """
    One training step of student network: forward prop + backprop + update parameters
    Return: (loss, accuracy) of current batch
    """
    optimizer.zero_grad()
    teacher_pred = None
    if alpha > 0:
        with torch.no_grad():
            teacher_pred = teacher_net(X)
    student_pred = student_net(X)
    loss = studentLossFn(teacher_pred, student_pred, y, T, alpha)
    loss.backward()
    optimizer.step()

    # Strictly enforce pruning masks
    enforce_pruning_masks(student_net.model, student_net.masks)

    accuracy = float(torch.sum(torch.argmax(student_pred, dim=1) == y).item()) / y.shape[0]
    return loss, accuracy

def trainStudentOnHparam(teacher_net, student_net, hparam, num_epochs, 
                        train_loader, val_loader, 
                        print_every=0, 
                        fast_device=torch.device('cuda:0')):
    """
    Trains teacher on given hyperparameters for given number of epochs; Pass val_loader=None when not required to validate for every epoch
    Return: List of training loss, accuracy for each update calculated only on the batch; List of validation loss, accuracy for each epoch
    """
    train_loss_list, train_acc_list, val_acc_list = [], [], []
    T = hparam['T']
    alpha = hparam['alpha']
    student_net.dropout_input = hparam['dropout_input']
    student_net.dropout_hidden = hparam['dropout_hidden']

    optimizer = optim.SGD(
        student_net.parameters(),
        lr=hparam['lr'],
        momentum=hparam['momentum'],
        weight_decay=hparam['weight_decay']
    )
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
        if epoch == 0 and val_loader is not None:
            _, val_acc = getLossAccuracyOnDataset(student_net, val_loader, fast_device)
            val_acc_list.append(val_acc)
            print(f"epoch: {epoch} validation accuracy: {val_acc:.3f}")
        for i, data in enumerate(train_loader, 0):
            X, y = data
            X, y = X.to(fast_device), y.to(fast_device)
            loss, acc = studentTrainStep(teacher_net, student_net, studentLossFn, optimizer, X, y, T, alpha)
            train_loss_list.append(loss)
            train_acc_list.append(acc)

            if print_every > 0 and i % print_every == print_every - 1:
                print(f"[{epoch + 1}, {i + 1}/{len(train_loader)}] train loss: {loss:.3f} train accuracy: {acc:.3f}")

        if val_loader is not None:
            _, val_acc = getLossAccuracyOnDataset(student_net, val_loader, fast_device)
            val_acc_list.append(val_acc)
            print(f"epoch: {epoch + 1} validation accuracy: {val_acc:.3f}")
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