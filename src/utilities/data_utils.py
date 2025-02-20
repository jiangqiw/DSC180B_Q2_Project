import torch
import torchvision
import torchvision.transforms as transforms
import os

def load_data_CIFAR10(batch_size=128):
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cifar10_path = os.path.abspath(os.path.join(current_dir, '../../CIFAR10_dataset/'))

    # Set up transformations for CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-10 dataset
    train_val_dataset = torchvision.datasets.CIFAR10(root=cifar10_path, train=True,
                                                     download=True, transform=transform_train)

    test_dataset = torchvision.datasets.CIFAR10(root=cifar10_path, train=False,
                                                download=True, transform=transform_test)

    # Split the training dataset into training and validation
    num_train = int(0.95 * len(train_val_dataset))
    num_val = len(train_val_dataset) - num_train
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [num_train, num_val])

    # DataLoader setup
    train_val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_val_loader, train_loader, val_loader, test_loader


def load_data_CIFAR100(batch_size=128):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cifar100_path = os.path.abspath(os.path.join(current_dir, '../../CIFAR100_dataset/'))

    # Set up transformations for CIFAR-100
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # Load CIFAR-100 dataset
    train_val_dataset = torchvision.datasets.CIFAR100(root=cifar100_path, train=True,
                                                       download=True, transform=transform_train)

    test_dataset = torchvision.datasets.CIFAR100(root=cifar100_path, train=False,
                                                  download=True, transform=transform_test)

    # Split the training dataset into training and validation
    num_train = int(0.95 * len(train_val_dataset))
    num_val = len(train_val_dataset) - num_train
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [num_train, num_val])

    # DataLoader setup
    train_val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_val_loader, train_loader, val_loader, test_loader
