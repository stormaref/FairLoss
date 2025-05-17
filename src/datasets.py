import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

def get_labels(name, test=False):
    """
    Returns the labels for the specified dataset.
    If test is True, returns the test labels.
    """
    if test:
        Dataset = getattr(torchvision.datasets, name)
        test_set = Dataset(root='./data', train=False, download=True, transform=None)
        return test_set.targets.numpy()
    else:
        # For training set, we need to load the dataset with the appropriate transform
        train_transform = T.Compose([T.ToTensor()])
        Dataset = getattr(torchvision.datasets, name)
        full_train = Dataset(root='./data', train=True, download=True, transform=train_transform)
        return full_train.targets.numpy()
    
def get_data_loaders(name, batch_size, val_split=0.1, data_dir='./data'):
    """
    Returns train_loader, val_loader, test_loader with appropriate transforms.
    The training set is split into train and validation subsets.
    """
    train_transform, test_transform = _get_transforms(name)
    Dataset = getattr(torchvision.datasets, name)

    # Load full training set once
    full_train = Dataset(root=data_dir, train=True, download=True, transform=train_transform)
    total = len(full_train)
    val_size = int(val_split * total)
    train_size = total - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size])

    # Apply test transform to val set
    val_set.dataset = Dataset(root=data_dir, train=True, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # Test set
    test_set = Dataset(root=data_dir, train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


def _get_transforms(name):
    """
    Returns (train_transform, test_transform) for the given dataset name.
    For MNIST and FashionMNIST, converts to 3-channel grayscale.
    """
    if name in ['MNIST', 'FashionMNIST']:
        common = [T.Grayscale(num_output_channels=3), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(28, padding=4),
            *common
        ])
        test_transform = T.Compose(common)
    elif name == 'CIFAR10':
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return train_transform, test_transform
