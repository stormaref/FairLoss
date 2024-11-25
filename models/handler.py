import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from models.trainer import Trainer
from models.tester import Tester

class Handler:
    def __init__(self, model, device, criterion, optimizer, train_dataset, test_dataset, transform):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

        # Apply transform to datasets
        train_dataset.transform = transform
        test_dataset.transform = transform

        # Split train_dataset into training and validation datasets
        targets = train_dataset.targets

        train_indices, val_indices = train_test_split(
            range(len(train_dataset)),
            test_size=0.1,
            stratify=targets,
            random_state=42
        )

        self.train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(train_dataset, val_indices)

        # Prepare data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=512, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=128, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Initialize trainer and tester
        self.trainer = Trainer(self.model, self.device, self.train_loader, self.val_loader, self.criterion, self.optimizer)
        self.tester = Tester(self.model, self.device, self.test_loader)
        
    def train(self, num_epochs):
        self.trainer.train(num_epochs)
        
    def test(self):
        return self.tester.test()