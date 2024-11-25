import torch
from models.handler import Handler
from models.model import ResnetClassifier
from models.loss import FairLoss
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision import transforms
from torchvision.datasets import FashionMNIST, CIFAR10, CIFAR100, MNIST
from models.noise import InstanceDependentNoiseAdder

class Repeater:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = torch.device(device)
        self.train_dataset = CIFAR10(root='data', train=True, download=False)
        self.test_dataset = CIFAR10(root='data', train=False, download=False)
        self.transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomGrayscale(p=0.5),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        

    def save_weights(self):
        model = ResnetClassifier()
        model = model.to(self.device)
        torch.save(model.state_dict(), 'model.pth')
        
    def repeat(self, repeat_count: int, epochs: int):
        total_ce_acc = 0
        total_fl_acc = 0
        for i in range(repeat_count):
            self.save_weights()
            
            model = ResnetClassifier()
            model = model.to(self.device)
            model.load_state_dict(torch.load('model.pth'))
            critertion = FairLoss(num_classes=10)
            optimizer = Adam(model.parameters(), lr=0.001)
            handler = Handler(model, self.device, critertion, optimizer, self.train_dataset, 
                              self.test_dataset, self.transform)
            handler.train(epochs)
            fl_acc = handler.test()
            total_fl_acc += fl_acc
            print(f'FairLoss accuracy in repeat {i + 1}: {fl_acc * 100}%')
            
            model = ResnetClassifier()
            model = model.to(self.device)
            model.load_state_dict(torch.load('model.pth', weights_only=True))
            critertion = CrossEntropyLoss()
            optimizer = Adam(model.parameters(), lr=0.001)
            handler = Handler(model, self.device, critertion, optimizer, self.train_dataset, 
                              self.test_dataset, self.transform)
            handler.train(epochs)
            ce_acc = handler.test()
            total_ce_acc += ce_acc
            print(f'CrossEntropyLoss accuracy in repeat {i + 1}: {ce_acc * 100}%')
            
        avg_ce_acc = total_ce_acc / repeat_count
        avg_fl_acc = total_fl_acc / repeat_count
        print(f'Average CE: {avg_ce_acc * 100}%')
        print(f'Average FL: {avg_fl_acc * 100}%')