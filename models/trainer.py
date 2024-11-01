import torch
import torch.nn as nn
from models.model import ResnetClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer:
    def __init__(
        self, 
        model: ResnetClassifier, 
        device,
        train_dataloader, 
        val_dataloader, 
        criterion: nn.Module, 
        optimizer: torch.optim.Optimizer
    ):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.train_step(epoch, num_epochs)
            val_loss, val_acc = self.val_step()
            print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss}, Val Acc: {val_acc}')
        self.plot()
            
    def train_step(self, epoch, num_epochs):
        self.model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0
        progress_bar = tqdm(self.train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in progress_bar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_samples += labels.size(0)
            correct_preds += predicted.eq(labels).sum().item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(self.train_dataloader)
        accuracy = correct_preds / total_samples
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        progress_bar.set_postfix(loss=avg_loss, accuracy=accuracy)

    def val_step(self):
        self.model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in self.val_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_samples += labels.size(0)
                correct_preds += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(self.val_dataloader)
        val_accuracy = correct_preds / total_samples
        self.val_losses.append(avg_val_loss)
        self.val_accuracies.append(val_accuracy)
        return val_loss, val_accuracy

    def plot(self):
        epochs = range(1, self.num_epochs + 1)
        plt.figure(figsize=(12, 5))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Training Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label='Training Accuracy')
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()
        