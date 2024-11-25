import torch
import torch.nn as nn
from models.model import ResnetClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class Trainer:
    def __init__(
        self, 
        model: ResnetClassifier, 
        device,
        train_dataloader, 
        val_dataloader, 
        criterion: nn.Module, 
        optimizer: torch.optim.Optimizer,
        checkpoint_dir='checkpoints'
    ):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.best_checkpoint_path = None

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            self.train_step(epoch, num_epochs)
            val_loss, val_acc = self.val_step()
            # print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Checkpointing based on validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_checkpoint_path = self.save_checkpoint(epoch)

        # Load the best model after training
        if self.best_checkpoint_path:
            self.load_checkpoint(self.best_checkpoint_path)
            # print(f'Best model loaded from {self.best_checkpoint_path} with validation loss {self.best_val_loss:.4f}')
        
        self.plot(num_epochs)

    def train_step(self, epoch, num_epochs):
        self.model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0
        # progress_bar = tqdm(self.train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in self.train_dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_samples += labels.size(0)
            correct_preds += predicted.eq(labels).sum().item()
            # avg_loss = running_loss / (total_samples / labels.size(0))
            # accuracy = correct_preds / total_samples
            # progress_bar.set_postfix({'avg_loss':avg_loss, 'accuracy':accuracy})

        # Log epoch metrics
        self.train_losses.append(running_loss / len(self.train_dataloader))
        self.train_accuracies.append(correct_preds / total_samples)

    def val_step(self):
        self.model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in self.val_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
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
        return avg_val_loss, val_accuracy

    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'best_model_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.best_val_loss,
        }, checkpoint_path)
        # print(f'Checkpoint saved at {checkpoint_path}')
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # print(f'Loaded checkpoint from {checkpoint_path} at epoch {checkpoint["epoch"]}')

    def plot(self, num_epochs):
        epochs = range(1, num_epochs + 1)
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
        # plt.show()
