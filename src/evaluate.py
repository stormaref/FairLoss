import torch
import torch.nn.functional as F
from tqdm import tqdm

def evaluate(model, data_loader, criterion, device='cpu'):
    """
    Evaluate the model on the given data loader.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to run evaluation on
    
    Returns:
        tuple: (average_loss, accuracy, predictions)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Calculate loss
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Store predictions and targets for additional metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    average_loss = total_loss / len(data_loader.dataset)
    accuracy = correct / total
    
    return average_loss, accuracy, (all_predictions, all_targets)