import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from models.loss import FairLoss

def generate_logits1(x, y):
    # Example function that combines sine and cosine transformations
    return torch.sin(10*(torch.pow(x,2)+torch.pow(y,2)))/10  # Scale for visibility

def generate_logits1(x, y):
    # Example function that combines sine and cosine transformations
    return torch.max(x) + torch.max(y)  # Scale for visibility

def generate_logits6(x, y):
    # Example function that combines sine and cosine transformations
    return 1/(15*(torch.pow(x,2)+torch.pow(y,2)))  # Scale for visibility

def generate_logits6(x, y):
    # Example function that combines sine and cosine transformations
    return 1-torch.abs(x+y)-torch.abs(y-x)  # Scale for visibility

def generate_logits2(x, y):
    # Example function that combines sine and cosine transformations
    return (x**2 + y**2).sqrt()  # Scale for visibility

def generate_logits1(x, y):
    return torch.sin(5*x)*torch.cos(5*y)/5

def generate_logits1(x, y):
    return x*torch.cos(y)

def generate_logits1(x, y):
    return 10*x*torch.sigmoid((10*x+2*y) * torch.sigmoid(5*x-7*y)) + 2*y

def generate_logits(x, y):
    return torch.max(y) + torch.max(x)

def generate_logits(x, y):
    return torch.max(x, y) * torch.sigmoid(x) * torch.sigmoid(y)

# Initialize FairLoss and CrossEntropy loss objects
fair_loss = FairLoss(num_classes=2)
cross_entropy_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]))

# Generate parameter ranges for logit values
param1_range = torch.linspace(-5, 5, 250)
param2_range = torch.linspace(-5, 5, 250)

# Create a meshgrid for parameters
param1_grid, param2_grid = torch.meshgrid(param1_range, param2_range, indexing='xy')

# Prepare to store loss values
fair_values = np.zeros((250, 250))
cross_entropy_values = np.zeros((250, 250))
logit_values = np.zeros((250, 250, 2))

# Compute the loss values for each point on the grid
for i in range(250):
    for j in range(250):
        logit = generate_logits(param1_grid[i, j], param2_grid[i, j])
        logit_values[i, j] = [logit.item()]

        # Fair Loss using generated logits
        fair_logits = torch.softmax(torch.tensor([[logit.item(), 0]], dtype=torch.float32), dim=1)# Shape [1, 2]
        fair_target = torch.tensor([1], dtype=torch.long)  # Target class for FairLoss
        
        # Cross-Entropy Loss using generated logits
        logits = torch.softmax(torch.tensor([[logit.item(), 0]], dtype=torch.float32), dim=1)# Shape [1, 2]
        cross_entropy_target = torch.tensor([1], dtype=torch.long)  # Target class for Cross-Entropy loss
        cross_entropy_values[i, j] = cross_entropy_loss(logits, cross_entropy_target).item()
        
        fair_values[i, j] = fair_loss(fair_logits, fair_target).item() - cross_entropy_loss(logits, cross_entropy_target).item()

# Plotting
fig = plt.figure(figsize=(18, 6))

# Plot Fair Loss
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(param1_grid.numpy(), param2_grid.numpy(), fair_values, cmap='viridis')
ax1.set_title('Fair Loss')
ax1.set_xlabel('Parameter 1')
ax1.set_ylabel('Parameter 2')
ax1.set_zlabel('Loss Value')

# Plot Cross-Entropy Loss
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(param1_grid.numpy(), param2_grid.numpy(), cross_entropy_values, cmap='plasma')
ax2.set_title('Cross-Entropy Loss')
ax2.set_xlabel('Logit 1')
ax2.set_ylabel('Logit 2')
ax2.set_zlabel('Loss Value')

# Plot Original Logits
ax3 = fig.add_subplot(133, projection='3d')
logit_surface = logit_values[:, :, 0]
logit2_surface = logit_values[:, :, 1]
ax3.plot_surface(param1_grid.numpy(), param2_grid.numpy(), logit_surface, cmap='coolwarm', alpha=0.6)
ax3.plot_surface(param1_grid.numpy(), param2_grid.numpy(), logit2_surface, cmap='coolwarm', alpha=0.6)
ax3.set_title('Generated Logits')
ax3.set_xlabel('Logit 1')
ax3.set_ylabel('Logit 2')
ax3.set_zlabel('Logit Value')

plt.show()