import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models.loss import FairLoss

# Simple Neural Network with two weights
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.w1 = nn.Parameter(torch.randn(1))
        self.w2 = nn.Parameter(torch.randn(1))

    def forward(self, x):
        z = self.w1 * x[0] + self.w2 * x[1]
        # return torch.tensor([z, -z])
        return torch.softmax(torch.tensor([z, -z]), dim=0)  # Using softmax for two-class output

# Custom FairLoss (for illustration, define your own logic here)
# Generate a grid of weights
w1_values = np.linspace(-5, 5, 100)
w2_values = np.linspace(-5, 5, 100)
W1, W2 = np.meshgrid(w1_values, w2_values)

# Ground truth labels for loss calculation
y_true = torch.tensor([1, 0], dtype=torch.float32)  # Example true label for CrossEntropy (class 0)

# Instantiate loss functions
cross_entropy_loss = nn.CrossEntropyLoss()
fair_loss = FairLoss(num_classes=2)

# Initialize the neural network
model = SimpleNN()

# Compute the losses
loss_ce = np.zeros(W1.shape)
loss_fair = np.zeros(W1.shape)

for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        model.w1.data = torch.tensor(W1[i, j], dtype=torch.float32)
        model.w2.data = torch.tensor(W2[i, j], dtype=torch.float32)

        output = model(torch.tensor([1.0, 1.0]))  # Example input

        # CrossEntropyLoss expects class indices
        predicted_class = torch.argmax(output).unsqueeze(0)  # Get class index
        loss_ce[i, j] = cross_entropy_loss(output.unsqueeze(0), torch.tensor([1])).item()
        loss_fair[i, j] = fair_loss(output.unsqueeze(0), torch.tensor([1])).item()

# Plotting the losses in 3D
fig = plt.figure(figsize=(12, 6))

# Plot CrossEntropyLoss
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(W1, W2, loss_ce, cmap='viridis', alpha=0.7)
ax1.set_title('Cross Entropy Loss')
ax1.set_xlabel('Weight 1')
ax1.set_ylabel('Weight 2')
ax1.set_zlabel('Loss')

# Plot FairLoss
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(W1, W2, loss_fair, cmap='plasma', alpha=0.7)
ax2.set_title('Fair Loss')
ax2.set_xlabel('Weight 1')
ax2.set_ylabel('Weight 2')
ax2.set_zlabel('Loss')

plt.tight_layout()
plt.show()
