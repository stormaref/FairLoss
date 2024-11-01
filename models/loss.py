import torch
import torch.nn as nn

class FairLoss(nn.Module):
    def __init__(self, num_classes, smooth_max_beta=1000.0, scaling_factor=100):
        super(FairLoss, self).__init__()
        self.base_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.num_classes = num_classes
        self.smooth_max_beta = smooth_max_beta
        self.scaling_factor = scaling_factor

    def forward(self, output, target):
        smooth_max = (1.0 / self.smooth_max_beta) * torch.logsumexp(self.smooth_max_beta * output, dim=1, keepdim=False)
        result = output - smooth_max.unsqueeze(1)
        elementwise_mult = result * target  # Shape: (N, C)
        ones_vector = torch.ones(self.num_classes, 1)  # Shape: (C, 1)
        final_scalar = torch.matmul(elementwise_mult, ones_vector)  # Shape: (N, 1)
        sigmoid_output = torch.sigmoid(self.scaling_factor * -1 * final_scalar)  # Shape: (N, 1)
        final_output = (sigmoid_output - 0.5) * 2  # Shape: (N, 1)
        base_loss = self.base_loss_fn(output, target)  # Keep base_loss shape as (N,)
        final_output = final_output.squeeze(1)  # Shape: (N,)
        total_loss = base_loss + (1 / self.num_classes) * final_output  # Keep final_output as is
        return total_loss.mean()  # Shape: (N,)