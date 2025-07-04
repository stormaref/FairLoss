import torch
import torch.nn as nn
import torch.nn.functional as F

class FairLoss(nn.Module):
    def __init__(self, smooth_max_beta=1000.0, scaling_factor=1000):
        super(FairLoss, self).__init__()
        self.base_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.smooth_max_beta = smooth_max_beta
        self.scaling_factor = scaling_factor  # Default is positive

    def one_hot_encode(self, target, num_classes):
        # Ensure the target is of type long (int64)
        target = target.long()
        # Use F.one_hot to create the one-hot encoded tensor
        one_hot = F.one_hot(target, num_classes=num_classes)  # Shape: (N, num_classes)
        return one_hot.float()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        num_classes = output.shape[1]
        # Calculate base loss for each sample, maintaining output shape
        base_loss = self.base_loss_fn(output, target)  # Keep base_loss shape as (N,)
        # print(f'Base loss: {base_loss}, shape: {base_loss.shape}')  # Debugging output and shape

        target = self.one_hot_encode(target, num_classes)
        # print(f'Output: {output}')  # Debugging output
        # print(f'Output shape: {output.shape}, target shape: {target.shape}')  # Debugging output
        # Calculate SmoothMax using logsumexp
        # Calculate SmoothMax using logsumexp for each item in the batch
        smooth_max = (1.0 / self.smooth_max_beta) * torch.logsumexp(self.smooth_max_beta * output, dim=1, keepdim=False)
        # print(f'SmoothMax: {smooth_max}, shape: {smooth_max.shape}')  # Debugging output and shape

        # Subtract SmoothMax from output
        result = output - smooth_max.unsqueeze(1)
        # print(f'Result after subtraction: {result}, shape: {result.shape}')  # Debugging output and shape

        # Element-wise multiplication with target
        element_wise_mult = result * target  # Shape: (N, C)
        # print(f'Element-wise multiplication: {element_wise_mult}, shape: {element_wise_mult.shape}')  # Debugging output and shape

        # Create a vector of ones with the same shape as the number of classes
        ones_vector = torch.ones(num_classes, 1, device=output.device)  # Shape: (C, 1)
        # print(f'Ones vector: {ones_vector}, shape: {ones_vector.shape}')  # Debugging output and shape

        # Transpose the ones vector and multiply to get a tensor for each sample
        final_scalar = torch.matmul(element_wise_mult, ones_vector)  # Shape: (N, 1)
        # print(f'Final scalar before squeezing: {final_scalar}, shape: {final_scalar.shape}')  # Debugging output and shape

        # Apply the transformation: sigmoid(scaling_factor * -1 * final_scalar)
        sigmoid_output = torch.sigmoid(self.scaling_factor * -1 * final_scalar)  # Shape: (N, 1)
        # print(f'Sigmoid output: {sigmoid_output}, shape: {sigmoid_output.shape}')  # Debugging output and shape

        # Final transformation: (y - 0.5) * 2
        final_output = (sigmoid_output - 0.5) * 2  # Shape: (N, 1)
        # print(f'Final output after transformation: {final_output}, shape: {final_output.shape}')  # Debugging output and shape

        # Add the scaled final output to base loss without summing
        final_output = final_output.squeeze(1)  # Shape: (N,)
        # print(f'Final output after squeezing: {final_output}, shape: {final_output.shape}')  # Debugging output and shape

        # Total loss will also have shape (N,)
        penalty = final_output / num_classes  # Shape: (N,)
        total_loss = base_loss + penalty  # Keep final_output as is
        # print(f'Total loss: {total_loss}, shape: {total_loss.shape}')  # Debugging output and shape

        return total_loss.mean()  # Shape: (N,)
    
    
def get_loss_fn(name):
    if name.lower() == 'ce':
        return nn.CrossEntropyLoss()
    elif name.lower() == 'fair_loss':
        return FairLoss()
    else:
        raise ValueError(f"Unknown loss: {name}")