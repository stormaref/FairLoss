from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FairLoss(nn.Module):
    """Classification loss combining cross-entropy with a SmoothMax-based penalty.

    Args:
        cross_entropy: Cross-entropy loss with ``reduction="none"``. Defaults to
            ``nn.CrossEntropyLoss(reduction="none")``.
        smooth_max_beta: Temperature for the smooth maximum approximation.
        scaling_factor: Scale applied before the sigmoid penalty transform.
        reduction: Reduction applied to the combined loss.

    Forward:
        output: Logits of shape (N, C).
        target: Integer class labels of shape (N,).
        Returns: Scalar mean loss over the batch.
    """

    def __init__(
        self,
        cross_entropy: Optional[nn.CrossEntropyLoss] = None,
        smooth_max_beta: float = 1000.0,
        scaling_factor: float = 1000.0,
        reduction: Literal["mean", "none", "sum"] = "mean",
    ):
        super().__init__()
        if cross_entropy is None:
            cross_entropy = nn.CrossEntropyLoss(reduction="none")

        if not isinstance(cross_entropy, nn.CrossEntropyLoss):
            raise TypeError(
                "cross_entropy must be an instance of nn.CrossEntropyLoss, "
                f"got {type(cross_entropy).__name__}."
            )
        if cross_entropy.reduction != "none":
            raise ValueError(
                'cross_entropy must use reduction="none" so FairLoss can add the '
                f"penalty per sample before applying its own reduction, got "
                f'reduction="{cross_entropy.reduction}".'
            )

        self.cross_entropy = cross_entropy
        self.smooth_max_beta = smooth_max_beta
        self.scaling_factor = scaling_factor
        self.reduction = reduction

    def one_hot_encode(self, target: torch.Tensor, num_classes: int) -> torch.Tensor:
        target = target.long()
        return F.one_hot(target, num_classes=num_classes).float()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = output.shape[1]
        base_loss = self.cross_entropy(output, target)

        target = self.one_hot_encode(target, num_classes)
        smooth_max = (1.0 / self.smooth_max_beta) * torch.logsumexp(
            self.smooth_max_beta * output, dim=1, keepdim=False
        )

        result = output - smooth_max.unsqueeze(1)
        element_wise_mult = result * target

        ones_vector = torch.ones(num_classes, 1, device=output.device)
        final_scalar = torch.matmul(element_wise_mult, ones_vector)

        sigmoid_output = torch.sigmoid(self.scaling_factor * -1 * final_scalar)
        final_output = (sigmoid_output - 0.5) * 2
        final_output = final_output.squeeze(1)

        penalty = final_output / num_classes
        total_loss = base_loss + penalty

        if self.reduction == "mean":
            return total_loss.mean()
        if self.reduction == "sum":
            return total_loss.sum()

        return total_loss
