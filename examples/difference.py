import torch
import torch.nn as nn
from fairloss import FairLoss

output = torch.tensor(
    [[0.34, 0.33, 0.33], [0.49, 0.51, 0.0]], dtype=torch.float32)
target = torch.tensor([0, 0], dtype=torch.long)

cross_entropy_fn = nn.CrossEntropyLoss(reduction="none")
fair_loss_fn = FairLoss(reduction="none")

cross_entropy = cross_entropy_fn(output, target)
fair_loss = fair_loss_fn(output, target)

print(f"Cross-entropy:  {cross_entropy}")
print(f"FairLoss:       {fair_loss}")
