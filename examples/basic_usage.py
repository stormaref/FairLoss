"""Minimal example: train a linear classifier with FairLoss."""

import torch
import torch.nn as nn
import torch.optim as optim

from fairloss import FairLoss


def main():
    model = nn.Linear(784, 10)
    criterion = FairLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for step in range(3):
        x = torch.randn(64, 784)
        y = torch.randint(0, 10, (64,))

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"step={step + 1} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
