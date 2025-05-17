import torch
import torch.nn.functional as F

def evaluate(model, dataloader, criterion, device='cpu', mc_dropout=False, mc_passes=10):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0

    def enable_dropout(m):
        if isinstance(m, torch.nn.Dropout):
            m.train()

    if mc_dropout:
        model.apply(enable_dropout)

    all_outputs = []
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            if mc_dropout:
                outputs = torch.stack([model(data) for _ in range(mc_passes)])
                probs = outputs.softmax(dim=-1).mean(dim=0)
                all_outputs.append(probs)
                loss = criterion(outputs.mean(dim=0), target)
            else:
                output = model(data)
                probs = F.softmax(output, dim=-1)
                all_outputs.append(probs)
                loss = criterion(output, target)

            loss_total += loss.item() * data.size(0)
            pred = probs.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)

    avg_loss = loss_total / total
    accuracy = correct / total
    return avg_loss, accuracy, torch.cat(all_outputs, dim=0)