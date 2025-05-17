import torchvision.models as models
import torch.nn as nn

def make_model(arch_name, num_classes=10):
    arch = arch_name.lower()
    if arch == 'resnet18':
        model = models.resnet18()
    elif arch == 'resnet34':
        model = models.resnet34()
    elif arch == 'resnet50':
        model = models.resnet50()
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")
    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model