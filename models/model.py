from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

class ResnetClassifier(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(ResnetClassifier, self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone.fc = nn.Sequential(nn.Flatten(), nn.Linear(self.backbone.fc.in_features, num_classes))

    def forward(self, x):
        return self.backbone(x)