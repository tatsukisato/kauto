
import torch
import torch.nn as nn
import torchvision.models as models

class SimpleCNN(nn.Module):
    """Simple CNN model for baseline"""
    
    def __init__(self, num_classes=11, pretrained=True, freeze_backbone=True):
        super().__init__()
        
        # Load ResNet18
        # Using pretrained=True for compatibility. 
        # In newer torchvision, use weights=models.ResNet18_Weights.DEFAULT
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        except:
            self.backbone = models.resnet18(pretrained=pretrained)
        
        # Freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Replace FC layer
        # This layer will have requires_grad=True by default
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)
