# === Added for exp006 improvement ===
import torch
import torch.nn as nn
import torchvision.models as models
from src.metric_learning import ArcFaceHead

class AtmaCupModel(nn.Module):
    """
    Refactored model structure for Exp006 Phase 2.
    Separates Backbone and Head for easier extension (ArcFace, Embedding Head etc.)
    """
    def __init__(self, num_classes=11, pretrained=True, freeze_backbone=True, 
                 use_arcface=False, use_embedding_head=False):
        super().__init__()
        self.use_arcface = use_arcface
        self.use_embedding_head = use_embedding_head
        
        # 1. Backbone
        # To separate fully, we remove the original fc layer after loading
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone_raw = models.resnet18(weights=weights)
        except:
            backbone_raw = models.resnet18(pretrained=pretrained)
        
        if freeze_backbone:
            for param in backbone_raw.parameters():
                param.requires_grad = False
        
        # Remove the final fc layer from backbone
        layers = list(backbone_raw.children())[:-1] # Remove fc, keep pool
        self.backbone = nn.Sequential(*layers)
        
        self.in_features = backbone_raw.fc.in_features
        
        # 2. Embedding Head (Optional - Phase 2 Step 3)
        self.embedding_head = None
        current_in_features = self.in_features
        
        if self.use_embedding_head:
            # BN -> FC -> BN structure
            # Let's say we project to 512 dim
            emb_dim = 512 
            self.embedding_head = nn.Sequential(
                nn.BatchNorm1d(current_in_features),
                nn.Linear(current_in_features, emb_dim, bias=False),
                nn.BatchNorm1d(emb_dim)
            )
            current_in_features = emb_dim
            
        # 3. Classifier Head
        # Phase 2 Step 2: ArcFace Option will go here
        if self.use_arcface:
            self.classifier = ArcFaceHead(current_in_features, num_classes)
        else:
            # Standard Linear Head
            self.classifier = nn.Linear(current_in_features, num_classes)

    def forward(self, x, targets=None):
        # Backbone: [B, C, H, W] -> [B, 512, 1, 1]
        features = self.backbone(x)
        # Flatten: [B, 512]
        features = torch.flatten(features, 1)
        
        # Embedding Head
        if self.embedding_head is not None:
            features = self.embedding_head(features)
        
        # Classifier
        if self.use_arcface:
             # ArcFace usually needs targets/labels during training to calculate angular margin
             # In inference, targets is None
             output = self.classifier(features, targets)
        else:
             output = self.classifier(features)
             
        return output
