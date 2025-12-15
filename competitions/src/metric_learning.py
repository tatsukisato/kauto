# === Added for exp006 improvement ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceHead(nn.Module):
    """
    ArcMap: ArcFace + Softmax
    References:
        https://arxiv.org/pdf/1801.07698.pdf
        https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # Normalize input and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # If label is None (Inference), just return scaled cosine
        if label is None:
            return cosine * self.s
        
        # Training logic
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Keep numerical stability
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot encoding for target labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Apply margin only to ground truth classes
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output
