import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.attention(x)

class DCFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Backbone initialization
        backbone = getattr(torchvision.models, config['model']['backbone'])(pretrained=True)
        self.layer1 = nn.Sequential(*list(backbone.children())[:5])  # Layer2
        self.layer2 = backbone.layer2  # Layer3
        self.layer3 = backbone.layer3  # Layer4
        
        # Attention modules
        self.attn1 = AttentionFusion(config['model']['attention_dims'][0])
        self.attn2 = AttentionFusion(config['model']['attention_dims'][1])
        self.attn3 = AttentionFusion(config['model']['attention_dims'][2])
        
        # Classifier
        self.conv1d = nn.Conv1d(sum(config['model']['attention_dims']), 
                               config['model']['classifier_dims'], 
                               kernel_size=3)
        self.classifier = nn.Linear(config['model']['classifier_dims'], 1)
        
    def forward(self, x):
        # Feature extraction
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        
        # Attention fusion
        a1 = self.attn1(f1)
        a2 = self.attn2(f2)
        a3 = self.attn3(f3)
        
        # Upsample and fuse
        f1_up = F.interpolate(f1, scale_factor=4, mode='bilinear')
        f2_up = F.interpolate(f2, scale_factor=2, mode='bilinear')
        
        fused = a1*f1_up + a2*f2_up + a3*f3
        
        # Classification
        x = self.conv1d(fused.flatten(2))
        return torch.sigmoid(self.classifier(x.mean(2)))