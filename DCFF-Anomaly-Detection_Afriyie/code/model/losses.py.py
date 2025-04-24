import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs.squeeze(), targets)
        pt = torch.exp(-bce_loss)
        focal_loss = (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()