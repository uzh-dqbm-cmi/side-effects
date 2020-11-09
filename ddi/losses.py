import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    """

    def __init__(self, margin, reduction='mean', eps=1e-8):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.eps = eps

    def forward(self, dist, target):
        """
        Args:
            dist: tensor, (batch, ), computed distance between two inputs
            target: tensor, (batch, ), labels (0/1) 
        """
        margin = self.margin
        reduction = self.reduction
        repel = (1-target) * (0.5 * torch.pow(torch.clamp(margin - dist, min=0.0), 2))
        attract = target * 0.5 * torch.pow(dist, 2) 

        if reduction == 'mean':
            loss_contrastive = torch.mean(repel + attract)
        elif reduction == 'sum':
            loss_contrastive = torch.sum(repel + attract)
        elif reduction == 'none':
            loss_contrastive = repel + attract
        return loss_contrastive


class CosEmbLoss(nn.Module):
    """
    Cosine Embedding loss
    """

    def __init__(self, margin, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, sim, target):
        """
        Args:
            sim: tensor, (batch, ), computed similarity between two inputs
            target: tensor, (batch, ), labels (0/1) 
        """
        margin = self.margin
        reduction = self.reduction
        repel = (1-target) * (torch.clamp(sim - margin, min=0.0))
        attract = target * (1-sim)
        
        if reduction == 'mean':
            loss_contrastive = torch.mean(repel + attract)
        elif reduction == 'sum':
            loss_contrastive = torch.sum(repel + attract)
        elif reduction == 'none':
            loss_contrastive = repel + attract
        return loss_contrastive