import torch
import torch.nn.functional as F
from torch import nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2):
        # Euclidean distance between masked and unmasked embeddings
        euclidean_distance = F.pairwise_distance(output1, output2)
        # Minimize the distance to align embeddings
        loss = torch.mean(euclidean_distance ** 2)
        return loss
        
