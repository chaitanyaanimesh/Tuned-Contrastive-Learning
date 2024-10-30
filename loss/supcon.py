import torch
import torch.nn as nn

from math import log


class SUPCON(nn.Module):
    def __init__(self, temperature=0.1):

        super(SUPCON, self).__init__()
        self.temperature = temperature

    def forward(self, projections, targets):

        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")

        dot_product_tempered = torch.mm(projections, projections.T)
        exp_dot_tempered = torch.exp((dot_product_tempered) / self.temperature) 
       

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_positives = mask_similar_class * mask_anchor_out
        mask_negatives = ~mask_similar_class
        positives_per_samples = torch.sum(mask_positives, dim=1)
        negatives_per_samples = torch.sum(mask_negatives, dim=1)
        
        supcon_loss = torch.sum(-torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_positives, dim=1)+(torch.sum(exp_dot_tempered * mask_negatives, dim=1)))) * mask_positives,dim=1) / positives_per_samples
        
        supcon_loss_mean = torch.mean(supcon_loss)
        return supcon_loss_mean
