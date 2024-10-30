import torch
import torch.nn as nn

from math import log


class TCL(nn.Module):
    def __init__(self, temperature=0.1, k1=5000.0, k2=1.0):

        super(TCL, self).__init__()
        self.temperature = temperature
        self.k1 = torch.tensor(k1,requires_grad=False)
        self.k2 = torch.tensor(k2,requires_grad=False)

    def forward(self, projections, targets):

        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")

        dot_product_tempered = torch.mm(projections, projections.T)
        exp_dot_tempered = torch.exp((dot_product_tempered) / self.temperature)
        exp_dot_tempered_n = torch.exp(-1 * dot_product_tempered) 
       

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_positives = mask_similar_class * mask_anchor_out
        mask_negatives = ~mask_similar_class
        positives_per_samples = torch.sum(mask_positives, dim=1)
        negatives_per_samples = torch.sum(mask_negatives, dim=1)
        
        tcl_loss = torch.sum(-torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_positives, dim=1)+(self.k1*torch.sum(exp_dot_tempered_n * mask_positives, dim=1))+(self.k2*torch.sum(exp_dot_tempered * mask_negatives, dim=1)))) * mask_positives,dim=1) / positives_per_samples
        
        tcl_loss_mean = torch.mean(tcl_loss)
        return tcl_loss_mean
