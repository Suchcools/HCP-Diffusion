import torch
from torch import nn
from hcpdiff.loss import EDMLoss, SSIMLoss, GWLoss  # Replace 'your_module_name' with the actual module name where MinSNRLoss is defined

class CombinedLoss(nn.Module):
    def __init__(self, edm_weight=1.0, ssim_weight=0.005, gw_weight=0.001, gamma=1.0,**kwargs):
        super(CombinedLoss, self).__init__()

        self.edm_loss = EDMLoss(gamma=gamma, **kwargs)
        
        self.ssim_loss = SSIMLoss( **kwargs)
        self.gw_loss = GWLoss(**kwargs)

        self.edm_weight = edm_weight
        self.ssim_weight = ssim_weight
        self.gw_weight = gw_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor: # input: latent target: latent sigma [b,64,64,64]
        edm_loss = self.edm_loss(input, target, sigma)
        # ssim_loss = self.ssim_loss(input, target)
        gw_loss = self.gw_loss(input, target)

        # Combine losses with weights
        combined_loss = (
            self.edm_weight * edm_loss +  # [20, 64, 64, 64]
            self.gw_weight * gw_loss
            # self.ssim_weight * ssim_loss
        )

        return combined_loss

import torch
combined_loss = CombinedLoss()
loss = combined_loss(torch.Tensor(2,64,64,64), torch.Tensor(2,64,64,64), torch.Tensor(2,1,1,1))
print(loss.shape)
