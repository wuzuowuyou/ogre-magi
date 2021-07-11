from .l1_loss import MaskL1Loss
from .dice_loss import AdaptiveDiceLoss
import torch.nn as nn

class L1DiceLoss(nn.Module):
    '''
    L1Loss on thresh, DiceLoss on thresh_binary and binary.
    '''

    def __init__(self, eps=1e-6, l1_scale=10):
        super(L1DiceLoss, self).__init__()
        self.dice_loss = AdaptiveDiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()
        self.l1_scale = l1_scale

    def forward(self, pred, batch):
        dice_loss, metrics = self.dice_loss(pred, batch)
        l1_loss_k, l1_metric_k = self.l1_loss(pred['thresh_k'], batch['thresh_map_k'], batch['thresh_mask_k'])
        l1_loss_v, l1_metric_v = self.l1_loss(pred['thresh_v'], batch['thresh_map_v'], batch['thresh_mask_v'])

        l1_loss = l1_loss_k + l1_loss_v
        loss = dice_loss + self.l1_scale * l1_loss
        metrics.update(**l1_metric_k)
        metrics.update(**l1_metric_v)

        return loss, metrics