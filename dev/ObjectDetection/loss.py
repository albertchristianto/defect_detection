''' Adapted from https://github.com/ZeroE04/R-CenterNet/
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()  
  neg_inds = gt.lt(1).float()
  neg_weights = torch.pow(1 - gt, 4) 
  pred = torch.clamp(pred.sigmoid_(), min=1e-4, max=1-1e-4)
  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
  num_pos  = pos_inds.float().sum() 
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()
  if num_pos == 0:
    loss = -neg_loss
  else:
    loss = -(pos_loss + neg_loss) / num_pos 
  return loss

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, pred_tensor, target_tensor):
    return self.neg_loss(pred_tensor, target_tensor)

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, pred, target, mask):
    loss = F.smooth_l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss
