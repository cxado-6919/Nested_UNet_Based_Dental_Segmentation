import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
"""
U-net 논문 원문에서는 pixel-wise softmax와 cross-entropy loss를 적용하였지만, 불균형한 데이터에 취약하다는 단점이 있으므로
성능이 더 우수한 CombinedLoss(Binary cross-entropy loss + Dice loss)를 적용함
"""
def dice_loss(pred, target, smooth=1e-6):
    num_classes = pred.shape[1]
    pred_soft = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    intersection = (pred_soft * target_one_hot).sum(dim=(2, 3))
    cardinality = pred_soft.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice_score = (2. * intersection + smooth) / (cardinality + smooth)
    
    loss = 1 - dice_score.mean()
    return loss

class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super(CombinedLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        d_loss = dice_loss(pred, target)
        return self.weight_ce * ce_loss + self.weight_dice * d_loss
    
def iou_score(pred, target, num_classes):
    pred = pred.view(-1)
    target = target.view(-1)
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).float().sum().item()
        union = pred_inds.float().sum().item() + target_inds.float().sum().item() - intersection
        if union == 0:
            continue
        ious.append(intersection / union)
    if len(ious) == 0:
        return 0
    return np.mean(ious)