import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
import torch
import torch.nn as nn


def Sigmoid_Focal_Loss(pred, target, alpha=0.5, gamma=2):
    """
    pred : [B, 1, H, W]
    target : [B, 1, H, W]
    """
    # 调整target的大小
    if pred.shape[2:] != target.shape[2:]:
        target = F.interpolate(target, size=pred.shape[2:], mode='nearest')
    target = target.squeeze(1)

    # 对预测输出应用sigmoid函数
    p = torch.sigmoid(pred.squeeze(1))
    # 计算二元交叉熵损失
    ce_loss = F.binary_cross_entropy_with_logits(pred.squeeze(1), target, reduction="none")

    # 计算p_t，即sigmoid输出和真实目标的加权和
    p_t = p * target + (1 - p) * (1 - target)

    # 计算focal loss
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        # 计算alpha_t，即正负样本的加权和
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        # 对loss进行加权
        loss = alpha_t * loss

    loss = loss.mean()

    # 返回计算得到的损失值
    return loss



def Sigmoid_IoU_Loss(pred, target):
    """
    pred : [B, 1, H, W]
    target : [B, 1, H, W]
    """
    # 调整target的大小
    if pred.shape[2:] != target.shape[2:]:
        target = F.interpolate(target, size=pred.shape[2:], mode='nearest')

    pred = torch.sigmoid(pred) > 0.5  # 将预测结果转化为0/1二值图像

    target = target.view(-1)  # 将target展平为一维数组
    pred = pred.view(-1)  # 将pred展平为一维数组
    tp = (target * pred).sum().float()  # 计算TP
    fp = ((1 - target.float()) * pred.float()).sum().float()  # 计算FP
    fn = (target.float() * (1 - pred.float())).sum().float()  # 计算FN
    iou = tp / (tp + fp + fn + 1e-8)  # iou

    # 将IoU转换为损失
    loss = 1 - iou
    return loss


class Focal_IoU_LOSS(nn.Module):

    def __init__(self):
        super(Focal_IoU_LOSS, self).__init__()

    def forward(self, pred, target):
        loss1 = Sigmoid_Focal_Loss(pred, target)
        loss2 = Sigmoid_IoU_Loss(pred, target)
        return loss1 * 0.5 + loss2 * 0.5
