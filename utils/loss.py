import torch
from torch.autograd import Function
import torch.nn.functional as F 
import torch.nn as nn 
from itertools import repeat
import numpy as np
from torch.autograd import Variable
from skimage.measure import label, regionprops

def KL_divergence(logvar, mu):
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kld.mean()

def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d().cuda()

    return criterion(pred, label)

class CrossEntropy2d(nn.Module):

    def __init__(self, reduction='mean', ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.reduction = reduction 
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        #print('target.dim(): ', target.dim())
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        #print('target.max(): ', target.max())
        #print('target.min(): ', target.min())
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, reduction=self.reduction)
        return loss

class BCEWithLogitsLoss2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(2), "{0} vs {1} ".format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        #print('target.max(): ', target.max())
        #print('target.min(): ', target.min())
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict[target_mask]
        loss = F.binary_cross_entropy_with_logits(predict, target, weight=weight, reduction='mean')
        return loss

class MaskDiceLoss(nn.Module):
    def __init__(self):
        super(MaskDiceLoss, self).__init__()

    def dice_loss(self, gt, pre, eps=1e-6):
        num_classes = pre.shape[1]
        # get one hot ground gruth
        gt = gt.long()
        gt_one_hot = torch.eye(num_classes)[gt.squeeze(1)]
        gt_one_hot = gt_one_hot.permute(0, 3, 1, 2).float()
        # dice loss 
        pre = pre.float()
        #print('pre.shape: ', pre.shape)
        #print('gt.shape: ', gt_one_hot.shape)
        if gt.cuda:
            gt_one_hot = gt_one_hot.cuda()
        dims = (0,) + tuple(range(2, gt.ndimension()))
        intersection = torch.sum(pre * gt_one_hot, dims)
        cardinality = torch.sum(pre + gt_one_hot, dims)
        dice = (2. * intersection / (cardinality + eps)).mean()

        return 1 - dice

    def ce_loss(self, gt, pre):
        pre = pre.permute(0,2,3,1).contiguous()
        pre = pre.view(pre.numel() // 2, 2)
        gt = gt.view(gt.numel())
        loss = F.cross_entropy(pre, gt.long())

        return loss

    def forward(self, out, labels):
        labels = labels.float()
        out = out.float()

        cond = labels[:, 0, 0, 0] >= 0 # first element of all samples in a batch 
        nnz = torch.nonzero(cond) # get how many labeled samples in a batch 
        nbsup = len(nnz)
        #print('labeled samples number:', nbsup)
        if nbsup > 0:
            masked_outputs = torch.index_select(out, 0, nnz.view(nbsup)) #select all supervised labels along 0 dimention 
            masked_labels = labels[cond]

            dice_loss = self.dice_loss(masked_labels, masked_outputs)
            #ce_loss = self.ce_loss(masked_labels, masked_outputs)

            loss = dice_loss
            return loss, nbsup
        return Variable(torch.FloatTensor([0.]).cuda(), requires_grad=False), 0

class MaskMSELoss(nn.Module):
    def __init__(self, args):
        super(MaskMSELoss, self).__init__()
        self.args = args

    def forward(self, out, zcomp, uncer, th=0.15):
        # transverse to float 
        out = out.float() # current prediction
        zcomp = zcomp.float() # the psudo label 
        uncer = uncer.float() #current prediction uncertainty
        if self.args.is_uncertain:
            mask = uncer > th
            mask = mask.float()
            mse = torch.sum(mask*(out - zcomp)**2) / torch.sum(mask) 
        else:
            mse = torch.sum((out - zcomp)**2) / out.numel()

        return mse

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pre, gt, eps=1e-6):
        num_classes = pre.shape[1]
        # get one hot ground gruth
        gt = gt.long()
        gt_one_hot = torch.eye(num_classes)[gt.squeeze(1)]
        gt_one_hot = gt_one_hot.permute(0, 3, 1, 2).float()
        # dice loss 
        pre = pre.float()
        #print('pre.shape: ', pre.shape)
        #print('gt.shape: ', gt_one_hot.shape)
        if gt.cuda:
            gt_one_hot = gt_one_hot.cuda()
        dims = (0,) + tuple(range(2, gt.ndimension()))
        intersection = torch.sum(pre * gt_one_hot, dims)
        cardinality = torch.sum(pre + gt_one_hot, dims)
        dice = (2. * intersection / (cardinality + eps)).mean()

        return 1 - dice
    
    @staticmethod
    def dice_coeficient(output, target):
        output = output.float()
        target = target.float()
        
        output = output
        smooth = 1e-20
        iflat = output.view(-1)
        tflat = target.view(-1)
        #print(iflat.shape)
        
        intersection = torch.dot(iflat, tflat)
        dice = (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

        return dice 
