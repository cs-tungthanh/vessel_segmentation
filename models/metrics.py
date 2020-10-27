import torch
import numpy as np

'''
:param
    - output, target must be the same size: 
            (batch size, depth, channel, width, heigh)
            2D(b - c - h - w) 
        or  3D(b - c - d - h - w) 
'''

def iou(y_pred, y_true):
    # flatten by batch element
    y_pred = y_pred.view(y_pred.shape[0], -1)
    y_true = y_true.view(y_pred.shape[0], -1)
    thres_preds = (y_pred >= 0.5).float()

    nominator = torch.sum(thres_preds*y_true, 1)
    denominator = torch.sum(thres_preds, 1) + torch.sum(y_true, 1) - nominator  + 1e-5
    return torch.mean(nominator/denominator)

def VOE(y_pred, y_true):
    # volumetric overlap error - VOE
    return 1 - iou(y_pred, y_true)

def VD(y_pred, y_true):
    # Relative Volume Difference - RVD
    # flatten by batch element
    y_pred = y_pred.view(y_pred.shape[0], -1)
    y_true = y_true.view(y_pred.shape[0], -1)
    thres_preds = (y_pred >= 0.5).float()

    pred = torch.sum(thres_preds, dim=1)
    gt = torch.sum(y_true, dim=1)
    VD = (pred - gt + 1e-5) / (gt + 1e-5)
    return torch.mean(VD)

def dice(y_pred, y_true):
    # flatten by batch element
    y_pred = y_pred.view(y_pred.shape[0], -1)
    y_true = y_true.view(y_pred.shape[0], -1)
    thres_preds = (y_pred >= 0.5).float()

    nominator = 2 * torch.sum(thres_preds*y_true, 1)
    denominator = torch.sum(thres_preds, 1) + torch.sum(y_true, 1) + 1e-5
    return torch.mean(nominator/denominator)

def accuracy(y_pred, y_true):
    # flatten by batch element
    y_pred = y_pred.view(y_pred.shape[0], -1)
    y_true = y_true.view(y_pred.shape[0], -1)
    thres_preds = (y_pred >= 0.5).float()

    TP = torch.sum(thres_preds*y_true, 1)
    TN = torch.sum((1-thres_preds)*(1-y_true), 1)
    FP = torch.sum(thres_preds*(1-y_true), 1)
    FN = torch.sum((1-thres_preds)*y_true, 1)
    return torch.mean((TP+TN) / (TP+TN+FP+FN + 1e-5))

def recall(y_pred, y_true):
    # flatten by batch element
    y_pred = y_pred.view(y_pred.shape[0], -1)
    y_true = y_true.view(y_pred.shape[0], -1)
    thres_preds = (y_pred >= 0.5).float()

    TP = torch.sum(thres_preds*y_true, 1)
    FN = torch.sum((1-thres_preds)*y_true, 1)
    return torch.mean(TP / (TP+FN + 1e-5))

def specificity(y_pred, y_true):
    # flatten by batch element
    y_pred = y_pred.view(y_pred.shape[0], -1)
    y_true = y_true.view(y_pred.shape[0], -1)
    thres_preds = (y_pred >= 0.5).float()

    TN = torch.sum((1-thres_preds)*(1-y_true), 1)
    FP = torch.sum(thres_preds*(1-y_true), 1)
    return torch.mean(TN / (TN+FP + 1e-5))

def precision(y_pred, y_true):
    # positive predictive value (PPV)
    # flatten by batch element
    y_pred = y_pred.view(y_pred.shape[0], -1)
    y_true = y_true.view(y_pred.shape[0], -1)
    thres_preds = (y_pred >= 0.5).float()

    TP = torch.sum(thres_preds*y_true, 1)
    FP = torch.sum(thres_preds*(1-y_true), 1)
    return torch.mean(TP / (TP + FP + 1e-5))

def NPV(y_pred, y_true):
    # flatten by batch element
    y_pred = y_pred.view(y_pred.shape[0], -1)
    y_true = y_true.view(y_pred.shape[0], -1)
    thres_preds = (y_pred >= 0.5).float()

    TN = torch.sum((1-thres_preds)*(1-y_true), 1)
    FN = torch.sum((1-thres_preds)*y_true, 1)
    return torch.mean(TN / (TN + FN + 1e-5))

def FPR(y_pred, y_true):
    # flatten by batch element
    y_pred = y_pred.view(y_pred.shape[0], -1)
    y_true = y_true.view(y_pred.shape[0], -1)
    thres_preds = (y_pred >= 0.5).float()

    TN = torch.sum((1-y_true)*(1-thres_preds), 1)
    FP = torch.sum((1-y_true)*thres_preds, 1)
    return torch.mean(FP / (FP + TN + 1e-5))

def FNR(y_pred, y_true):
    # flatten by batch element
    y_pred = y_pred.view(y_pred.shape[0], -1)
    y_true = y_true.view(y_pred.shape[0], -1)
    thres_preds = (y_pred >= 0.5).float()

    FN = torch.sum(y_true*(1-thres_preds), 1)
    TP = torch.sum(y_true*thres_preds, 1)
    return torch.mean(FN / (FN + TP + 1e-5))
