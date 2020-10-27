import torch

class BCELoss(torch.nn.Module):
    """
    Binary Cross Entropy Loss
    Args:
        shape: (batch_size, ...) 
        y_pred (torch tensor): tensor of predictions values for 2-class.
        y_true (torch tensor): tensor of ground truth for 2-class.
    Returns:
        BCELoss (torch tensor): array have the same size of y_pred  
    """
    def __init__(self, reduction='none'):
        super().__init__()
        self.loss = torch.nn.BCELoss(reduction=reduction)

    def forward(self, y_pred, y_true):
        if y_pred.shape[1] != y_true.shape[1]:
            y_true = torch.cat((y_true, 1-y_true), dim=1)
        assert (y_true.shape == y_pred.shape), 'predict & target shape do not match'
        return self.loss(y_pred, y_true)
