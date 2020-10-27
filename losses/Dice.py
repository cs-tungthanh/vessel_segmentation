import torch

class DiceLoss(torch.nn.Module):
    """
    Soft Dice Loss for multi-classes
    Args:
        input - torch tensor - shape: (batch_size, ...)
        is_soft (boolean): whether use soft Dice loss or not
    Returns:
        soft_dice_loss (float): computed value of dice loss.     
    """
    def __init__(self, is_soft=False):
        super().__init__()
        self.epsilon = 1e-5
        self.is_soft = is_soft

    def forward(self, y_pred, y_true):
        if y_pred.shape[1] != y_true.shape[1]:
            y_true = torch.cat((y_true, 1-y_true), dim=1)
        assert (y_true.shape == y_pred.shape), 'predict & target shape do not match'
        # flatten by batch element
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_true = y_true.view(y_pred.shape[0], -1)
       
        # compute loss
        numerator = 2 * torch.sum(y_true*y_pred, 1) + self.epsilon
        if self.is_soft:
            denominator = torch.sum(y_true**2, 1) + torch.sum(y_pred**2, 1) + self.epsilon
        else:
            denominator = torch.sum(y_true, 1) + torch.sum(y_pred, 1) + self.epsilon
        return  torch.mean(1.0 - numerator / denominator)

class GeneralizedDice(torch.nn.Module):
    '''
        Formaltion: L = 1 - 2 * (w*intersection)/(w*union)
                    w compute by the sum of positive/negative respectively
        param: 
            Input - Torch Tensor - shape (N, ...)
        return: 
            Mean Tensor
    '''
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        assert (y_true.shape == y_pred.shape), "predict & target shape don't match"
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_true = y_true.view(y_pred.shape[0], -1)
        
        w_p = 1 / (y_true.sum(1)).pow(2)
        w_n = 1/ ((1-y_true).sum(1)).pow(2)
        intersection = w_p * (y_pred*y_true).sum(1) + w_n *((1-y_pred)*(1-y_true)).sum(1)
        union = w_p * (y_pred.sum(1) + y_true.sum(1)) + w_n * ((1-y_pred).sum() + (1-y_true).sum())
        loss = 1 - 2 * (intersection / union)
        return loss.mean()

class GeneralizedDice1(torch.nn.Module):
    '''
        Formaltion: L = 1 - 2 * (w*intersection)/(w*union)
                    w compute by the sum of positive/negative respectively
        param: 
            Input - Torch Tensor - shape (N, ...)
        return: 
            Mean Tensor
    '''
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        assert (y_true.shape == y_pred.shape), "predict & target shape don't match"
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_true = y_true.view(y_pred.shape[0], -1)
        
        w_p = 1 / (y_true.sum(1)).pow(2)
        intersection = w_p * (y_pred*y_true).sum(1)
        union = w_p * (y_pred.sum(1) + y_true.sum(1))
        loss = 1 - 2 * (intersection / union)
        return loss.mean()
        