import torch

class FocalLoss(torch.nn.Module):
    '''
        Formaltion: L = - at * (1-pt)^gamma * log(pt)
        param: 
            Input - Torch Tensor - shape (N, ...)
            alpha: weight for positive class  and (1-alpha) for negative class
        return: 
            Mean Tensor
    '''
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.log = torch.nn.BCELoss(reduction='none')

    def forward(self, y_pred, y_true):
        if y_pred.shape[1] != y_true.shape[1]:
            y_true = torch.cat((y_true, 1-y_true), dim=1)
        assert (y_true.shape == y_pred.shape), "predict & target shape don't match"
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_true = y_true.view(y_pred.shape[0], -1)

        _alpha = torch.zeros_like(y_true)
        _alpha[y_true!=0] = self.alpha
        _alpha[y_true==0] = 1 - self.alpha
        
        logpt = self.log(y_pred, y_true)
        _pt = y_true*(1-y_pred) + (1-y_true)*y_pred
        loss = (torch.pow(_pt, self.gamma) * _alpha * logpt).mean(dim=1)
        return loss.mean()
