import torch

class Tversky_Focal_Loss(torch.nn.Module):
    '''
    formulation: (1 - TP / (TP + aFP + (1-a)FN))^gamma
    param:
        input - tensor: (batch, ...)
        alpha: float in [0, 1] - weight for false pos case and (1- alpha) for false neg case
        gamma : γ varies in the range [1, 3]
    '''
    def __init__(self, alpha=0.5, gamma=1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        if y_pred.shape[1] != y_true.shape[1]:
            y_true = torch.cat((y_true, 1-y_true), dim=1)
        assert (y_true.shape == y_pred.shape), "predict & target shape don't match"
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_true = y_true.view(y_pred.shape[0], -1)
        
        TP_loss = (y_true*y_pred).sum(dim=1)
        FN_loss = (y_true*(1-y_pred)).sum(dim=1)
        FP_loss = ((1-y_true)*y_pred).sum(dim=1)
        
        loss = (1 - TP_loss / (TP_loss + self.alpha*FN_loss + (1-self.alpha)*FP_loss + 1e-5)).pow(1/self.gamma)
        return loss.mean()
