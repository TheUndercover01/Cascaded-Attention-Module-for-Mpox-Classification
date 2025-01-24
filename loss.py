import torch.nn as nn
import torch
import torch.nn.functional as F




class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:  # 'none'
            return focal_loss


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_acc = float('-inf')

    def early_stop(self, validation_acc):
        if validation_acc > self.max_validation_acc:
            self.max_validation_acc = validation_acc
            self.counter = 0
        elif validation_acc <= (self.max_validation_acc - self.min_delta):
            print("max_acc", self.max_validation_acc)
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class FocalCELoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', num_classes=4, device=None):
        """
        Focal Cross Entropy Loss

        Args:
            alpha (torch.Tensor, optional): Weight for each class. Must be of size C.
            gamma (float): Focusing parameter
            reduction (str): 'none' | 'mean' | 'sum'
            num_classes (int): Number of classes
            device (torch.device): Device to put the weights on
        """
        super(FocalCELoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Handle class weights (alpha)
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha)
            else:
                self.alpha = alpha
            assert len(self.alpha) == num_classes, "Alpha size must match number of classes"
            # Move alpha to the correct device
            self.alpha = self.alpha.to(self.device)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Tensor of shape (N, C) where C is the number of classes
            targets: Tensor of shape (N,) with values in [0, C-1]
        """
        # Ensure inputs and targets are on the same device as alpha
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha,
                                reduction='none')

        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:  # 'none'
            return focal_loss



class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, lambda_c=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long().to(x.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return self.lambda_c * loss