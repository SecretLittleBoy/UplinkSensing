import torch


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class MSE_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.MSE = torch.nn.MSELoss()

    def forward(self, X, Y):
        return self.MSE(X, Y)
