import torch
import numpy as np
from scipy.special import comb


def loss_fn_M_B(y_pred, y, device, lam=1):
    loss_m = loss_fn_M(y_pred, y, device, lam)
    loss_b = loss_fn_B(y_pred, y, device)
    loss = loss_m + loss_b
    return loss


# Mean Errors
def loss_fn_M(y_pred, y, device, lam=1):
    batch_size, future = y.shape
    loss1 = torch.linalg.vector_norm(y - y_pred, dim=1)
    loss2 = torch.zeros(batch_size, device=device)
    for i in range(future):
        wi = np.log(future - i + 1)
        p = y[:, i]
        q = y_pred[:, i] + 1e-8
        loss2 += lam * wi * (-(p * torch.log2(q)) - ((1 - p) * torch.log2(1 - q)))
    loss = torch.mean(loss1 + loss2)

    return loss


# Bhattacharya Distance
def loss_fn_B(y_pred, y, device):
    batch_size, future = y.shape
    rate_real = 1 - torch.mean(y, dim=1)
    rate_real[rate_real == 1] = 1 - 1 / future
    rate_real[rate_real == 0] = 1 / future  # Avoid inf loss numericaly
    rate_pred = 1 - torch.mean(y_pred, dim=1)

    A = torch.tensor([comb(future, t) for t in range(future)]).repeat([batch_size, 1]).to(device)
    B = rate_pred * rate_real  # + 1e-14
    C = (1 - rate_pred) * (1 - rate_real)  # + 1e-14

    BC = torch.zeros([batch_size, future], device=device)
    for t in range(future):
        BC[:, t] = A[:, t] * B ** (t / 2) * C ** ((future - t) / 2)  # + 1e-14
    BC = torch.sum(BC, dim=1)
    BD = -torch.log(BC)
    loss = torch.mean(BD)

    if torch.isinf(loss):
        a = 5

    return loss

