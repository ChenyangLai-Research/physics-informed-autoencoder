"""Core model extracted from archived research code; see project.json and CHANGELOG.md."""

import numpy as np
import torch
from torch import nn

device = torch.device("cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(90, 60),
            nn.Tanh(),
            nn.Linear(60, 30),
            nn.Tanh(),
            nn.Linear(30, 10),
            nn.Tanh(),
            nn.Linear(10, 3),  # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 10),
            nn.Tanh(),
            nn.Linear(10, 30),
            nn.Tanh(),
            nn.Linear(30, 60),
            nn.Tanh(),
            nn.Linear(60, 90),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def euc_loss(decoded, y, ind, mean_cordi, thr, x_scaler, y_scaler):

    # inverse transform
    decoded = decoded * (
        torch.from_numpy(x_scaler.data_max_ - x_scaler.data_min_)
    ) + torch.from_numpy(x_scaler.data_min_)
    y = y * (
        torch.from_numpy(y_scaler.data_max_ - y_scaler.data_min_)
    ) + torch.from_numpy(y_scaler.data_min_)

    thr = torch.from_numpy(thr).type(torch.FloatTensor)
    mean_cordi = torch.from_numpy(mean_cordi).type(torch.FloatTensor)

    decoded_vel = torch.diff(decoded, n=1, axis=1) / 0.02
    cur = (y[:, :-1] + y[:, 1:]) / 2

    cur_part = cur[:, ind]
    vel_part = decoded_vel[:, ind]

    cur_sq = (cur_part - mean_cordi[:, 0]).pow(2)
    decoded_sq = (vel_part - mean_cordi[:, 1]).pow(2)

    dist = torch.sqrt(cur_sq + decoded_sq)

    relu = nn.ReLU()
    loss = torch.mean(relu(dist - thr))

    return loss
