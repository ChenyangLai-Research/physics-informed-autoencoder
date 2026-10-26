import numpy as np
import torch
from scipy.io import loadmat
from model import AutoEncoder, euc_loss
from runtime import tensor, normalize, require, fit, finish


def run(a):
    if a.demo:
        rng = np.random.default_rng(a.seed)
        t = np.linspace(0, 1.8, 90)
        pos = np.array(
            [
                (0.5 + 0.1 * rng.normal()) * np.sin(t * 3 + 0.05 * rng.normal())
                for _ in range(a.max_samples)
            ]
        )
        cur = np.gradient(pos, 0.02, axis=1) * 0.2 + 0.02 * rng.normal(size=pos.shape)
    else:
        d = loadmat(
            require(a.data_root, "paper/code_inpaper/healthy.mat"),
            variable_names=["AME_dis_mat", "current_mat"],
        )
        ix = np.linspace(0, 1800, 91, dtype=int)[1:]
        pos = d["AME_dis_mat"][: a.max_samples, ix]
        cur = d["current_mat"][: a.max_samples, ix]
    indices = np.random.default_rng(a.seed).permutation(len(pos))
    n = int(0.8 * len(pos))
    tr, va = indices[:n], indices[n:]
    x, xv, sx = normalize(pos[tr], pos[va])
    y, yv, sy = normalize(cur[tr], cur[va])
    # Physical current/velocity envelope is fitted only to the training simulations.
    v = np.diff(pos[tr], axis=1) / 0.02
    c = (cur[tr, :-1] + cur[tr, 1:]) / 2
    coordinate = np.stack([c, v], axis=-1)
    mean = coordinate.mean(0)
    threshold = np.percentile(np.linalg.norm(coordinate - mean, axis=2), 50, axis=0)
    x, y, xv = tensor(x), tensor(y), tensor(xv)
    model = AutoEncoder()
    segments = [np.arange(39), np.arange(39, 70), np.arange(70, 89)]

    def loss(idx):
        rec = model(x[idx])[1]
        mse = (rec - x[idx]).square().mean()
        physics = sum(
            euc_loss(rec, y[idx], seg, mean[seg], threshold[seg], sx, sy)
            for seg in segments
        )
        return mse + physics, {"reconstruction": mse, "physics": physics}

    history = fit(model, loss, n, a)
    model.eval()
    with torch.no_grad():
        pred = model(xv)[1].numpy()
    scores = np.mean((pred - xv.numpy()) ** 2, axis=1)
    return finish(
        model,
        a,
        history,
        pred,
        xv.numpy(),
        {
            "task": "healthy position reconstruction",
            "physics_weight": 1.0,
            "heldout_reconstruction_scores": scores.tolist(),
            "architecture": [90, 60, 30, 10, 3, 10, 30, 60, 90],
        },
        {"position": sx, "current": sy},
    )
