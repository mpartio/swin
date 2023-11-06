import torch
from skimage.metrics import structural_similarity
from torch import nn
import numpy as np
from configs import get_args

args = get_args()


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def MAE(pred, true):
    return np.mean(np.abs(pred - true), axis=(0, 1)).sum()


def MSE(pred, true):
    return np.mean((pred - true) ** 2, axis=(0, 1)).sum()


def compute_metrics(predictions, targets):
    targets = targets.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()

    batch_size = predictions.shape[0]
    seq_len = predictions.shape[1]

    ssim = 0

    for batch in range(batch_size):
        for frame in range(seq_len):
            ssim += structural_similarity(
                targets[batch, frame].squeeze(),
                predictions[batch, frame].squeeze(),
                data_range=1.0,
            )

    ssim /= batch_size * seq_len

    mse = MSE(predictions, targets)
    mae = MAE(predictions, targets)
    bcef = nn.BCEWithLogitsLoss()
    bce = bcef(torch.from_numpy(predictions), torch.from_numpy(targets)).numpy()

    return mse, mae, ssim, bce
