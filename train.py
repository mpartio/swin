from swin import create_model
from saf import create_generators
from configs import get_args
from utils import compute_metrics
from torch.utils.data import DataLoader
from torch.cuda import amp
from torch.cuda.amp import autocast as autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import os
import time

args = get_args()
scaler = amp.GradScaler()
parameter_weights = None

assert args.n_pred == 1, "Only n_pred=1 is supported for now"


def create_parameter_weights():
    def pressure_level_weight(x: float):
        # Create similar weighting to pressure level as in graphcast paper.
        # See fig 6 in graphcast paper
        # In summaru the weights are normalized to sum to 1 so that the highest
        # pressure level has the smallest weight.

        plevels = np.asarray([300, 500, 700, 850, 925, 1000])
        plevels_norm = plevels / np.sum(plevels)

        y = plevels_norm[np.where(plevels == x)][0]

        return round(y, 4)

    w_list = []
    for par in args.parameters:
        if par == "effective_cloudiness":
            w = 1.0
        else:
            name, leveln, levelv = par.split("_")
            if leveln == "isobaricInhPa":
                w = pressure_level_weight(int(levelv))
                if name in ("u", "v"):
                    w = w * 0.5
            elif name == "pres" and leveln == "heightAboveSea":
                w = 0.2

        w_list.append(w)

    w_list = torch.tensor(np.array(w_list))

    return w_list


def model_forward(m, inputs):
    assert inputs.shape == (
        args.batch_size,
        (args.n_hist * len(args.parameters)),
        args.input_size[0],
        args.input_size[1],
    ), "inputs.shape: {}, should be ({}, {}, {}, {})".format(
        inputs.shape,
        args.batch_size,
        (args.n_hist * len(args.parameters)),
        args.input_size[0],
        args.input_size[1],
    )

    outputs = []

    for i in range(args.n_pred):
        y = m(inputs)

        if len(y.shape) == 3:
            y = y.unsqueeze(1)

        outputs.append(y)

        if i < args.n_pred - 1:
            inputs = torch.cat((inputs[:, 1:, :], y), dim=1)

    # X B C H W
    # [1, 16, 1, 224, 224]
    outputs = torch.stack(outputs)

    # Remove extra dimension X
    # B C H W
    # [16, 1, 224, 224]
    outputs = torch.squeeze(outputs, dim=0)

    # B C X H W
    # [1, 16, 1, 224, 224]
    # [1, 1, 20, 224, 224]
    # outputs = torch.permute(outputs, (2, 1, 0, 3, 4))

    assert outputs.shape == (
        args.batch_size,
        (args.n_pred * len(args.parameters)),
        args.input_size[0],
        args.input_size[1],
    ), "outputs.shape: {}, should be ({}, {}, {}, {})".format(
        outputs.shape,
        args.batch_size,
        (args.n_pred * len(args.parameters)),
        args.input_size[0],
        args.input_size[1],
    )

    return outputs


def weighted_loss(y_pred, y_true):
    # y_pred, y_true: B C H W
    # multiply channels with parameter weights and get mean
    # which is then the weighted loss

    w_loss = torch.einsum(
        "nchw,c->nchw", (y_pred - y_true) ** 2, parameter_weights
    ).mean()

    return w_loss


def setup():
    train_ds, valid_ds = create_generators(train_val_split=0.8)

    m = create_model(args.model_name)

    if args.load_model:
        m.load_state_dict(
            torch.load(
                "{}/trained_model_state_dict".format(args.model_dir),
                map_location=torch.device(args.device),
            )
        )

        print("Loaded model from {}".format(args.model_dir))

    optimizer = torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-6)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        pin_memory=True,
        drop_last=True,
    )

    criterion = weighted_loss

    global parameter_weights
    parameter_weights = create_parameter_weights().to(args.device)

    return m, criterion, optimizer, train_loader, valid_loader


def train_single(epoch, m, loader, criterion, optimizer):
    m.train()
    num_batches = len(loader)
    losses = []

    for b_idx, (inputs, targets) in enumerate(tqdm(loader)):
        optimizer.zero_grad()
        inputs, targets = map(lambda x: x.float().to(args.device), (inputs, targets))

        with autocast():
            outputs = model_forward(m, inputs)

            assert (
                outputs.shape == targets.shape
            ), "outputs.shape: {}, targets.shape: {}".format(
                outputs.shape, targets.shape
            )
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())

    return np.mean(losses)


def test(epoch, m, loader, criterion):
    m.eval()
    num_batches = len(loader)
    losses, mses, maes, ssims, bces = [], [], [], [], []

    for batch_idx, (inputs, targets) in enumerate(tqdm(loader)):
        with torch.no_grad():
            inputs, targets = map(
                lambda x: x.float().to(args.device), [inputs, targets]
            )

            outputs = model_forward(m, inputs)

            losses.append(criterion(outputs, targets).item())

            mse, mae, ssim, bce = compute_metrics(outputs, targets)

            mses.append(mse)
            maes.append(mae)
            ssims.append(ssim)
            bces.append(bce)

    return np.mean(losses), np.mean(mses), np.mean(maes), np.mean(ssims), np.mean(bces)


def train(m, criterion, optimizer, train_loader, valid_loader, epochs=500):
    best_metric = (
        0,
        float("inf"),
        float("inf"),
        float("inf"),
        float("inf"),
        float("inf"),
    )

    early_stop = 20
    os.makedirs(args.model_dir, exist_ok=True)
    with open(f"{args.model_dir}/args.txt", "w") as f:
        print(args, file=f)
    with open(f"{args.model_dir}/model-name.txt", "w") as f:
        print(args.model_name, file=f)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=0.0001,
        threshold_mode="abs",
        verbose=True,
    )

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train_single(epoch, m, train_loader, criterion, optimizer)

        if True:  # (epoch + 1) % 2 == 0:
            valid_loss, mse, mae, ssim, bce = test(epoch, m, valid_loader, criterion)

            print("Validation loss for this epoch: {:.5f}".format(valid_loss))
            if valid_loss < best_metric[1]:
                torch.save(m.state_dict(), f"{args.model_dir}/trained_model_state_dict")
                best_metric = (epoch, valid_loss, mse, mae, ssim, bce)
                print(
                    f"Saved current state to {args.model_dir}/trained_model_state_dict"
                )

            print(
                "Current best: Epoch:{:03d} LOSS:{:.5f} MSE:{:.0f} MAE:{:.0f} SSIM:{:.3f} BCE:{:.3f}".format(
                    best_metric[0],
                    best_metric[1],
                    best_metric[2],
                    best_metric[3],
                    best_metric[4],
                    best_metric[5],
                )
            )

            if epoch - best_metric[0] > early_stop:
                print(
                    "More than {} epoch without improvements - early stop".format(
                        early_stop
                    )
                )
                break

        scheduler.step(train_loss)

        print(f" Epoch {epoch}/{epochs}: time usage: {time.time() - start_time:.0f}s")


def main():
    m, criterion, optimizer, tr_l, va_l = setup()
    train(m, criterion, optimizer, tr_l, va_l)


if __name__ == "__main__":
    main()
