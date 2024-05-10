import torch
import numpy as np
from configs import get_args
from tqdm import tqdm

args = get_args()


def split_surface_data(data):
    indexes = []
    for p in args.parameters:
        if p.split("_")[-2] != "isobaricInhPa":
            indexes.append(args.parameters.index(p))

    data = data[:, indexes, :, :]
    if len(data.shape) == 3:
        data = data.unsqueeze(1)
    data = data.to(args.device)

    return data


def split_upper_air_data(data):
    levels = []
    for p in args.parameters:
        if p.split("_")[-2] == "isobaricInhPa":
            levels.append(int(p.split("_")[-1]))
    assert len(levels) % 3 == 0
    levels = list(set(levels))

    # to B C Z W H

    d = {item: idx for idx, item in enumerate(args.parameters)}
    upper_air_data = None

    for level in levels:
        params = [f"{x}_isobaricInhPa_{level}" for x in ["t", "r", "u", "v", "z"]]
        indexes = [d.get(x) for x in params]
        if upper_air_data is None:
            upper_air_data = data[:, indexes, :, :].reshape(
                data.shape[0], 5, 1, data.shape[2], data.shape[3]
            )
        else:
            upper_air_data = torch.cat(
                (
                    upper_air_data,
                    data[:, indexes, :, :].reshape(
                        data.shape[0], 5, 1, data.shape[2], data.shape[3]
                    ),
                ),
                dim=2,
            )

    upper_air_data = upper_air_data.to(args.device)

    return upper_air_data


def split_weights(weights):
    surface_indexes, upper_air_indexes = [], []

    for p in args.parameters:
        if p.split("_")[-2] != "isobaricInhPa":
            surface_indexes.append(args.parameters.index(p))
        else:
            upper_air_indexes.append(args.parameters.index(p))

    surface_weights = weights[surface_indexes].to(args.device)
    upper_air_weights = weights[upper_air_indexes].to(args.device)

    return surface_weights, upper_air_weights


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
        if leveln == "isobaricInhPa":
            w = pressure_level_weight(int(levelv))
            if name in ("u", "v"):
                w = w * 0.5
        else:
            if name == "pres" and leveln == "heightAboveSea":
                w = 1.0
            elif name in ("ucorr", "vcorr"):
                w = 0.5
            elif name in ("tcorr", "rcorr", "fgcorr", "effective-cloudiness"):
                w = 1.0
            else:
                w = 0.2

        w_list.append(w)

    return torch.tensor(np.array(w_list))


def create_static_features(ds):
    # create 100 element batches
    times_len = ds.sizes["time"]
    num_batches = times_len // 100
    if num_batches == 0:
        num_batches = 1

    indexes = list(range(ds.sizes["time"]))
    indexes = np.array_split(np.asarray(indexes), num_batches)
    means = []
    squares = []

    for batch_idx in tqdm(indexes):
        batch = ds.isel(time=batch_idx)[args.parameters].to_array().values  # C, T, Y, X
        means.append(np.mean(batch, axis=(1, 2, 3)))  # C
        squares.append(np.mean(batch**2, axis=(1, 2, 3)))  # C

    mean = np.mean(np.stack(means), axis=0)  # C
    second_moment = np.mean(np.stack(squares), axis=0)
    std = np.sqrt(second_moment - mean**2)  # (C)

    assert mean.shape[0] == len(args.parameters)

    # data = np.moveaxis(data, 0, -1)
    # data = (data - mean) / std
    # data = np.moveaxis(data, -1, 0)

    # diff = np.diff(data, axis=1)
    # diff_mean = torch.tensor(np.mean(diff, axis=(1, 2, 3)))
    # diff_std = torch.tensor(np.std(diff, axis=(1, 2, 3)))

    return torch.tensor(mean), torch.tensor(std), None, None  # diff_mean, diff_std
