import torch
import numpy as np
from configs import get_args
from tqdm import tqdm

args = get_args()


def split_surface_data(data):
    assert args.parameters[0] == "effective_cloudiness"
    assert "pres_heightAboveSea_0" not in args.parameters

    data = data[:, 0, :, :]
    if len(data.shape) == 3:
        data = data.unsqueeze(1)
    data = data.to(args.device)
    return data


def split_upper_air_data(data):
    levels = list(set([int(x.split("_")[-1]) for x in args.parameters[1:]]))
    assert len(args.parameters[1:]) % 3 == 0

    # to B C Z W H

    d = {item: idx for idx, item in enumerate(args.parameters)}

    upper_air_data = None
    for level in levels:
        params = [f"{x}_isobaricInhPa_{level}" for x in ["u", "v", "z"]]
        indexes = [d.get(x) for x in params]
        if upper_air_data is None:
            upper_air_data = data[:, indexes, :, :].reshape(
                data.shape[0], 3, 1, data.shape[2], data.shape[3]
            )
        else:
            upper_air_data = torch.cat(
                (
                    upper_air_data,
                    data[:, indexes, :, :].reshape(
                        data.shape[0], 3, 1, data.shape[2], data.shape[3]
                    ),
                ),
                dim=2,
            )

    upper_air_data = upper_air_data.to(args.device)

    return upper_air_data


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

    return torch.tensor(np.array(w_list))


def create_static_features(ds):

    # create 100 element batches
    times_len = ds.sizes["time"]
    num_batches = times_len // 100

    indexes = np.array_split(np.asarray(list(range(ds.sizes['time']))), num_batches)
    means = []
    squares = []

    for batch_idx in tqdm(indexes):
        batch = ds.isel(time=batch_idx)[args.parameters].to_array().values # C, T, Y, X
        means.append(np.mean(batch, axis=(1,2,3))) # C
        squares.append(np.mean(batch**2, axis=(1,2,3))) # C

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

    return mean, std, None, None  # diff_mean, diff_std
