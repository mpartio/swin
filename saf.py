import numpy as np
import glob
import xarray as xr
import zarr
import sys
import os
import torch
import einops
from datetime import datetime, timedelta
from enum import Enum
from configs import get_args
from torch.utils.data import IterableDataset
from pangu_utils import create_parameter_weights, create_static_features

args = get_args()


def read_xarray_dataset(dirname):
    ds = xr.open_mfdataset(
        "{}/*.zarr".format(dirname), engine="zarr", data_vars="minimal"
    )

    return ds


def create_generators(train_val_split=0.8, always_recalculate_missing_mean=False):
    if args.dataseries_directory is not None:
        print("Reading data from directory: {}".format(args.dataseries_directory))
        ds = read_xarray_dataset(args.dataseries_directory)
    elif args.dataseries_file is not None:
        print("Reading data from file: {}".format(args.dataseries_file))
        ds = xr.open_dataset(args.dataseries_file)
    else:
        raise ValueError("No dataseries file or directory specified")

    if os.path.exists(f"{args.load_model_from}/parameter_mean.pt"):
        mean = torch.load(f"{args.load_model_from}/parameter_mean.pt")
        std = torch.load(f"{args.load_model_from}/parameter_std.pt")

        if len(mean) != len(args.parameters):
            if always_recalculate_missing_mean is False:
                print(
                    "Mean and parameters length mismatch: were means calculated from another dataset? Remove parameter_mean.pt and parameter_std.pt and try again"
                )
                sys.exit(1)

            mean, std, _, _ = create_static_features(ds[args.parameters])

    else:
        print("Parameter static features not found, calculating them now... ", end="")
        mean, std, _, _ = create_static_features(ds[args.parameters])
        torch.save(mean, "parameter_mean.pt")
        torch.save(std, "parameter_std.pt")
        w = create_parameter_weights()
        torch.save(w, "parameter_weights.pt")
        print("done")

    ds_len = len(ds["time"])
    sample_length = args.n_hist + args.n_pred

    indexes = np.arange(0, ds_len - sample_length, sample_length)

    train_len = int(len(indexes) * train_val_split)
    train_indexes = indexes[0:train_len]
    val_indexes = indexes[train_len:]

    train_indexes = (
        np.repeat(train_indexes, sample_length).reshape(-1, sample_length)
        + np.arange(sample_length)
    ).flatten()

    val_indexes = (
        np.repeat(val_indexes, sample_length).reshape(-1, sample_length)
        + np.arange(sample_length)
    ).flatten()

    train_ds = ds.isel(time=train_indexes)
    val_ds = ds.isel(time=val_indexes)

    train_gen = SAFDataGenerator(
        ds=train_ds,
        mean=mean,
        std=std,
    )
    val_gen = SAFDataGenerator(ds=val_ds, mean=mean, std=std)

    print("Train generator number of samples: {}".format(len(train_gen)))
    print("Validation generator number of samples: {}".format(len(val_gen)))

    return train_gen, val_gen


class SAFDataGenerator:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.sample_length = args.n_hist + args.n_pred

    def __str__(self):
        return "SAFDataGenerator: {}".format(self.__dict__)

    def __len__(self):
        """Return number of samples in this dataset"""
        return int(len(self.ds.time) / self.sample_length)

    def __getitem__(self, idx):
        indexes = np.arange(idx, idx + self.sample_length)
        data = self.ds.isel(time=indexes)[args.parameters].to_array().values

        if len(data.shape) == 3:
            data = np.expand_dims(data, axis=1)
        else:
            data = np.moveaxis(data, 0, 1)

        x = torch.from_numpy(data[: args.n_hist]).contiguous()
        y = torch.from_numpy(data[args.n_hist :]).contiguous()

        assert len(x.shape) == 4, f"x shape is not (x,x,x,x), its {x.shape}"
        assert len(y.shape) == 4, f"y shape is not (x,x,x,x), its {y.shape}"

        assert x.shape == (
            args.n_hist,
            len(args.parameters),
            args.input_size[1],
            args.input_size[0],
        ), f"x shape is {x.shape}, should be ({args.n_hist, len(args.parameters)}, {args.input_size[1]}, {args.input_size[0]})"
        assert y.shape == (
            args.n_pred,
            len(args.parameters),
            args.input_size[1],
            args.input_size[0],
        ), f"y shape is {y.shape}, should be ({args.n_pred, len(args.parameters)}, {args.input_size[1]}, {args.input_size[0]})"

        _, _, h, w = x.shape

        # reshape so that channels are last

        x = einops.rearrange(x, "t c h w -> t (h w) c", h=h, w=w)
        y = einops.rearrange(y, "t c h w -> t (h w) c", h=h, w=w)

        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        # .. and back to original shape

        x = einops.rearrange(x, "t (h w) c -> t c h w", h=h, w=w)
        y = einops.rearrange(y, "t (h w) c -> t c h w", h=h, w=w)

        return (x, y)

    def __call__(self):
        for i in range(len(self.__len())):
            elem = self.__getitem__(i)
            yield elem

    def get_static_features(self, parameter):
        assert parameter in ("lsm_heightAboveGround_0", "z_heightAboveGround_0")
        return torch.tensor(self.ds[parameter].values)


class SAFDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator()

    def __len__(self):
        return len(self.generator)
