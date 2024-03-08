import numpy as np
import glob
import xarray as xr
import zarr
from datetime import datetime, timedelta
from enum import Enum
import copy
import torch
from configs import get_args
import os
from torch.utils.data import IterableDataset

args = get_args()


def read_xarray_dataset(dirname):
    ds = xr.open_mfdataset(
        "{}/*.zarr".format(dirname), engine="zarr", data_vars=args.parameters
    )
    return ds


def get_mean_std(ds):
    data = ds.to_array().values

    if len(data.shape) == 3:
        data = np.expand_dims(data, axis=1)

    mean = np.mean(data, axis=(1, 2, 3))
    std = np.std(data, axis=(1, 2, 3))

    return mean, std


def create_generators(train_val_split=0.8):
    if args.dataseries_directory is not None:
        print("Reading data from directory: {}".format(args.dataseries_directory))
        ds = read_xarray_dataset(args.dataseries_directory)
    elif args.dataseries_file is not None:
        print("Reading data from file: {}".format(args.dataseries_file))
        ds = xr.open_dataset(args.dataseries_file)
    else:
        raise ValueError("No dataseries file or directory specified")

    if os.path.exists("parameter_mean.pt"):
        mean = torch.load("parameter_mean.pt")
        std = torch.load("parameter_std.pt")
    else:
        mean, std = get_mean_std(ds[args.parameters])
        torch.save(mean, "parameter_mean.pt")
        torch.save(std, "parameter_std.pt")

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

        y = torch.squeeze(y, 0)

        if len(x.shape) > 3:
            x = x.reshape(-1, args.input_size[0], args.input_size[1])

        assert len(x.shape) == 3, f"x shape is not (x,x,x), its {x.shape}"
        assert len(y.shape) == 3, f"y shape is not (x,x,x), its {y.shape}"

        if y.shape == args.input_size:
            y = torch.unsqueeze(y, 0)

        assert x.shape == (
            args.n_hist * len(args.parameters),
            args.input_size[0],
            args.input_size[1],
        ), f"x shape is {x.shape}, should be ({args.n_hist * len(args.parameters)}, {args.input_size[0]}, {args.input_size[1]})"
        assert y.shape == (
            args.n_pred * len(args.parameters),
            args.input_size[0],
            args.input_size[1],
        ), f"y shape is {y.shape}, should be ({args.n_pred * len(args.parameters)}, {args.input_size[0]}, {args.input_size[1]})"

        # torch.Size([1, 2, 224, 224])
        # to
        # torch.Size([1, 50176, 2])

        x_orig_shape = x.shape
        y_orig_shape = y.shape
        x = x.reshape(args.n_hist, len(args.parameters), -1).permute(0, 2, 1)
        y = y.reshape(args.n_pred, len(args.parameters), -1).permute(0, 2, 1)

        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        # .. and back to original shape

        x = x.permute(0, 2, 1).reshape(x_orig_shape)
        y = y.permute(0, 2, 1).reshape(y_orig_shape)

        return (x, y)

    def __call__(self):
        for i in range(len(self.__len())):
            elem = self.__getitem__(i)
            yield elem


class SAFDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator()

    def __len__(self):
        return len(self.generator)
