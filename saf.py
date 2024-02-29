import numpy as np
import glob
import xarray as xr
import zarr
from datetime import datetime, timedelta
from enum import Enum
import copy
import torch
from configs import get_args

from torch.utils.data import IterableDataset

args = get_args()


def read_xarray_dataset(dirname):
    ds = xr.open_mfdataset(
        "{}/*.zarr".format(dirname), engine="zarr", data_vars=["effective_cloudiness"]
    )
    print(ds)
    return ds


def create_generators(train_val_split=0.8, sample_length=5):
    ds = read_xarray_dataset(args.dataseries_directory)

    ds_len = len(ds["time"])

    indexes = np.arange(0, ds_len - sample_length, sample_length)
    #    np.random.shuffle(indexes)

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

    train_gen = SAFDataGenerator(ds=train_ds, sample_length=sample_length)
    val_gen = SAFDataGenerator(ds=val_ds, sample_length=sample_length)

    print("Train generator number of samples: {}".format(len(train_gen)))
    print("Validation generator number of samples: {}".format(len(val_gen)))

    return train_gen, val_gen


class SAFDataGenerator:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __len__(self):
        # """Return number of batches in this dataset"""
        # return len(self.placeholder) // self.batch_size
        """Return number of samples in this dataset"""
        return int(len(self.ds.time) / self.sample_length)

    def __getitem__(self, idx):
        indexes = np.arange(idx, idx + self.sample_length)
        data = self.ds["effective_cloudiness"][indexes, :, :].values

        x = torch.from_numpy(data[:-1]).contiguous()
        y = torch.unsqueeze(torch.from_numpy(data[-1]).contiguous(), 0)

        assert x.shape == (
            self.sample_length - 1,
            args.input_size[0],
            args.input_size[1],
        ), f"x shape is {x.shape}, should be ({self.sample_length}, {args.input_size[0]}, {args.input_size[1]})"
        assert y.shape == (
            1,
            args.input_size[0],
            args.input_size[1],
        ), f"y shape is {y.shape}, should be (1, {args.input_size[0]}, {args.input_size[1]})"

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


class SAFDataLoader:
    def __init__(self, **kwargs):
        self.img_size = (args.input_size, args.input_size)
        self.include_datetime = False
        self.include_topography = False
        self.include_terrain_type = False
        self.include_sun_elevation_angle = False
        self.dataseries_file = args.dataseries_file
        self.batch_size = args.batch_size

        self._placeholder = []
        self.leadtimes = []

        self.initialize()

    def initialize(self):
        # Read static datas, so that each dataset generator
        # does not have to read them

        if args.leadtime_conditioning:
            self.leadtimes = np.asarray(
                [
                    np.full(self.img_size, x / args.leadtime_conditioning).astype(
                        np.float32
                    )
                    for x in range(args.leadtime_conditioning)
                ]
            )

        if self.include_topography:
            self.topography_data = np.expand_dims(
                create_topography_data(self.img_size), axis=0
            )

        if self.include_terrain_type:
            self.terrain_type_data = np.expand_dims(
                create_terrain_type_data(self.img_size), axis=0
            )

        if self.include_sun_elevation_angle:
            if self.operating_mode == OpMode.INFER:
                self.sun_elevation_angle_data = {}
                for i in range(args.n_pred):
                    ts = self.analysis_time + timedelta(minutes=(1 + i) * 15)
                    self.sun_elevation_angle_data[
                        ts.strftime("%Y%m%dT%H%M%S")
                    ] = create_sun_elevation_angle(ts, (128, 128))
            else:
                self.sun_elevation_angle_data = create_sun_elevation_angle_data(
                    self.img_size,
                )

        # create placeholder data

        self.xds = read_xarray_dataset(args.dataseries_directory)

        i = 0

        step = args.n_hist + args.n_pred
        n_fut = args.n_pred

        assert (
            len(self.elements) - step
        ) >= 0, "Too few data to make a prediction: {} (need at least {})".format(
            len(self.elements), step
        )

        if args.leadtime_conditioning == 0:
            while i <= len(self.elements) - step:
                x = list(self.elements[i : i + args.n_hist])
                y = list(self.elements[i + args.n_hist : i + step])

                self._placeholder.append([x, y])

                i += step

        else:
            step = args.n_hist + args.leadtime_conditioning
            while i <= len(self.elements) - step:
                x = list(self.elements[i : i + args.n_hist])

                for lt in range(args.leadtime_conditioning):
                    x_ = copy.deepcopy(x)
                    x_.append(lt)
                    # x_.append(self.include_datetime)
                    # x_.append(self.include_topography)
                    # x_.append(self.include_terrain_type)
                    # x_.append(self.include_sun_elevation_angle)
                    y = [self.elements[i + args.n_hist + lt]]

                    self._placeholder.append([x_, y])

                i += step

        assert len(self._placeholder) > 0, "Placeholder array is empty"

        print(
            "Placeholder timeseries length: {} number of samples: {} sample length: x={},y={} example sample: {}".format(
                len(self.elements),
                len(self._placeholder),
                len(self._placeholder[0][0]),
                len(self._placeholder[0][1]),
                self._placeholder[np.random.randint(len(self._placeholder))],
            )
        )

        np.random.shuffle(self._placeholder)

    def __len__(self):
        """Return number of samples"""
        return len(self._placeholder)

    def get_dataset(self, take_ratio=None, skip_ratio=None):
        placeholder = None

        if take_ratio is not None:
            l = int(len(self._placeholder) * take_ratio)
            placeholder = self._placeholder[0:l]

        if skip_ratio is not None:
            l = int(len(self._placeholder) * skip_ratio)
            placeholder = self._placeholder[l:]

        if placeholder is None:
            placeholder = copy.deepcopy(self._placeholder)

        assert len(placeholder) > 0

        gen = SAFDataGenerator(
            # placeholder=placeholder,
            leadtimes=self.leadtimes,
            batch_size=args.batch_size,
        )

        dataset = SAFDataset(gen)

        return dataset
