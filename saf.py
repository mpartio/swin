import numpy as np
import glob
from datetime import datetime, timedelta
from enum import Enum
import copy
import torch
from configs import get_args

from torch.utils.data import IterableDataset

args = get_args()


def read_times_from_preformatted_files_directory(dirname):
    toc = {}
    for f in glob.glob("{}/*-times.npy".format(dirname)):
        times = np.load(f)

        for i, t in enumerate(times):
            toc[t] = {"filename": f.replace("-times", ""), "index": i, "time": t}

    times = list(toc.keys())

    times.sort()
    print("Read {} times from {}".format(len(times), dirname))
    return times, toc


def read_datas_from_preformatted_files_directory(dirname, toc, times):
    datas = []
    print("Reading data for {}".format(times))
    for t in times:
        e = toc[t]
        idx = e["index"]
        filename = e["filename"]
        datafile = np.load(filename, mmap_mode="r")
        datas.append(datafile[idx])

    return datas, times


def read_times_from_preformatted_file(filename):
    ds = np.load(filename)
    data = ds["arr_0"]
    times = ds["arr_1"]

    toc = {}
    for i, t in enumerate(times):
        toc[t] = {"index": i, "time": t}

    print("Read {} times from {}".format(len(times), filename))

    return times, data, toc


def read_datas_from_preformatted_file(all_times, all_data, req_times, toc):
    datas = []
    for t in req_times:
        index = toc[t]["index"]
        datas.append(all_data[index])

    return datas, req_times


class SAFDataGenerator:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        assert len(self.placeholder) > 0
        print(
            "Generator number of samples: {} number of batches: {} batch size: {}".format(
                len(self), len(self) // self.batch_size, self.batch_size
            )
        )

    def __len__(self):
        # """Return number of batches in this dataset"""
        # return len(self.placeholder) // self.batch_size
        """Return number of samples in this dataset"""
        return len(self.placeholder)

    def __getitem__(self, idx):
        # placeholder X elements:
        # 0.. args.n_hist: history of actual data (YYYYMMDDTHHMMSS, string)
        # args.n_hist + 1: include datetime (bool)
        # args.n_hist + 2: include topography (bool)
        # args.n_hist + 3: include terrain type (bool)
        # args.n_hist + 4: include sun elevation angle (bool)

        ph = self.placeholder[idx]

        X = ph[0]
        Y = ph[1]

        x_hist = X[0 : args.n_hist]

        x, y, xtimes, ytimes = self.get_xy(x_hist, Y)

        x = np.asarray(x)
        y = np.asarray(y)

        if args.leadtime_conditioning:
            lt = X[args.n_hist]
            lt = np.expand_dims(self.leadtimes[lt], axis=0)
            lt = np.expand_dims(lt, axis=-1)
            x = np.concatenate((x, lt), axis=0)

        assert np.max(x) < 1.01, "x max: {:.2f}".format(np.max(x))

        x = np.squeeze(x)
        y = np.squeeze(y, axis=-1)

        if args.n_pred > 1:
            y = np.squeeze(y)

        assert x.shape == (
            args.n_hist + int(bool(args.leadtime_conditioning)),
            args.input_size,
            args.input_size,
        ), "x shape is {}, should be ({}, {}, {}))".format(
            x.shape,
            args.n_hist + int(bool(args.leadtime_conditioning)),
            args.input_size,
            args.input_size,
        )
        assert y.shape == (
            args.n_pred,
            args.input_size,
            args.input_size,
        ), f"y shape is {y.shape}, should be ({args.n_pred}, {args.input_size}, {args.input_size})"

        x = torch.from_numpy(x).contiguous()
        y = torch.from_numpy(y).contiguous()

        return (x, y)

    def get_xy(self, x_elems, y_elems):
        xtimes = []
        ytimes = []

        if self.dataseries_file is not None:
            x, xtimes = read_datas_from_preformatted_file(
                self.elements, self.data, x_elems, self.toc
            )
            y, ytimes = read_datas_from_preformatted_file(
                self.elements, self.data, y_elems, self.toc
            )

        return x, y, xtimes, ytimes

    def __call__(self):
        for i in range(len(self.placeholder)):
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

        if self.dataseries_file is not None:
            self.elements, self.data, self.toc = read_times_from_preformatted_file(
                self.dataseries_file
            )

        elif self.dataseries_directory is not None:
            self.elements, self.toc = read_times_from_preformatted_files_directory(
                self.dataseries_directory
            )

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
            placeholder=placeholder,
            leadtimes=self.leadtimes,
            batch_size=args.batch_size,
            img_size=self.img_size,
            dataseries_file=self.dataseries_file,
            elements=self.elements,
            data=self.data,
            toc=self.toc,
        )

        dataset = SAFDataset(gen)

        return dataset
