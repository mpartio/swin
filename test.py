import time
import os
import torch
import math
import datetime
from torch import nn
from swin import create_model
from configs import get_args
from utils import set_seed
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import zarr
from tqdm import tqdm

args = get_args()


class StreamingTensor:
    def __init__(self, file):
        print("Loading input file {}".format(file))
        self.ds = xr.open_zarr(file)
        self.i = 0
        self.mean = torch.load("parameter_mean.pt")
        self.std = torch.load("parameter_std.pt")

    def __next__(self):
        indexes = np.arange(self.i, self.i + args.n_hist)

        # Note: times do not have a separate dimension
        # C, B, H, W
        test_data = self.ds.isel(time=indexes)[args.parameters].to_array().values
        test_times = self.ds["time"][indexes].values

        # move channels to end so that data can be normalized
        # B, H, W, C
        test_data = np.moveaxis(test_data, 0, -1)
        test_data = (test_data - self.mean) / self.std

        # set dimension to correct order:
        # B, C, H, W
        test_data = np.moveaxis(test_data, -1, 1)

        if test_data.shape[0] < args.n_hist:
            raise StopIteration

        assert test_data.shape == (
            1,
            (args.n_hist * len(args.parameters)),
            args.input_size[0],
            args.input_size[1],
        ), "test_data.shape = {}, should be (1, {}, {}, {})".format(
            test_data.shape,
            (args.n_hist * len(args.parameters)),
            args.input_size[0],
            args.input_size[0],
        )

        self.i += 1
        return torch.Tensor(test_data), test_times

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.ds.time) - 4 + 1


def model_forward(m, inputs, mean, std):
    outputs = []

    mean = torch.Tensor(mean).to(args.device)
    std = torch.Tensor(std).to(args.device)

    assert inputs.shape == (
        1,
        args.n_hist * len(args.parameters),
        args.input_size[0],
        args.input_size[1],
    ), f"inputs.shape: {inputs.shape}, should be (1, {args.n_hist * len(args.parameters)}, {args.input_size[0]}, {args.input_size[1]})"

    x = inputs

    for i in range(args.n_hist):
        # plot first channel only
        plt.imshow(np.squeeze(x.cpu().numpy()[0, 0, ...]), cmap="gray_r")
        plt.savefig("images/inputs_{:02d}.png".format(i))

    for i in tqdm(range(args.n_pred)):
        # B, C, H, W
        y = m(x)
        yc = y.detach().cpu().numpy()
        plt.imshow(np.squeeze(yc[:, 0, :, :]), cmap="gray_r")
        plt.savefig("images/outputs_{:02d}.png".format(i))

        if len(y.shape) == 3:
            y = y.unsqueeze(1)

        x = y

        # de-normalize
        # B, H, W, C
        y = torch.permute(y, (0, 2, 3, 1))
        y = y * std + mean
        # B, C, H, W
        y = torch.permute(y, (0, 3, 1, 2))
        outputs.append(y)

        # x = x[:, 1:, ...]
        # x = torch.cat((x, y), dim=1)

    # T, B, C, H, W
    outputs = torch.stack(outputs).squeeze(0)
    assert len(outputs.shape) == 5

    # B, T, C, H, W
    outputs = torch.permute(outputs, (1, 0, 2, 3, 4))
    #    outputs = torch.squeeze(outputs, dim=0)

    assert outputs.shape == (
        args.batch_size,
        args.n_pred,
        len(args.parameters),
        args.input_size[0],
        args.input_size[1],
    ), f"outputs.shape: {outputs.shape}, should be ({args.batch_size}, {args.n_pred}, {len(args.parameters)}, {args.input_size[0]}, {args.input_size[1]})"

    return outputs


def gen_timelist(base_time, step=datetime.timedelta(minutes=15)):
    """Generate a list of times for the predictions"""
    #    base_time = datetime.datetime.strptime(base_time, "%Y%m%dT%H%M%S")
    base_time = datetime.datetime.fromtimestamp(
        base_time.astype("datetime64[s]").astype("int")
    )
    return [
        (base_time + x * step).strftime("%Y%m%dT%H%M%S") for x in range(args.n_pred + 1)
    ]


def test(m, test_gen):
    m.eval()
    outputs = []
    times = []
    with torch.no_grad():
        print("Running model")
        n = len(test_gen)
        for i, (inputs, test_times) in enumerate(tqdm(test_gen)):
            inputs = inputs.float().to(args.device)
            output = (
                model_forward(m, inputs, test_gen.mean, test_gen.std)
                .detach()
                .cpu()
                .numpy()
            )

            # output = np.concatenate((analysis, output), axis=1)
            outputs.append(output)
            times.append(gen_timelist(test_times[-1]))
            assert len(times[-1]) == args.n_pred + 1

            # only single prediction from data
            break

    outputs = np.squeeze(np.asarray(outputs), axis=1)

    return outputs, times


def create_xarray_dataset(data, dates):
    if len(dates) == 0:
        return None
    ds = xr.Dataset(
        coords={"x": x, "y": y, "time": dates},
    )

    for i, param in enumerate(args.params + ["effective_cloudiness"]):
        ds[param] = (["time", "y", "x"], data[:, i, :, :])

    ds = ds.chunk(
        {
            "time": 1,
            "y": interp_points_shape[0],
            "x": interp_points_shape[1],
        }
    )
    ds.rio.write_crs(create_spatial_ref(), inplace=True)

    return ds


if __name__ == "__main__":
    set_seed()

    with open("{}/model-name.txt".format(args.model_dir), "r") as fp:
        model_name = fp.readline().strip()

    m = create_model(model_name).to(args.device)
    m.load_state_dict(
        torch.load(
            "{}/trained_model_state_dict".format(args.model_dir),
            map_location=torch.device(args.device),
        )
    )

    test_gen = StreamingTensor(args.dataseries_file)
    predictions, times = test(m, test_gen)

    print(predictions.shape)
    effc = predictions[:, :, 0, ...]

    print(np.min(effc, axis=2), np.mean(effc), np.max(effc))
    #filename = "out.zarr"
    #ds = create_xarray_dataset(predictions, times)
    #ds.to_zarr(filename)
