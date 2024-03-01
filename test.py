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

args = get_args()


class StreamingTensor:
    def __init__(self, file, n_hist):
        print("Loading input file {}".format(file))
        self.ds = xr.open_zarr(file)
        #        test_file = np.load(file)
        #        self.data = test_file["arr_0"]
        #        self.times = test_file["arr_1"]
        self.n_hist = n_hist
        self.i = 0

    def __next__(self):
        indexes = np.arange(self.i, self.i + self.n_hist)
        print(self.ds)
        test_data = self.ds["effective_cloudiness"][indexes, :, :].values
        test_times = self.ds["time"][indexes].values

        # tensorflow dimension order
        test_data = np.expand_dims(test_data, axis=0)

        if test_data.shape[1] < args.n_hist:
            raise StopIteration

        assert test_data.shape == (
            1,
            args.n_hist,
            args.input_size[0],
            args.input_size[1],
        ), "test_data.shape = {}, should be (1, {}, {}, {})".format(
            test_data.shape, args.n_hist, args.input_size[0], args.input_size[0]
        )

        self.i += 1
        return torch.Tensor(test_data), test_times

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.ds.time) - 4 + 1


def model_forward(m, inputs, targets_len):
    outputs = []

    assert inputs.shape == (
        args.batch_size,
        args.n_hist,
        args.input_size[0],
        args.input_size[1],
    ), f"inputs.shape: {inputs.shape}, should be ({args.batch_size}, {args.n_hist}, {args.input_size[0]}, {args.input_size[1]})"

    x = inputs

    for i in range(args.n_hist):
        plt.imshow(np.squeeze(x.cpu().numpy()[0, i, ...]), cmap='gray_r')
        plt.savefig("images/inputs_{:02d}.png".format(i))

    for i in range(targets_len):
        if args.leadtime_conditioning:
            lt = np.full(
                (args.input_size, args.input_size),
                i / args.leadtime_conditioning,
            ).astype(np.float32)
            lt = np.expand_dims(lt, axis=(0, 1))
            lt = torch.Tensor(lt).to(args.device)

            x = torch.cat((inputs, lt), dim=1)

        print('X', x.shape, torch.mean(x))

        y = m(x)
        plt.imshow(np.squeeze(y.cpu().numpy()), cmap='gray_r')
        plt.savefig("images/outputs_{:02d}.png".format(i))

        print('Y',y.shape, torch.mean(y))
        if len(y.shape) == 3:
            y = y.unsqueeze(1)

        outputs.append(y)

        x = x[:, 1:, ...]
        x = torch.cat((x, y), dim=1)

    outputs = torch.stack(outputs)
    outputs = torch.permute(outputs, (2, 1, 0, 3, 4))
    outputs = torch.squeeze(outputs, dim=0)

    assert outputs.shape == (
        args.batch_size,
        targets_len,
        args.input_size[0],
        args.input_size[1],
    ), f"outputs.shape: {outputs.shape}, should be ({args.batch_size}, {targets_len}, {args.input_size[0]}, {args.input_size[1]})"

    return outputs


def gen_timelist(base_time, n_pred, step=datetime.timedelta(minutes=15)):
    """Generate a list of times for the predictions"""
    base_time = datetime.datetime.strptime(base_time, "%Y%m%dT%H%M%S")
    return [(base_time + x * step).strftime("%Y%m%dT%H%M%S") for x in range(n_pred + 1)]


def test(m, test_gen, n_pred):
    m.eval()
    outputs = []
    times = []
    with torch.no_grad():
        print("Running model ", end="")
        n = len(test_gen)
        for i, (inputs, test_times) in enumerate(test_gen):
            # extract "analysis data" from input
            analysis = np.expand_dims(inputs[:, args.n_hist - 1, ...], axis=0)
            inputs = inputs.float().to(args.device)
            output = model_forward(m, inputs, n_pred).cpu().numpy()
            output = np.concatenate((analysis, output), axis=1)
            outputs.append(output)
            times.append(gen_timelist(test_times[-1], n_pred))
            assert len(times[-1]) == n_pred + 1

            if i % (n // 10) == 0:
                print("{:.0f}%".format(100 * i / n), end="", flush=True)

            elif (i + 1) % (math.ceil(n / 100.0)) == 0:
                print(".", end="", flush=True)

            # only single prediction from data
            break
    print("")

    outputs = np.squeeze(np.asarray(outputs), axis=1)

    return outputs, times


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

    test_gen = StreamingTensor(args.dataseries_file, args.n_hist)
    predictions, times = test(m, test_gen, args.n_pred)

#    filename = os.path.realpath(args.model_dir).split("/")[-1] + ".npz"
#    with open(filename, "wb") as fp:
#        np.savez(fp, predictions, times)
#        print(f"Wrote {len(predictions)} predictions to file '{filename}'")
