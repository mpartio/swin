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

args = get_args()


class StreamingTensor:
    def __init__(self, file, n_hist):
        print("Loading input file {}".format(file))
        test_file = np.load(file)
        self.data = test_file["arr_0"]
        self.times = test_file["arr_1"]
        self.n_hist = n_hist
        self.i = 0

    def __next__(self):
        test_data = self.data[self.i : self.i + args.n_hist, ...]
        test_times = self.times[self.i : self.i + args.n_hist, ...]
        # tensorflow dimension order
        test_data = np.expand_dims(np.squeeze(test_data, axis=-1), axis=0)

        if test_data.shape[1] < args.n_hist:
            raise StopIteration

        assert test_data.shape == (
            1,
            args.n_hist,
            args.input_size,
            args.input_size,
        ), "test_data.shape = {}, should be (1, {}, {}, {})".format(
            test_data.shape, args.n_hist, args.input_size, args.input_size
        )

        self.i += 1
        return torch.Tensor(test_data), test_times

    def __iter__(self):
        return self

    def __len__(self):
        return self.data.shape[0] - 4 + 1


def model_forward(m, inputs, targets_len):
    outputs = []

    assert inputs.shape == (
        args.batch_size,
        args.n_hist,
        args.input_size,
        args.input_size,
    ), f"inputs.shape: {inputs.shape}, should be ({args.batch_size}, {args.n_hist}, {args.input_size}, {args.input_size})"

    for i in range(targets_len):
        if args.leadtime_conditioning:
            lt = np.full(
                (args.input_size, args.input_size),
                i / args.leadtime_conditioning,
            ).astype(np.float32)
            lt = np.expand_dims(lt, axis=(0, 1))
            lt = torch.Tensor(lt).to(args.device)

        x = torch.cat((inputs, lt), dim=1)
        y = m(x)

        if len(y.shape) == 3:
            y = y.unsqueeze(1)

        outputs.append(y)

    outputs = torch.stack(outputs)
    outputs = torch.permute(outputs, (2, 1, 0, 3, 4))
    outputs = torch.squeeze(outputs, dim=0)

    assert outputs.shape == (
        args.batch_size,
        targets_len,
        args.input_size,
        args.input_size,
    ), f"outputs.shape: {outputs.shape}, should be ({args.batch_size}, {targets_len}, {args.input_size}, {args.input_size})"

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

    filename = os.path.realpath(args.model_dir).split("/")[-1] + ".npz"
    with open(filename, "wb") as fp:
        np.savez(fp, predictions, times)
        print(f"Wrote {len(predictions)} predictions to file '{filename}'")
