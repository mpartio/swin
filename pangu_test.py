from saf import create_generators
from torch import nn
import torch
from torch.utils.data import DataLoader
from pangu_model import Pangu, Pangu_lite
from tqdm import tqdm
from configs import get_args
from streaming_tensor import StreamingTensor
from pangu_utils import split_surface_data, split_upper_air_data, split_weights
import matplotlib.pyplot as plt
import numpy as np
import datetime

args = get_args()


def model_forward(m, input_surface, surface_mask, input_upper_air, mean, std):
    for i in range(args.n_hist):
        # plot first channel only
        plt.imshow(np.squeeze(input_surface.cpu().numpy()[0, 0, ...]), cmap="gray_r")
        plt.savefig("images/inputs_{:02d}.png".format(i))

    outputs = []
    for i in tqdm(range(args.n_pred)):
        # B, C, H, W
        output_surface, output_upper_air = m(
            input_surface, surface_mask, input_upper_air
        )
        output_surface = output_surface.detach().cpu()  # B C H W
        output_upper_air = output_upper_air.detach().cpu().squeeze(2)  # B C Z H W
        # y = torch.cat((output_surface, output_upper_air), dim=3)

        plt.imshow(np.squeeze(output_surface[:, 0, :, :]), cmap="gray_r")
        plt.savefig("images/outputs_{:02d}.png".format(i))

        # if len(y.shape) == 3:
        #    y = y.unsqueeze(1)

        # de-normalize
        output_surface = torch.permute(output_surface, (0, 2, 3, 1))  # B H W C
        output_surface = output_surface * std + mean
        output_surface = torch.permute(output_surface, (0, 3, 1, 2))  # B C H W
        outputs.append(output_surface.numpy())

        # x = x[:, 1:, ...]
        # x = torch.cat((x, y), dim=1)

    outputs = np.stack(outputs)  # .squeeze(0)
    assert len(outputs.shape) == 4

    return outputs


def gen_timelist(base_time, step=datetime.timedelta(minutes=60)):
    """Generate a list of times for the predictions"""
    base_time = datetime.datetime.fromtimestamp(
        base_time.astype("datetime64[s]").astype("int")
    )
    return [
        (base_time + x * step).strftime("%Y%m%dT%H%M%S") for x in range(args.n_pred + 1)
    ]


def test(model, test_gen):
    """Training code"""
    # Prepare for the optimizer and scheduler

    #    parameter_mean = torch.load("parameter_mean.pt")
    #    parameter_std = torch.load("parameter_std.pt")

    surface_mean, upper_air_mean = split_weights(test_gen.mean)
    surface_std, upper_air_std = split_weights(test_gen.std)

    # surface_mean = parameter_mean[0].to(args.device)
    # upper_mean = parameter_mean[1:].to(args.device)
    # surface_std = parameter_std[0].to(args.device)
    # upper_std = parameter_std[1:].to(args.device)

    lsm = test_gen.get_static_features("lsm_heightAboveGround_0")
    z = test_gen.get_static_features("z_heightAboveGround_0")
    surface_mask = torch.stack([lsm, z]).to(args.device)
    surface_mask = surface_mask.unsqueeze(0).repeat(args.batch_size, 1, 1, 1)

    model.eval()

    outputs = []
    times = []
    with torch.no_grad():
        for i, (inputs, test_times) in enumerate(tqdm(test_gen)):
            input_surface = split_surface_data(inputs)
            input_upper_air = split_upper_air_data(inputs)

            output = model_forward(
                m,
                input_surface,
                surface_mask,
                input_upper_air,
                surface_mean,
                surface_std,
            )

            # output = np.concatenate((analysis, output), axis=1)
            outputs.append(output)
            times.append(gen_timelist(test_times[-1]))
            assert len(times[-1]) == args.n_pred + 1

            # only single prediction from data
            break

    outputs = np.squeeze(np.asarray(outputs), axis=1)

    return outputs, times


if __name__ == "__main__":
    if args.model_name == "pangu_lite":
        m = Pangu_lite().to(args.device)
    elif args.model_name == "pangu":
        m = Pangu().to(args.device)
    else:
        raise ValueError("Unknown model name: {}".format(args.model_name))

    m.load_state_dict(
        torch.load(
            "{}/trained_model_state_dict".format(args.model_dir),
            map_location=torch.device(args.device),
        )
    )

    test_gen = StreamingTensor(args.dataseries_file, args.parameters, args.model_dir)
    predictions, times = test(m, test_gen)
