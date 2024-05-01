from saf import create_generators
from torch import nn
import torch
from torch.utils.data import DataLoader
from pangu_model import Pangu, Pangu_lite
from tqdm import tqdm
from configs import get_args
from streaming_tensor import StreamingTensor
from pangu_utils import split_surface_data, split_upper_air_data, split_weights
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import datetime

args = get_args()


def save_grib(output_data, times, filepath, grib_options=None):
    assert filepath[-5:] == "grib2"
    bio = BytesIO()
    times = times[0]
    analysistime = datetime.datetime.strptime(
        times[0], "%Y%m%dT%H%M%S"
    )  # utcfromtimestamp(int(times[0]) / 1e9)
    # (1, 6, 268, 238, 39)

    # T, C, H, W
    output_data = output_data[0]  # surface data only

    assert (
        len(times) == output_data.shape[0]
    ), "times ({}) do not match data ({})".format(len(times), output_data.shape[0])

    param_keys = {
        "gust": {
            "discipline": 0,
            "parameterCategory": 2,
            "parameterNumber": 22,
            "productDefinitionTemplateNumber": 8,
            "typeOfStatisticalProcessing": 2,
            "typeOfFirstFixedSurface": 103,
            "level": 10,
            "lengthOfTimeRange": 1,
        },
        "mld": {
            "discipline": 0,
            "parameterCategory": 3,
            "parameterNumber": 18,
            "typeOfFirstFixedSurface": 103,
            "level": 0,
        },
        "pres": {
            "discipline": 0,
            "parameterCategory": 3,
            "parameterNumber": 0,
            "typeOfFirstFixedSurface": 103,
            "level": 0,
        },
        "r": {
            "discipline": 0,
            "parameterCategory": 1,
            "parameterNumber": 1,
            "typeOfFirstFixedSurface": 100,
            "level": 1000,
        },
        "t": {
            "discipline": 0,
            "parameterCategory": 0,
            "parameterNumber": 0,
            "typeOfFirstFixedSurface": 100,
            "level": 1000,
        },
        "tcc": {
            "discipline": 0,
            "parameterCategory": 6,
            "parameterNumber": 1,
            "typeOfFirstFixedSurface": 103,
            "level": 0,
        },
        "u": {
            "discipline": 0,
            "parameterCategory": 2,
            "parameterNumber": 2,
            "typeOfFirstFixedSurface": 100,
            "level": 1000,
        },
        "v": {
            "discipline": 0,
            "parameterCategory": 2,
            "parameterNumber": 3,
            "typeOfFirstFixedSurface": 100,
            "level": 1000,
        },
        "z": {
            "discipline": 0,
            "parameterCategory": 3,
            "parameterNumber": 4,
            "typeOfFirstFixedSurface": 100,
            "level": 1000,
        },
    }

    def pk(param, override={}):
        x = param_keys[param].copy()
        if override is not None:
            for k, v in override.items():
                x[k] = v
        return x

    grib_keys = [
        pk("tcc"),
        pk("gust"),
        pk("mld"),
        pk("pres"),
        pk("pres", {"typeOfFirstFixedSurface": 101}),
        pk("r"),
        pk("r", {"level": 300}),
        pk("r", {"level": 500}),
        pk("r", {"level": 700}),
        pk("r", {"level": 850}),
        pk("r", {"level": 925}),
        pk("r", {"typeOfFirstFixedSurface": 103, "level": 2}),
        pk("t", {"typeOfFirstFixedSurface": 103, "level": 0}),
        pk("t"),
        pk("t", {"level": 300}),
        pk("t", {"level": 500}),
        pk("t", {"level": 700}),
        pk("t", {"level": 850}),
        pk("t", {"level": 925}),
        pk("t", {"typeOfFirstFixedSurface": 103, "level": 2}),
        pk("u"),
        pk("u", {"level": 300}),
        pk("u", {"level": 500}),
        pk("u", {"level": 700}),
        pk("u", {"level": 850}),
        pk("u", {"level": 925}),
        pk("u", {"typeOfFirstFixedSurface": 103, "level": 10}),
        pk("v"),
        pk("v", {"level": 300}),
        pk("v", {"level": 500}),
        pk("v", {"level": 700}),
        pk("v", {"level": 850}),
        pk("v", {"level": 925}),
        pk("v", {"typeOfFirstFixedSurface": 103, "level": 10}),
        pk("z"),
        pk("z", {"level": 300}),
        pk("z", {"level": 500}),
        pk("z", {"level": 700}),
        pk("z", {"level": 850}),
        pk("z", {"level": 925}),
    ]
    assert len(grib_keys) == output_data.shape[-1]
    for i in range(len(times)):
        forecasttime = analysistime + dt.timedelta(hours=i)

        for j in range(0, output_data.shape[-1]):
            data = output_data[0, i, :, :, j]

            if j == 0:
                data = np.clip(data, 0, 100)

            h = ecc.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            ecc.codes_set(h, "gridType", "lambert")
            ecc.codes_set(h, "shapeOfTheEarth", 6)
            ecc.codes_set(h, "Nx", data.shape[1])
            ecc.codes_set(h, "Ny", data.shape[0])
            ecc.codes_set(h, "DxInMetres", 2370000 / (data.shape[1] - 1))
            ecc.codes_set(h, "DyInMetres", 2670000 / (data.shape[0] - 1))
            ecc.codes_set(h, "jScansPositively", 1)
            ecc.codes_set(h, "latitudeOfFirstGridPointInDegrees", 50.319616)
            ecc.codes_set(h, "longitudeOfFirstGridPointInDegrees", 0.27828)
            ecc.codes_set(h, "Latin1InDegrees", 63.3)
            ecc.codes_set(h, "Latin2InDegrees", 63.3)
            ecc.codes_set(h, "LoVInDegrees", 15)
            ecc.codes_set(h, "latitudeOfSouthernPoleInDegrees", -90)
            ecc.codes_set(h, "longitudeOfSouthernPoleInDegrees", 0)
            ecc.codes_set(h, "dataDate", int(analysistime.strftime("%Y%m%d")))
            ecc.codes_set(h, "dataTime", int(analysistime.strftime("%H%M")))
            ecc.codes_set(h, "centre", 86)
            ecc.codes_set(h, "generatingProcessIdentifier", 251)
            ecc.codes_set(h, "packingType", "grid_ccsds")
            ecc.codes_set(h, "indicatorOfUnitOfTimeRange", 1)
            ecc.codes_set(h, "typeOfGeneratingProcess", 2)  # deterministic forecast
            ecc.codes_set(h, "typeOfProcessedData", 2)  # analysis and forecast products

            for k, v in grib_keys[j].items():
                ecc.codes_set(h, k, v)

            forecasthour = int((forecasttime - analysistime).total_seconds() / 3600)

            if j == 0:
                forecasthour -= 1

            ecc.codes_set(h, "forecastTime", forecasthour)

            data = np.flipud(data)
            ecc.codes_set_values(h, data.flatten())
            ecc.codes_write(h, bio)
            ecc.codes_release(h)

    if filepath[0:5] == "s3://":
        write_to_s3(filepath, bio)
    else:
        try:
            os.makedirs(os.path.dirname(filepath))
        except FileExistsError as e:
            pass
        except FileNotFoundError as e:
            pass
        with open(filepath, "wb") as fp:
            fp.write(bio.getbuffer())

    print(f"Wrote file {filepath}")


def model_forward(m, input_surface, surface_mask, input_upper_air, mean, std):
    for i in range(args.n_hist):
        # plot first channel only
        plt.imshow(np.squeeze(input_surface.cpu().numpy()[0, 0, ...]), cmap="gray_r")
        plt.savefig("images/inputs_{:02d}.png".format(i))

    outputs_surface, outputs_upper_air = [], []

    for i in tqdm(range(args.n_pred)):
        # B, C, H, W
        output_surface, output_upper_air = m(
            input_surface, surface_mask, input_upper_air
        )

        input_surface = output_surface
        input_upper_air = output_upper_air

        output_surface = output_surface.detach()  # B C H W
        output_upper_air = output_upper_air.detach().squeeze(2)  # B C Z H W

        # de-normalize
        output_surface = torch.permute(output_surface, (0, 2, 3, 1))  # B H W C
        output_surface = output_surface * std + mean
        output_surface = torch.permute(output_surface, (0, 3, 1, 2))  # B C H W

        output_surface = output_surface.cpu()
        output_upper_air = output_upper_air.cpu()
        plt.imshow(np.squeeze(output_surface[:, 0, :, :]), cmap="gray_r")
        plt.savefig("images/outputs_{:02d}.png".format(i))

        outputs_surface.append((output_surface.numpy()))
        outputs_upper_air.append((output_upper_air.numpy()))

        # x = x[:, 1:, ...]
        # x = torch.cat((x, y), dim=1)

    outputs_surface = np.stack(outputs_surface).squeeze(1)
    outputs_upper_air = np.stack(outputs_upper_air).squeeze(1)

    return outputs_surface, outputs_upper_air


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

    lsm = test_gen.get_static_features("lsm_heightAboveGround_0")
    z = test_gen.get_static_features("z_heightAboveGround_0")
    surface_mask = torch.stack([lsm, z]).to(args.device)
    surface_mask = surface_mask.unsqueeze(0).repeat(args.batch_size, 1, 1, 1)

    model.eval()

    outputs = []
    times = []
    with torch.no_grad():
        for i, (inputs, test_times) in enumerate(tqdm(test_gen)):
            input_surface = split_surface_data(inputs).to(args.device)
            input_upper_air = split_upper_air_data(inputs).to(args.device)

            outputs_surface, outputs_upper_air = model_forward(
                m,
                input_surface,
                surface_mask,
                input_upper_air,
                surface_mean,
                surface_std,
            )

            outputs_surface = np.concatenate(
                (input_surface.cpu(), outputs_surface), axis=0
            )
            outputs_upper_air = np.concatenate(
                (input_upper_air.cpu(), outputs_upper_air), axis=0
            )

            # outputs.append((outputs_surface, outputs_upper_air))
            outputs = [outputs_surface, outputs_upper_air]
            times.append(gen_timelist(test_times[-1]))
            assert len(times[-1]) == args.n_pred + 1

            # only single prediction from data
            break

    # outputs = np.asarray(outputs)
    print(outputs[0].shape, outputs[1].shape)
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
    save_grib(predictions, times, "prediction.grib2")
