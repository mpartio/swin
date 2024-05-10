import numpy as np
import datetime
import torch
import eccodes as ecc
import os
import einops
from saf import create_generators
from torch import nn
from torch.utils.data import DataLoader
from pangu_model import Pangu, Pangu_lite
from tqdm import tqdm
from configs import get_args
from streaming_tensor import StreamingTensor
from pangu_utils import split_surface_data, split_upper_air_data, split_weights
from io import BytesIO

args = get_args()


def save_grib(output_data, times, filepath, grib_options=None):
    assert filepath[-5:] == "grib2"
    bio = BytesIO()
    analysistime = times[0]

    # T, C, H, W
    output_data = output_data[0]  # surface data only

    assert (
        len(times) == output_data.shape[0]
    ), "times ({}) do not match data ({})".format(len(times), output_data.shape[0])

    params = args.parameters
    param_keys = {
        "fgcorr_heightAboveGround_10": {
            "discipline": 0,
            "parameterCategory": 2,
            "parameterNumber": 22,
            "productDefinitionTemplateNumber": 8,
            "typeOfStatisticalProcessing": 2,
            "typeOfFirstFixedSurface": 103,
            "level": 10,
            "lengthOfTimeRange": 1,
        },
        "mld_heightAboveGround_0": {
            "discipline": 0,
            "parameterCategory": 3,
            "parameterNumber": 18,
            "typeOfFirstFixedSurface": 103,
            "level": 0,
        },
        "pres_heightAboveSea_0": {
            "discipline": 0,
            "parameterCategory": 3,
            "parameterNumber": 0,
            "typeOfFirstFixedSurface": 101,
            "level": 0,
        },
        "pres_heightAboveGround_0": {
            "discipline": 0,
            "parameterCategory": 3,
            "parameterNumber": 0,
            "typeOfFirstFixedSurface": 103,
            "level": 0,
        },
        "r_heightAboveGround_2": {
            "discipline": 0,
            "parameterCategory": 1,
            "parameterNumber": 1,
            "typeOfFirstFixedSurface": 103,
            "level": 2,
        },
        "rcorr_heightAboveGround_2": {
            "discipline": 0,
            "parameterCategory": 1,
            "parameterNumber": 1,
            "typeOfFirstFixedSurface": 103,
            "level": 2,
        },
        "t_heightAboveGround_0": {
            "discipline": 0,
            "parameterCategory": 0,
            "parameterNumber": 0,
            "typeOfFirstFixedSurface": 103,
            "level": 0,
        },
        "t_heightAboveGround_2": {
            "discipline": 0,
            "parameterCategory": 0,
            "parameterNumber": 0,
            "typeOfFirstFixedSurface": 103,
            "level": 2,
        },
        "tcorr_heightAboveGround_2": {
            "discipline": 0,
            "parameterCategory": 0,
            "parameterNumber": 0,
            "typeOfFirstFixedSurface": 103,
            "level": 2,
        },
        "effective-cloudiness_heightAboveGround_0": {
            "discipline": 0,
            "parameterCategory": 6,
            "parameterNumber": 1,
            "typeOfFirstFixedSurface": 103,
            "level": 0,
        },
        "u_isobaricInhPa": {
            "discipline": 0,
            "parameterCategory": 2,
            "parameterNumber": 2,
            "typeOfFirstFixedSurface": 100,
        },
        "v_isobaricInhPa": {
            "discipline": 0,
            "parameterCategory": 2,
            "parameterNumber": 2,
            "typeOfFirstFixedSurface": 100,
        },
        "ucorr_heightAboveGround_10": {
            "discipline": 0,
            "parameterCategory": 2,
            "parameterNumber": 2,
            "typeOfFirstFixedSurface": 103,
            "level": 10,
        },
        "vcorr_heightAboveGround_10": {
            "discipline": 0,
            "parameterCategory": 2,
            "parameterNumber": 3,
            "typeOfFirstFixedSurface": 103,
            "level": 10,
        },
        "t_isobaricInhPa": {
            "discipline": 0,
            "parameterCategory": 0,
            "parameterNumber": 0,
            "typeOfFirstFixedSurface": 100,
        },
        "r_isobaricInhPa": {
            "discipline": 0,
            "parameterCategory": 1,
            "parameterNumber": 1,
            "typeOfFirstFixedSurface": 100,
        },
        "z_isobaricInhPa": {
            "discipline": 0,
            "parameterCategory": 3,
            "parameterNumber": 4,
            "typeOfFirstFixedSurface": 100,
        },
    }

    for i in range(len(times)):
        forecasttime = analysistime + datetime.timedelta(hours=i)

        for j in range(0, output_data.shape[1]):
            data = output_data[i, j, :, :]

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

            param = params[j]

            if "isobaricInhPa" in param:
                ecc.codes_set(h, "level", int(param.split("_")[-1]))

                param = "_".join(param.split("_")[:-1])
            for k, v in param_keys[param].items():
                ecc.codes_set(h, k, v)

            forecasthour = int((forecasttime - analysistime).total_seconds() / 3600)

            if param.startswith("fgcorr"):
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

    outputs_surface, outputs_upper_air = [], []

    for i in tqdm(range(args.n_pred)):
        output_surface, output_upper_air = m(
            input_surface, surface_mask, input_upper_air
        )

        input_surface = output_surface
        input_upper_air = output_upper_air

        # import matplotlib.pyplot as plt

        # plt.imshow(np.squeeze(output_surface[:, 0, :, :]), cmap="gray_r")
        # plt.show()
        # plt.savefig("images/outputs_{:02d}.png".format(i))

        outputs_surface.append(output_surface)
        outputs_upper_air.append(output_upper_air)

        input_surface = output_surface
        input_upper_air = output_upper_air

    outputs_surface = torch.stack(outputs_surface).squeeze(1)
    outputs_upper_air = torch.stack(outputs_upper_air).squeeze(1)

    return outputs_surface, outputs_upper_air


def gen_timelist(base_time, step=datetime.timedelta(minutes=60)):
    """Generate a list of times for the predictions"""
    base_time = datetime.datetime.fromtimestamp(
        base_time.astype("datetime64[s]").astype("int")
    )
    return [
        (base_time + x * step).strftime("%Y%m%dT%H%M%S") for x in range(args.n_pred + 1)
    ]


def restore_surface_data(data, mean, std):
    """Restore the data from normalized form"""
    _, _, ny, nx = data.shape

    data = einops.rearrange(data, "b c h w -> b (h w) c", h=ny, w=nx)
    data = data * std + mean
    data = einops.rearrange(data, "b (h w) c -> b c h w", h=ny, w=nx)

    return data


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

    surface_prediction = []
    upper_air_prediction = []
    times = []
    with torch.no_grad():
        for i, (inputs, test_times) in enumerate(test_gen):
            input_surface = split_surface_data(inputs).to(args.device)
            input_upper_air = split_upper_air_data(inputs).to(args.device)

            # surface_prediction.append(restore_surface_data(input_surface, surface_mean, surface_std))

            outputs_surface, outputs_upper_air = model_forward(
                m,
                input_surface,
                surface_mask,
                input_upper_air,
                surface_mean,
                surface_std,
            )

            outputs_surface = torch.cat((input_surface, outputs_surface), dim=0)
            outputs_surface = (
                restore_surface_data(outputs_surface, surface_mean, surface_std)
                .detach()
                .cpu()
                .numpy()
            )
            surface_prediction.append(outputs_surface)

            outputs = [outputs_surface, outputs_upper_air]
            times.append(gen_timelist(test_times[-1]))
            assert len(times[-1]) == args.n_pred + 1

            break
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

    truth_times = test_gen.ds.time.values
    truth_times = [
        datetime.datetime.utcfromtimestamp(int(x) / 1e9) for x in truth_times
    ]
    truth_times = truth_times[1:]

    truth = test_gen.ds[args.parameters].to_array().values  # (c, t, h, w)
    truth = einops.rearrange(truth, "c t h w -> t c h w")
    truth = truth[
        1:, ...
    ]  # remove first time step as it is the "prev prev" initial state with neural-lam
    # truth = restore_surface_data(torch.Tensor(truth), test_gen.mean, test_gen.std)
    truth = np.expand_dims(truth, 0)  # add batch dimension to match predictions

    save_grib(truth, truth_times, "truth.grib2")

    predictions, times = test(m, test_gen)

    times = [datetime.datetime.strptime(x, "%Y%m%dT%H%M%S") for x in times[0]]
    save_grib(predictions, times, "prediction.grib2")
