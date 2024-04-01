import xarray as xr
import zarr
import torch
import numpy as np


class StreamingTensor:
    def __init__(self, file, parameters, model_dir):
        print("Loading input file {}".format(file))
        self.ds = xr.open_zarr(file)
        self.i = 0

        self.mean = torch.load(f"{model_dir}/parameter_mean.pt")  # .numpy()
        self.std = torch.load(f"{model_dir}/parameter_std.pt")  # .numpy()
        self.parameters = parameters
        self.n_hist = 1

        assert len(self.mean) == len(
            self.parameters
        ), "Mean and parameters length mismatch: were means calculated from another dataset? Remove parameter_mean.pt and parameter_std.pt and try again"

    def __next__(self):
        indexes = np.arange(self.i, self.i + self.n_hist)

        # Note: times do not have a separate dimension
        test_data = (
            self.ds.isel(time=indexes)[self.parameters].to_array().values
        )  # (C, B, H, W)
        test_data = torch.Tensor(test_data)
        test_times = self.ds["time"][indexes].values

        # move channels to end so that data can be normalized

        test_data = test_data.permute(1, 2, 3, 0)  # (B, H, W, C)
        test_data = (test_data - self.mean) / self.std

        # return back to normal order
        test_data = test_data.permute(0, 3, 1, 2)  # (B, C, H, W)

        if test_data.shape[0] < self.n_hist:
            raise StopIteration

        assert test_data.shape[1] == len(self.parameters)

        self.i += 1
        return test_data, test_times
        # return torch.Tensor(test_data), test_times

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.ds.time)

    def get_static_features(self, parameter):
        assert parameter in ("lsm_heightAboveGround_0", "z_heightAboveGround_0")
        return torch.tensor(self.ds[parameter].values)
