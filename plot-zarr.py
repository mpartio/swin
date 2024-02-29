from matplotlib import pyplot as plt
import sys
import xarray as xr
import zarr
import numpy as np
from tqdm import tqdm

def draw(datas):

    for i,x in enumerate(tqdm(datas)):
        plt.figure(i)
        plt.imshow(x, cmap='gray')
        plt.savefig(f'fig{i:03d}.png')
        plt.close()



def read(filename):
    ds = xr.open_zarr(filename)

    return ds.effective_cloudiness.to_numpy()

def plot():
    datas = read(sys.argv[1])
    draw(datas)

if len(sys.argv) == 1:
    print('Usage: python plot-zarr.py file.zarr')
    sys.exit(1)

plot()
