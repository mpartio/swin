import torch
from torch import nn
import numpy as np
from timm.models.layers import trunc_normal_, DropPath
from configs import get_args

args = get_args()


def swap_tuple(tpl):
    return tuple(reversed(tpl))


def crop2d(x: torch.Tensor, resolution):
    """
    Args:
        x (torch.Tensor): B, C, Lat, Lon
        resolution (tuple[int]): Lat, Lon
    """
    _, _, Lat, Lon = x.shape
    lat_pad = Lat - resolution[0]
    lon_pad = Lon - resolution[1]

    padding_top = lat_pad // 2
    padding_bottom = lat_pad - padding_top

    padding_left = lon_pad // 2
    padding_right = lon_pad - padding_left

    return x[
        :, :, padding_top : Lat - padding_bottom, padding_left : Lon - padding_right
    ]


def crop3d(x: torch.Tensor, resolution):
    """
    Args:
        x (torch.Tensor): B, C, Pl, Lat, Lon
        resolution (tuple[int]): Pl, Lat, Lon
    """
    _, _, Pl, Lat, Lon = x.shape
    pl_pad = Pl - resolution[0]
    lat_pad = Lat - resolution[1]
    lon_pad = Lon - resolution[2]

    padding_front = pl_pad // 2
    padding_back = pl_pad - padding_front

    padding_top = lat_pad // 2
    padding_bottom = lat_pad - padding_top

    padding_left = lon_pad // 2
    padding_right = lon_pad - padding_left
    return x[
        :,
        :,
        padding_front : Pl - padding_back,
        padding_top : Lat - padding_bottom,
        padding_left : Lon - padding_right,
    ]


def get_pad3d(input_resolution, window_size):
    """
    Args:
        input_resolution (tuple[int]): (Pl, Lat, Lon)
        window_size (tuple[int]): (Pl, Lat, Lon)

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
    """
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size

    padding_left = (
        padding_right
    ) = padding_top = padding_bottom = padding_front = padding_back = 0
    pl_remainder = Pl % win_pl
    lat_remainder = Lat % win_lat
    lon_remainder = Lon % win_lon

    if pl_remainder:
        pl_pad = win_pl - pl_remainder
        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front
    if lat_remainder:
        lat_pad = win_lat - lat_remainder
        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top
    if lon_remainder:
        lon_pad = win_lon - lon_remainder
        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

    return (
        padding_left,
        padding_right,
        padding_top,
        padding_bottom,
        padding_front,
        padding_back,
    )


def get_pad2d(input_resolution, window_size):
    """
    Args:
        input_resolution (tuple[int]): Lat, Lon
        window_size (tuple[int]): Lat, Lon

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom)
    """
    input_resolution = [2] + list(input_resolution)
    window_size = [2] + list(window_size)
    padding = get_pad3d(input_resolution, window_size)
    return padding[:4]


class PatchRecovery2D(nn.Module):
    """
    Patch Embedding Recovery to 2D Image.

    Args:
        img_size (tuple[int]): Lat, Lon
        patch_size (tuple[int]): Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x):
        output = self.conv(x)
        _, _, H, W = output.shape
        h_pad = H - self.img_size[0]
        w_pad = W - self.img_size[1]

        padding_top = h_pad // 2
        padding_bottom = int(h_pad - padding_top)

        padding_left = w_pad // 2
        padding_right = int(w_pad - padding_left)

        return output[
            :, :, padding_top : H - padding_bottom, padding_left : W - padding_right
        ]


class PatchRecovery3D(nn.Module):
    """
    Patch Embedding Recovery to 3D Image.

    Args:
        img_size (tuple[int]): Pl, Lat, Lon
        patch_size (tuple[int]): Pl, Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose3d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x: torch.Tensor):
        output = self.conv(x)
        _, _, Pl, Lat, Lon = output.shape

        pl_pad = Pl - self.img_size[0]
        lat_pad = Lat - self.img_size[1]
        lon_pad = Lon - self.img_size[2]

        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front

        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top

        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

        return output[
            :,
            :,
            padding_front : Pl - padding_back,
            padding_top : Lat - padding_bottom,
            padding_left : Lon - padding_right,
        ]


class PatchEmbed2D(nn.Module):
    """
    2D Image to Patch Embedding.

    Args:
        img_size (tuple[int]): Image size.
        patch_size (tuple[int]): Patch token size.
        in_chans (int): Number of input image channels.
        embed_dim(int): Number of projection output channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        height, width = img_size
        h_patch_size, w_path_size = patch_size
        padding_left = padding_right = padding_top = padding_bottom = 0

        h_remainder = height % h_patch_size
        w_remainder = width % w_path_size

        if h_remainder:
            h_pad = h_patch_size - h_remainder
            padding_top = h_pad // 2
            padding_bottom = int(h_pad - padding_top)

        if w_remainder:
            w_pad = w_path_size - w_remainder
            padding_left = w_pad // 2
            padding_right = int(w_pad - padding_left)

        self.pad = nn.ZeroPad2d(
            (padding_left, padding_right, padding_top, padding_bottom)
        )
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.pad(x)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class PatchEmbed3D(nn.Module):
    """
    3D Image to Patch Embedding.

    Args:
        img_size (tuple[int]): Image size.
        patch_size (tuple[int]): Patch token size.
        in_chans (int): Number of input image channels.
        embed_dim(int): Number of projection output channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        level, height, width = img_size
        l_patch_size, h_patch_size, w_patch_size = patch_size
        padding_left = (
            padding_right
        ) = padding_top = padding_bottom = padding_front = padding_back = 0

        l_remainder = level % l_patch_size
        h_remainder = height % l_patch_size
        w_remainder = width % w_patch_size

        if l_remainder:
            l_pad = l_patch_size - l_remainder
            padding_front = l_pad // 2
            padding_back = l_pad - padding_front
        if h_remainder:
            h_pad = h_patch_size - h_remainder
            padding_top = h_pad // 2
            padding_bottom = h_pad - padding_top
        if w_remainder:
            w_pad = w_patch_size - w_remainder
            padding_left = w_pad // 2
            padding_right = w_pad - padding_left

        self.pad = nn.ZeroPad3d(
            (
                padding_left,
                padding_right,
                padding_top,
                padding_bottom,
                padding_front,
                padding_back,
            )
        )
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, C, L, H, W = x.shape
        assert (
            L == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2]
        ), f"Input image size ({L}*{H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.pad(x)
        x = self.proj(x)
        if self.norm:
            x = self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return x


def window_partition(x: torch.Tensor, window_size):
    """
    Args:
        x: (B, Pl, Lat, Lon, C)
        window_size (tuple[int]): [win_pl, win_lat, win_lon]

    Returns:
        windows: (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C)
    """
    B, Pl, Lat, Lon, C = x.shape
    win_pl, win_lat, win_lon = window_size
    x = x.view(
        B, Pl // win_pl, win_pl, Lat // win_lat, win_lat, Lon // win_lon, win_lon, C
    )
    windows = (
        x.permute(0, 5, 1, 3, 2, 4, 6, 7)
        .contiguous()
        .view(-1, (Pl // win_pl) * (Lat // win_lat), win_pl, win_lat, win_lon, C)
    )
    return windows


def window_reverse(windows, window_size, Pl, Lat, Lon):
    """
    Args:
        windows: (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C)
        window_size (tuple[int]): [win_pl, win_lat, win_lon]

    Returns:
        x: (B, Pl, Lat, Lon, C)
    """
    win_pl, win_lat, win_lon = window_size
    B = int(windows.shape[0] / (Lon / win_lon))
    x = windows.view(
        B, Lon // win_lon, Pl // win_pl, Lat // win_lat, win_pl, win_lat, win_lon, -1
    )
    x = x.permute(0, 2, 4, 3, 5, 1, 6, 7).contiguous().view(B, Pl, Lat, Lon, -1)
    return x


def get_shift_window_mask(input_resolution, window_size, shift_size):
    """
    Along the longitude dimension, the leftmost and rightmost indices are actually close to each other.
    If half windows apper at both leftmost and rightmost positions, they are dircetly merged into one window.
    Args:
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        window_size (tuple[int]): Window size [pressure levels, latitude, longitude].
        shift_size (tuple[int]): Shift size for SW-MSA [pressure levels, latitude, longitude].

    Returns:
        attn_mask: (n_lon, n_pl*n_lat, win_pl*win_lat*win_lon, win_pl*win_lat*win_lon)
    """
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size
    shift_pl, shift_lat, shift_lon = shift_size

    img_mask = torch.zeros((1, Pl, Lat, Lon + shift_lon, 1))

    pl_slices = (slice(0, -win_pl), slice(-win_pl, -shift_pl), slice(-shift_pl, None))
    lat_slices = (
        slice(0, -win_lat),
        slice(-win_lat, -shift_lat),
        slice(-shift_lat, None),
    )
    lon_slices = (
        slice(0, -win_lon),
        slice(-win_lon, -shift_lon),
        slice(-shift_lon, None),
    )

    cnt = 0
    for pl in pl_slices:
        for lat in lat_slices:
            for lon in lon_slices:
                img_mask[:, pl, lat, lon, :] = cnt
                cnt += 1

    img_mask = img_mask[:, :, :, :Lon, :]

    mask_windows = window_partition(
        img_mask, window_size
    )  # n_lon, n_pl*n_lat, win_pl, win_lat, win_lon, 1
    mask_windows = mask_windows.view(
        mask_windows.shape[0], mask_windows.shape[1], win_pl * win_lat * win_lon
    )
    attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )
    return attn_mask


def get_earth_position_index(window_size):
    """
    This function construct the position index to reuse symmetrical parameters of the position bias.
    implementation from: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    Args:
        window_size (tuple[int]): [pressure levels, latitude, longitude]

    Returns:
        position_index (torch.Tensor): [win_pl * win_lat * win_lon, win_pl * win_lat * win_lon]
    """
    win_pl, win_lat, win_lon = window_size
    # Index in the pressure level of query matrix
    coords_zi = torch.arange(win_pl)
    # Index in the pressure level of key matrix
    coords_zj = -torch.arange(win_pl) * win_pl

    # Index in the latitude of query matrix
    coords_hi = torch.arange(win_lat)
    # Index in the latitude of key matrix
    coords_hj = -torch.arange(win_lat) * win_lat

    # Index in the longitude of the key-value pair
    coords_w = torch.arange(win_lon)

    # Change the order of the index to calculate the index in total
    coords_1 = torch.stack(torch.meshgrid([coords_zi, coords_hi, coords_w]))
    coords_2 = torch.stack(torch.meshgrid([coords_zj, coords_hj, coords_w]))
    coords_flatten_1 = torch.flatten(coords_1, 1)
    coords_flatten_2 = torch.flatten(coords_2, 1)
    coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    coords = coords.permute(1, 2, 0).contiguous()

    # Shift the index for each dimension to start from 0
    coords[:, :, 2] += win_lon - 1
    coords[:, :, 1] *= 2 * win_lon - 1
    coords[:, :, 0] *= (2 * win_lon - 1) * win_lat * win_lat

    # Sum up the indexes in three dimensions
    position_index = coords.sum(-1)

    return position_index


class UpSample(nn.Module):
    """
    Up-sampling operation.
    Implementation from: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        output_resolution (tuple[int]): [pressure levels, latitude, longitude]
    """

    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (B, N, C)
        """
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        x = self.linear1(x)
        x = x.reshape(B, in_pl, in_lat, in_lon, 2, 2, C // 2).permute(
            0, 1, 2, 4, 3, 5, 6
        )
        x = x.reshape(B, in_pl, in_lat * 2, in_lon * 2, -1)

        assert in_pl == out_pl, "the dimension of pressure level shouldn't change"
        pad_h = in_lat * 2 - out_lat
        pad_w = in_lon * 2 - out_lon

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = x[
            :,
            :out_pl,
            pad_top : 2 * in_lat - pad_bottom,
            pad_left : 2 * in_lon - pad_right,
            :,
        ]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])
        x = self.norm(x)
        x = self.linear2(x)
        return x


class DownSample(nn.Module):
    """
    Down-sampling operation
    Implementation from: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    Args:
        in_dim (int): Number of input channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        output_resolution (tuple[int]): [pressure levels, latitude, longitude]
    """

    def __init__(self, in_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        assert in_pl == out_pl, "the dimension of pressure level shouldn't change"
        h_pad = out_lat * 2 - in_lat
        w_pad = out_lon * 2 - in_lon

        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top

        pad_left = w_pad // 2
        pad_right = w_pad - pad_left

        pad_front = pad_back = 0

        self.pad = nn.ZeroPad3d(
            (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        )

    def forward(self, x):
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution
        x = x.reshape(B, in_pl, in_lat, in_lon, C)

        # Padding the input to facilitate downsampling
        x = self.pad(x.permute(0, -1, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        x = x.reshape(B, in_pl, out_lat, 2, out_lon, 2, C).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, out_pl * out_lat * out_lon, 4 * C)

        x = self.norm(x)
        x = self.linear(x)
        return x


class EarthAttention3D(nn.Module):
    """
    3D window attention with earth position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        window_size (tuple[int]): [pressure levels, latitude, longitude]
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        input_resolution,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wpl, Wlat, Wlon
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.type_of_windows = (input_resolution[0] // window_size[0]) * (
            input_resolution[1] // window_size[1]
        )

        self.earth_position_bias_table = nn.Parameter(
            torch.zeros(
                (window_size[0] ** 2)
                * (window_size[1] ** 2)
                * (window_size[2] * 2 - 1),
                self.type_of_windows,
                num_heads,
            )
        )  # Wpl**2 * Wlat**2 * Wlon*2-1, Npl//Wpl * Nlat//Wlat, nH

        earth_position_index = get_earth_position_index(
            window_size
        )  # Wpl*Wlat*Wlon, Wpl*Wlat*Wlon
        self.register_buffer("earth_position_index", earth_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.earth_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x: input features with shape of (B * num_lon, num_pl*num_lat, N, C)
            mask: (0/-inf) mask with shape of (num_lon, num_pl*num_lat, Wpl*Wlat*Wlon, Wpl*Wlat*Wlon)
        """
        B_, nW_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, nW_, N, 3, self.num_heads, C // self.num_heads)
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        earth_position_bias = self.earth_position_bias_table[
            self.earth_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.type_of_windows,
            -1,
        )  # Wpl*Wlat*Wlon, Wpl*Wlat*Wlon, num_pl*num_lat, nH
        earth_position_bias = earth_position_bias.permute(
            3, 2, 0, 1
        ).contiguous()  # nH, num_pl*num_lat, Wpl*Wlat*Wlon, Wpl*Wlat*Wlon
        attn = attn + earth_position_bias.unsqueeze(0)

        if mask is not None:
            nLon = mask.shape[0]
            attn = attn.view(
                B_ // nLon, nLon, self.num_heads, nW_, N, N
            ) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, nW_, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B_, nW_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EarthSpecificBlock(nn.Module):
    """
    3D Transformer Block
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size [pressure levels, latitude, longitude].
        shift_size (tuple[int]): Shift size for SW-MSA [pressure levels, latitude, longitude].
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=None,
        shift_size=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        window_size = (2, 6, 12) if window_size is None else window_size
        shift_size = (1, 3, 6) if shift_size is None else shift_size
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        padding = get_pad3d(input_resolution, window_size)
        self.pad = nn.ZeroPad3d(padding)

        pad_resolution = list(input_resolution)
        pad_resolution[0] += padding[-1] + padding[-2]
        pad_resolution[1] += padding[2] + padding[3]
        pad_resolution[2] += padding[0] + padding[1]

        self.attn = EarthAttention3D(
            dim=dim,
            input_resolution=pad_resolution,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        shift_pl, shift_lat, shift_lon = self.shift_size
        self.roll = shift_pl and shift_lon and shift_lat

        if self.roll:
            attn_mask = get_shift_window_mask(pad_resolution, window_size, shift_size)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor):
        Pl, Lat, Lon = self.input_resolution
        B, L, C = x.shape
        assert L == Pl * Lat * Lon, f"input feature has wrong size: {L} vs {Pl*Lat*Lon}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, Pl, Lat, Lon, C)

        # start pad
        x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

        _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape

        shift_pl, shift_lat, shift_lon = self.shift_size
        if self.roll:
            shifted_x = torch.roll(
                x, shifts=(-shift_pl, -shift_lat, -shift_lon), dims=(1, 2, 3)
            )
            x_windows = window_partition(shifted_x, self.window_size)
            # B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)
            # B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C

        win_pl, win_lat, win_lon = self.window_size
        x_windows = x_windows.view(
            x_windows.shape[0], x_windows.shape[1], win_pl * win_lat * win_lon, C
        )
        # B*num_lon, num_pl*num_lat, win_pl*win_lat*win_lon, C

        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # B*num_lon, num_pl*num_lat, win_pl*win_lat*win_lon, C

        attn_windows = attn_windows.view(
            attn_windows.shape[0], attn_windows.shape[1], win_pl, win_lat, win_lon, C
        )

        if self.roll:
            shifted_x = window_reverse(
                attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad
            )
            # B * Pl * Lat * Lon * C
            x = torch.roll(
                shifted_x, shifts=(shift_pl, shift_lat, shift_lon), dims=(1, 2, 3)
            )
        else:
            shifted_x = window_reverse(
                attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad
            )
            x = shifted_x

        # crop, end pad
        x = crop3d(x.permute(0, 4, 1, 2, 3), self.input_resolution).permute(
            0, 2, 3, 4, 1
        )

        x = x.reshape(B, Pl * Lat * Lon, C)
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """A basic 3D Transformer layer for one stage

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                EarthSpecificBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0, 0) if i % 2 == 0 else None,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


def get_resolutions(size: int):
    def factorisation(n):
        factors = []
        i = int(np.sqrt(n))
        while i > 1:
            if n % i == 0:
                return i, int(n // i)
            i -= 1

        return None, None

    return factorisation(size)


class Pangu(nn.Module):
    """
    Pangu A PyTorch impl of: `Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast`
    - https://arxiv.org/abs/2211.02556

    Args:
        embed_dim (int): Patch embedding dimension. Default: 192
        num_heads (tuple[int]): Number of attention heads in different layers.
        window_size (tuple[int]): Window size.
    """

    def __init__(self, embed_dim=192, num_heads=(6, 12, 12, 6), window_size=(2, 6, 12)):
        super().__init__()
        drop_path = np.linspace(0, 0.2, 8).tolist()
        # In addition, three constant masks(the topography mask, land-sea mask and soil type mask)

        num_surface_features = 10
        num_static_features = 2

        self.patchembed2d = PatchEmbed2D(
            img_size=swap_tuple(args.input_size),
            patch_size=(4, 4),
            in_chans=num_surface_features + num_static_features,  # add
            embed_dim=embed_dim,
        )

        num_upper_levels = 6
        num_upper_level_features = 5

        self.patchembed3d = PatchEmbed3D(
            img_size=(num_upper_levels,) + swap_tuple(args.input_size),
            patch_size=(2, 4, 4),
            in_chans=num_upper_level_features,
            embed_dim=embed_dim,
        )

        # original: input_resolution  = (8, 181, 360)
        #           output_resolution = (8, 91, 180)
        # input_resolution = (8, 56, 56)
        # output_resolution = (8, 32, 32)

        x_height = args.input_size[1] + (4 - args.input_size[1] % 4)
        x_width = args.input_size[0] + (4 - args.input_size[0] % 4)

        in_size = np.ceil(x_height * x_width * 12
            * (1 + num_upper_levels // 2)
            / 192 / 8
        )

        input_resolution = (8,) + get_resolutions(in_size)
        output_resolution = (8,) + (
            int(np.ceil(input_resolution[1] / 2)),
            int(np.ceil(input_resolution[2] / 2)),
        )

        self.layer1 = BasicLayer(
            dim=embed_dim,
            input_resolution=input_resolution,
            depth=2,
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[:2],
        )
        self.downsample = DownSample(
            in_dim=embed_dim,
            input_resolution=input_resolution,
            output_resolution=output_resolution,
        )

        input_resolution, output_resolution = swap_tuple(
            (input_resolution, output_resolution)
        )

        self.layer2 = BasicLayer(
            dim=embed_dim * 2,
            input_resolution=input_resolution,
            depth=6,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[2:],
        )
        self.layer3 = BasicLayer(
            dim=embed_dim * 2,
            input_resolution=input_resolution,
            depth=6,
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[2:],
        )
        self.upsample = UpSample(
            embed_dim * 2, embed_dim, input_resolution, output_resolution
        )
        self.layer4 = BasicLayer(
            dim=embed_dim,
            input_resolution=output_resolution,
            depth=2,
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[:2],
        )
        # The outputs of the 2nd encoder layer and the 7th decoder layer are concatenated along the channel dimension.
        self.patchrecovery2d = PatchRecovery2D(
            swap_tuple(args.input_size), (4, 4), 2 * embed_dim, num_surface_features
        )
        self.patchrecovery3d = PatchRecovery3D(
            (num_upper_levels,) + swap_tuple(args.input_size),
            (2, 4, 4),
            2 * embed_dim,
            num_upper_level_features,
        )

    def forward(self, surface, surface_mask, upper_air):
        """
        Args:
            surface (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=4.
            surface_mask (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=3.
            upper_air (torch.Tensor): 3D n_pl=13, n_lat=721, n_lon=1440, chans=5.
        """
        surface = torch.concat([surface, surface_mask], dim=1)
        surface = self.patchembed2d(surface)
        upper_air = self.patchembed3d(upper_air)

        x = torch.concat([surface.unsqueeze(2), upper_air], dim=2)
        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)

        x = self.layer1(x)

        skip = x

        x = self.downsample(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.upsample(x)
        x = self.layer4(x)

        output = torch.concat([x, skip], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        output_surface = output[:, :, 0, :, :]
        output_upper_air = output[:, :, 1:, :, :]

        output_surface = self.patchrecovery2d(output_surface)
        output_upper_air = self.patchrecovery3d(output_upper_air)
        return output_surface, output_upper_air


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Pangu_lite(nn.Module):
    """
    Pangu A PyTorch impl of: `Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast`
    - https://arxiv.org/abs/2211.02556

    Args:
        embed_dim (int): Patch embedding dimension. Default: 192
        num_heads (tuple[int]): Number of attention heads in different layers.
        window_size (tuple[int]): Window size.
    """

    def __init__(self, embed_dim=192, num_heads=(6, 12, 12, 6), window_size=(2, 6, 12)):
        super().__init__()
        drop_path = np.linspace(0, 0.2, 8).tolist()
        # In addition, three constant masks(the topography mask, land-sea mask and soil type mask)

        patch_size = (8, 8)

        num_surface_features = 2
        num_static_features = 2
        self.patchembed2d = PatchEmbed2D(
            img_size=swap_tuple(args.input_size),
            patch_size=patch_size,
            in_chans=num_surface_features + num_static_features,  # add
            embed_dim=embed_dim,
        )

        num_upper_levels = 6
        num_upper_level_features = 3
        self.patchembed3d = PatchEmbed3D(
            img_size=(num_upper_levels,) + swap_tuple(args.input_size),
            patch_size=(2,) + patch_size,
            in_chans=num_upper_level_features,
            embed_dim=embed_dim,
        )

        # input_resolution = (9, 32, 49)  # 448x448 all parameters
        # output_resolution = (8, 16, 25)  # 448x448 all parameters
        # input_resolution = (8, 67, 119)  # 1069x949 all parameters
        # output_resolution = (8, 34, 60)  # 1069x949 all parameters

        in_size = (
            (args.input_size[0] + (8 - args.input_size[0] % 8))
            * (args.input_size[1] + (8 - args.input_size[1] % 8))
            / 192
        )

        input_resolution = (8,) + get_resolutions(in_size)
        output_resolution = (8,) + (
            int(np.ceil(input_resolution[1] / 2)),
            int(np.ceil(input_resolution[2] / 2)),
        )

        self.layer1 = BasicLayer(
            dim=embed_dim,
            input_resolution=input_resolution,
            depth=2,
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[:2],
        )
        self.downsample = DownSample(
            in_dim=embed_dim,
            input_resolution=input_resolution,
            output_resolution=output_resolution,
        )

        input_resolution, output_resolution = swap_tuple(
            (input_resolution, output_resolution)
        )

        self.layer2 = BasicLayer(
            dim=embed_dim * 2,
            input_resolution=input_resolution,
            depth=6,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[2:],
        )
        self.layer3 = BasicLayer(
            dim=embed_dim * 2,
            input_resolution=input_resolution,
            depth=6,
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[2:],
        )

        self.upsample = UpSample(
            embed_dim * 2, embed_dim, input_resolution, output_resolution
        )

        self.layer4 = BasicLayer(
            dim=embed_dim,
            input_resolution=output_resolution,
            depth=2,
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[:2],
        )
        # The outputs of the 2nd encoder layer and the 8th decoder layer are concatenated along the channel dimension.
        self.patchrecovery2d = PatchRecovery2D(
            swap_tuple(args.input_size), patch_size, 2 * embed_dim, num_surface_features
        )
        self.patchrecovery3d = PatchRecovery3D(
            (num_upper_levels,) + swap_tuple(args.input_size),
            (2,) + patch_size,
            2 * embed_dim,
            num_upper_level_features,
        )

    def forward(self, surface, surface_mask, upper_air):
        """
        Args:
            surface (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=4.
            surface_mask (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=3.
            upper_air (torch.Tensor): 3D n_pl=13, n_lat=721, n_lon=1440, chans=5.
        """
        surface = torch.concat([surface, surface_mask], dim=1)
        surface = self.patchembed2d(surface)
        upper_air = self.patchembed3d(upper_air)

        x = torch.concat([surface.unsqueeze(2), upper_air], dim=2)
        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)
        x = self.layer1(x)

        skip = x

        x = self.downsample(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.upsample(x)
        x = self.layer4(x)

        output = torch.concat([x, skip], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        output_surface = output[:, :, 0, :, :]
        output_upper_air = output[:, :, 1:, :, :]

        output_surface = self.patchrecovery2d(output_surface)
        output_upper_air = self.patchrecovery3d(output_upper_air)
        return output_surface, output_upper_air


if __name__ == "__main__":
    model = Pangu_lite()
