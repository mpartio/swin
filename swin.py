import timm
import torch
import torch.nn as nn
from configs import get_args
from transformers import (
    SwinConfig,
    Swinv2Config,
    SwinForMaskedImageModeling,
    Swinv2ForMaskedImageModeling,
)

args = get_args()


def get_swin_config(name):
    # https://github.com/microsoft/Swin-Transformer/tree/main/configs

    if name == "swin-tiny-patch4-window7":
        configuration = SwinConfig(
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            embed_dim=96,
            drop_path_rate=0.2,
            window_size=7,
            num_channels=args.n_hist + int(bool(args.leadtime_conditioning)),
            image_size=args.input_size,
        )

    elif name == "swin-base-patch4-window7":
        configuration = SwinConfig(
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            embed_dim=128,
            drop_path_rate=0.5,
            window_size=7,
            num_channels=args.n_hist + int(bool(args.leadtime_conditioning)),
            image_size=args.input_size,
        )

    elif name == "swin-large-patch4-window7":
        configuration = SwinConfig(
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            embed_dim=192,
            drop_path_rate=0.2,
            window_size=7,
            num_channels=args.n_hist + int(bool(args.leadtime_conditioning)),
            image_size=args.input_size,
        )

    elif name == "swinv2-base-patch4-window8":
        configuration = Swinv2Config(
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            embed_dim=128,
            drop_path_rate=0.5,
            window_size=8,
            num_channels=args.n_hist + int(bool(args.leadtime_conditioning)),
            image_size=args.input_size,
        )
    print(configuration)
    return configuration


class CloudCastSwin(nn.Module):
    def __init__(self, model_name):
        super(CloudCastSwin, self).__init__()

        model_fam = model_name.split("-")[0]

        print("Creating model {} of family {}".format(model_name, model_fam))

        if model_fam == "swin":
            self.base = SwinForMaskedImageModeling(get_swin_config(model_name))

        elif model_fam == "swinv2":
            self.base = Swinv2ForMaskedImageModeling(get_swin_config(model_name))

        self.base.decoder = nn.Sequential(
            self.base.decoder[0],
            self.base.decoder[1],
            nn.Conv2d(
                args.n_hist + int(bool(args.leadtime_conditioning)),
                1,
                kernel_size=(1, 1),
                stride=(1, 1),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.base(x)
        x = x.reconstruction
        return x


def create_model(model_name):
    m = CloudCastSwin(model_name)
    m = m.to(args.device)
    return m


def main():
    m = create_model("swin-base-patch4-window7")
    x = torch.randn(1, args.n_hist, 224, 224).to(args.device)
    print(x.shape)
    y = m(x)
    print(y.shape)


if __name__ == "__main__":
    main()
