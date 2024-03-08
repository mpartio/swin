import timm
import torch
import torch.nn as nn
import numpy as np
from configs import get_args
from transformers import (
    SwinConfig,
    Swinv2Config,
    SwinForMaskedImageModeling,
    Swinv2ForMaskedImageModeling,
)
from swin_unet import SwinTransformerSys as SwinUnet
from trans_unet import VisionTransformer as TransUnet
import ml_collections

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
            num_channels=args.n_hist,
            image_size=args.input_size,
        )

    elif name == "swin-base-patch4-window7":
        configuration = SwinConfig(
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            embed_dim=128,
            drop_path_rate=0.5,
            window_size=7,
            num_channels=args.n_hist,
            image_size=args.input_size,
        )

    elif name == "swin-large-patch4-window7":
        configuration = SwinConfig(
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            embed_dim=192,
            drop_path_rate=0.2,
            window_size=7,
            num_channels=args.n_hist,
            image_size=args.input_size,
        )

    elif name == "swinv2-base-patch4-window8":
        configuration = Swinv2Config(
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            embed_dim=128,
            drop_path_rate=0.5,
            window_size=8,
            num_channels=args.n_hist,
            image_size=args.input_size,
        )

    elif name == "swinunet-tiny-patch4-window7":
        configuration = SwinConfig(
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            embed_dim=96,
            drop_path_rate=0.2,
            window_size=7,
            num_channels=args.n_hist,
            image_size=args.input_size,
        )

    print(configuration)

    return configuration


class CloudCastSwin(nn.Module):
    def __init__(self, model_name):
        super(CloudCastSwin, self).__init__()

        self.model_fam = model_name.split("-")[0]

        print("Creating model {} of family {}".format(model_name, self.model_fam))

        if self.model_fam == "swin":
            self.base = SwinForMaskedImageModeling(get_swin_config(model_name))

        elif self.model_fam == "swinv2":
            self.base = Swinv2ForMaskedImageModeling(get_swin_config(model_name))

        elif self.model_fam == "swinunet":
            self.base = SwinUnet()

        elif self.model_fam == "transunet":
            config = ml_collections.ConfigDict()
            config.patches = ml_collections.ConfigDict({"size": (16, 16)})
            config.hidden_size = 768
            config.transformer = ml_collections.ConfigDict()
            config.transformer.mlp_dim = 3072
            config.transformer.num_heads = 12
            config.transformer.num_layers = 12
            config.transformer.attention_dropout_rate = 0.0
            config.transformer.dropout_rate = 0.1
            # config.classifier = 'seg'
            config.representation_size = None
            config.resnet_pretrained_path = None
            # config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
            config.patch_size = 16

            config.activation = "softmax"

            config.patches.grid = (16, 16)
            config.resnet = ml_collections.ConfigDict()
            config.resnet.num_layers = (3, 4, 9)
            config.resnet.width_factor = 1

            config.classifier = "seg"
            # config.pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
            config.decoder_channels = (256, 128, 64, 16)
            config.skip_channels = [512, 256, 64, 16]
            config.n_classes = len(args.parameters)
            config.n_skip = 3
            config.activation = "softmax"
            config.patches.grid = (
                int(args.input_size[0] / 16),
                int(args.input_size[0] / 16),
            )
            ##config_vit = CONFIGS_ViT_seg[args.vit_name]
            # config_vit.n_classes = args.num_classes
            # config_vit.n_skip = args.n_skip

            self.base = TransUnet(
                config, img_size=args.input_size[0], num_classes=len(args.parameters)
            )

        elif self.model_fam == "swinunet3d":
            self.base = HyperSwinEncoderDecoder3D()

        elif self.model_fam == "swinunetXXX":
            config = get_swin_config(model_name)

            self.base = SwinUnet(
                img_size=config.image_size[0],
                patch_size=config.patch_size,
                in_chans=config.num_channels,
                num_classes=1,
                embed_dim=config.embed_dim,
                depths=config.depths,
                depths_decoder=[1, 2, 2, 2],
                num_heads=config.num_heads,
                window_size=config.window_size,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                qk_scale=None,
                drop_rate=config.hidden_dropout_prob,
                attn_drop_rate=config.attention_probs_dropout_prob,
                drop_path_rate=config.drop_path_rate,
                norm_layer=nn.LayerNorm,
                ape=False,
                patch_norm=True,
                use_checkpoint=False,
                final_upsample="expand_first",
            )

            # self.base = SwinUnet(get_swin_config(model_name))

        if self.model_fam in ("swin", "swinv2"):
            self.base.decoder = nn.Sequential(
                self.base.decoder[0],
                self.base.decoder[1],
                nn.Conv2d(
                    args.n_hist,
                    1,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                ),
                nn.Sigmoid(),
            )

    def forward(self, x):
        x = self.base(x)
        if self.model_fam in ("swin", "swinv2"):
            x = x.reconstruction

        return x


def create_model(model_name):
    m = CloudCastSwin(model_name)
    m = m.to(args.device)
    return m


def main():
    m = create_model("swinunet-tiny-patch4-window7")
    x = torch.randn(1, args.n_hist, 224, 224).to(args.device)
    X = x.cpu().detach().numpy()
    print(X.shape, np.min(X), np.mean(X), np.max(X))
    y = m(x)
    Y = y.cpu().detach().numpy()
    print(Y.shape, np.min(Y), np.mean(Y), np.max(Y))


if __name__ == "__main__":
    main()
