import argparse


def parse_size(s):
    try:
        w, h = map(int, s.split("x"))
        return w, h
    except:
        raise ValueError("Size must be W,H")


def default_params():
    params = [
        "effective-cloudiness_heightAboveGround_0",
        "mld_heightAboveGround_0",
        "pres_heightAboveGround_0",
        "fgcorr_heightAboveGround_10",
        "rcorr_heightAboveGround_2",
        "t_heightAboveGround_0",
        "tcorr_heightAboveGround_2",
        "ucorr_heightAboveGround_10",
        "vcorr_heightAboveGround_10",
        "pres_heightAboveSea_0",
        "r_isobaricInhPa_300",
        "t_isobaricInhPa_300",
        "u_isobaricInhPa_300",
        "v_isobaricInhPa_300",
        "z_isobaricInhPa_300",
        "r_isobaricInhPa_500",
        "t_isobaricInhPa_500",
        "u_isobaricInhPa_500",
        "v_isobaricInhPa_500",
        "z_isobaricInhPa_500",
        "r_isobaricInhPa_700",
        "t_isobaricInhPa_700",
        "u_isobaricInhPa_700",
        "v_isobaricInhPa_700",
        "z_isobaricInhPa_700",
        "r_isobaricInhPa_850",
        "t_isobaricInhPa_850",
        "u_isobaricInhPa_850",
        "v_isobaricInhPa_850",
        "z_isobaricInhPa_850",
        "r_isobaricInhPa_925",
        "t_isobaricInhPa_925",
        "u_isobaricInhPa_925",
        "v_isobaricInhPa_925",
        "z_isobaricInhPa_925",
        "r_isobaricInhPa_1000",
        "t_isobaricInhPa_1000",
        "u_isobaricInhPa_1000",
        "v_isobaricInhPa_1000",
        "z_isobaricInhPa_1000",
    ]
    return params


def get_args():
    parser = argparse.ArgumentParser("Swin training and evaluation", add_help=False)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument(
        "--input_size", default=(224, 224), type=parse_size, help="Input image size"
    )
    parser.add_argument("--model_name", default="", type=str, help="Model name")
    parser.add_argument(
        "--model_dir", default="/tmp", type=str, help="Model output dir"
    )
    parser.add_argument(
        "--dataseries_file",
        type=str,
    )
    parser.add_argument(
        "--dataseries_directory",
        type=str,
    )

    parser.add_argument(
        "--load_model_from",
        type=str,
        help="Load model from this directory",
    )

    parser.add_argument(
        "--n_hist", default=1, type=int, help="Number of history images"
    )
    parser.add_argument(
        "--n_pred", default=1, type=int, help="Number of predicted images"
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        default=default_params(),
        help="Parameters to use for training",
    )
    parser.add_argument(
        "--n_workers", default=4, type=int, help="Number of data loaders"
    )

    args = parser.parse_args()

    if type(args.parameters) == str:
        args.parameters = [args.parameters]

    if args.n_workers > args.batch_size:
        args.n_workers = args.batch_size

    if args.dataseries_file is None and args.dataseries_directory is None:
        raise ValueError("Either dataseries_file or dataseries_directory must be set")

    return args
