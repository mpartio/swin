import argparse

def parse_size(s):
    try:
        w, h = map(int, s.split("x"))
        return w, h
    except:
        raise ValueError("Size must be W,H")

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
        "--load_model",
        default=False,
        action="store_true",
        help="Load model from checkpoint",
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
        default="effective_cloudiness_heightAboveGround_0",
        help="Parameters to use for training",
    )
    parser.add_argument(
        "--n_workers", default=4, type=int, help="Number of data loaders"
    )

    args = parser.parse_args()

    if type(args.parameters) == str:
        args.parameters = [args.parameters]

    if args.n_workers > args.batch_size:
        print("n_workers set to batch_size")
        args.n_workers = args.batch_size

    if args.dataseries_file is None and args.dataseries_directory is None:
        raise ValueError("Either dataseries_file or dataseries_directory must be set")

    return args
