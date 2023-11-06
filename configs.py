import argparse


def get_args():
    parser = argparse.ArgumentParser("Swin training and evaluation", add_help=False)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--input_size", default=224, type=int, help="Input image size")
    parser.add_argument("--model_name", default="", type=str, help="Model name")
    parser.add_argument(
        "--model_dir", default="/tmp", type=str, help="Model output dir"
    )
    parser.add_argument(
        "--dataseries_file",
        default="/home/partio/cloudnwc/effective_cloudiness/data/dataseries/npz/224x224/nwcsaf-effective-cloudiness-20190801-20200801-img_size=224x224-float32.npz",
        type=str,
    )
    parser.add_argument("--plot_best", default=False, action="store_true")

    # model parameters
    parser.add_argument(
        "--n_hist", default=4, type=int, help="Number of history images"
    )
    parser.add_argument(
        "--n_pred", default=1, type=int, help="Number of predicted images"
    )
    parser.add_argument("--leadtime_conditioning", default=0, type=int)

    args = parser.parse_args()

    return args
