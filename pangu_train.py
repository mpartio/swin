from saf import create_generators
import torch
from torch.utils.data import DataLoader
from pangu_model import Pangu, Pangu_lite
from tqdm import tqdm
from configs import get_args
from pangu_utils import split_surface_data, split_upper_air_data, split_weights
import shutil
import os

args = get_args()


def calc_loss(
    output_surface,
    output_upper_air,
    target_surface,
    target_upper_air,
    weights_surface,
    weights_upper_air,
):
    l1loss = torch.nn.L1Loss(reduction="none")
    bcloss = torch.nn.BCEWithLogitsLoss(reduction="none")

    assert output_surface.shape == target_surface.shape  # (B C H W)
    assert output_upper_air.shape == target_upper_air.shape  # (B C Z H W)

    loss_effc = bcloss(
        output_surface[:, 0:1, :, :], target_surface[:, 0:1, :, :]
    ).permute(
        0, 2, 3, 1
    )  # (B C H W) to (B H W C)
    loss_mslp = l1loss(
        output_surface[:, 1:2, :, :], target_surface[:, 1:2, :, :]
    ).permute(
        0, 2, 3, 1
    )  # (B C H W) to (B H W C)

    loss_surface = torch.cat([loss_effc, loss_mslp], dim=-1)
    # loss_surface = l1loss(output_surface, target_surface).permute(
    #    0, 2, 3, 1
    # )  # (B C H W) to (B H W C)
    loss_surface = torch.mean(loss_surface * weights_surface)

    loss_upper_air = l1loss(output_upper_air, target_upper_air).permute(
        0, 3, 4, 1, 2
    )  # (B C Z H W) to (B H W Z C)

    loss_upper_air = (
        loss_upper_air.reshape(loss_upper_air.shape[:3] + (-1,)) * weights_upper_air
    )

    loss_surface = torch.mean(loss_surface)
    loss_upper_air = torch.mean(loss_upper_air)

    loss = loss_upper_air + loss_surface

    return loss


def train(model, train_loader, val_loader, surface_mask):
    """Training code"""
    # Prepare for the optimizer and scheduler
    optimizer = torch.optim.Adam(m.parameters(), lr=5e-4, weight_decay=3e-6)

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #    optimizer, 10, eta_min=0, last_epoch=-1, verbose=False
    # )  # used in the paper

    # Loss function
    # criterion = nn.L1Loss(reduction="none")

    # training epoch
    epochs = 200

    best_loss = float("inf")
    epochs_since_last_improvement = 0
    # scaler = torch.cuda.amp.GradScaler()

    weights = torch.load("parameter_weights.pt")
    surface_weights, upper_air_weights = split_weights(weights)

    parameter_mean = torch.load("parameter_mean.pt")
    parameter_std = torch.load("parameter_std.pt")

    surface_mean = parameter_mean[0].to(args.device)
    upper_mean = parameter_mean[1:].to(args.device)
    surface_std = parameter_std[0].to(args.device)
    upper_std = parameter_std[1:].to(args.device)

    # Train a single Pangu-Weather model
    for i in range(epochs + 1):
        epoch_loss = 0.0
        print(f"Epoch {i}")
        model.train()

        for id, train_data in enumerate(tqdm(train_loader)):
            # B C W H
            input_all, target_all = train_data
            input_surface = split_surface_data(input_all)
            input_upper_air = split_upper_air_data(input_all)

            target_surface = split_surface_data(target_all)
            target_upper_air = split_upper_air_data(target_all)

            optimizer.zero_grad()

            with torch.autocast(device_type=args.device):
                output_surface, output_upper_air = model(
                    input_surface, surface_mask, input_upper_air
                )

                loss = calc_loss(
                    output_surface,
                    output_upper_air,
                    target_surface,
                    target_upper_air,
                    surface_weights,
                    upper_air_weights,
                )
            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        # Begin to validate
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            for id, val_data in enumerate(tqdm(val_loader)):
                input_all, target_all = val_data

                input_surface = split_surface_data(input_all)
                input_upper_air = split_upper_air_data(input_all)

                target_surface = split_surface_data(target_all)
                target_upper_air = split_upper_air_data(target_all)

                output_surface, output_upper_air = model(
                    input_surface, surface_mask, input_upper_air
                )

                assert (
                    torch.isnan(output_surface).sum() == 0
                ), "output_surface has nan values"
                loss = calc_loss(
                    output_surface,
                    output_upper_air,
                    target_surface,
                    target_upper_air,
                    surface_weights,
                    upper_air_weights,
                )

                val_loss += loss.item()

            val_loss /= len(val_loader)
            print("Validation loss: {:.6f}".format(val_loss))

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                model_file_name = f"{args.model_dir}/trained_model_state_dict"
                torch.save(model.state_dict(), model_file_name)
                print(
                    "Saved current best model (loss={:.6f}) to {}".format(
                        val_loss, model_file_name
                    )
                )
                epochs_since_last_improvement = 0
            else:
                epochs_since_last_improvement += 1
                if epochs_since_last_improvement >= 10:
                    print(
                        f"No improvement in validation loss for {epochs_since_last_improvement} epochs, terminating training."
                    )
                    break


def save_meta():
    os.makedirs(args.model_dir, exist_ok=True)

    with open(f"{args.model_dir}/args.txt", "w") as f:
        print(args, file=f)

    shutil.copyfile("parameter_weights.pt", f"{args.model_dir}/parameter_weights.pt")
    shutil.copyfile("parameter_mean.pt", f"{args.model_dir}/parameter_mean.pt")
    shutil.copyfile("parameter_std.pt", f"{args.model_dir}/parameter_std.pt")


if __name__ == "__main__":
    m = Pangu_lite().to(args.device)
    # m = Pangu().to(args.device)
    train_ds, valid_ds = create_generators(train_val_split=0.8)

    lsm = train_ds.get_static_features("lsm_heightAboveGround_0")
    z = train_ds.get_static_features("z_heightAboveGround_0")
    #    soil = torch.ones_like(lsm)
    #    surface_mask = torch.stack([lsm, z, soil]).to(args.device)
    surface_mask = torch.stack([lsm, z]).to(args.device)

    surface_mask = surface_mask.unsqueeze(0).repeat(args.batch_size, 1, 1, 1)

    args = get_args()

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        pin_memory=False,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        pin_memory=False,
        drop_last=True,
    )

    if args.load_model:
        m.load_state_dict(
            torch.load(
                "{}/trained_model_state_dict".format(args.model_dir),
                map_location=torch.device(args.device),
            )
        )

        print("Loaded model from {}".format(args.model_dir))

    save_meta()
    train(m, train_loader, valid_loader, surface_mask)
