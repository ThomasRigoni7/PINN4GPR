import numpy as np
import torch
torch.manual_seed(42)
import torch.nn as nn
import cv2

from src.visualization.misc import save_field_animation
from src.pinns.paper.dataset import MyNormalizer, PaperDataset
from src.pinns.paper.model import MLP, IMG_SIZE, get_f, get_PINN_warmup_loss_fn, get_PINN_uniform_loss_fn, L4loss, show_predictions
from src.pinns.paper.train import train



def uniform_material(warmup: bool):
    """
    Trains a PINN for the uniform material model and compares it with the same architecture trained as a regular NN.
    """
    DEVICE = "cuda:2"
    LR = 0.001
    RNG = np.random.default_rng(42)
    N_COLLOCATION_POINTS = 40000
    COLLOCATION_DOMAIN_SIZE = (30e-9, 20, 20)
    EPOCHS_WARMUP = 20000
    EPOCHS = 3000

    snapshots = np.load("paper_data/uniform_wavefield.npz")["0000_E"]

    if IMG_SIZE != (200, 200):
        snaps = []
        for s in snapshots:
            s = cv2.resize(s, IMG_SIZE)
            snaps.append(s)
        snapshots = np.array(snaps)


    geometry = torch.broadcast_to(torch.tensor([1, 0, 1, 0]), (IMG_SIZE[0], IMG_SIZE[1], 4)).transpose(0, 2) 
    geometry = geometry.float().to(DEVICE)

    # define models and optimizers
    PINN_model = MLP(3, [128, 128, 128, 128, 128], 1, nn.SiLU)
    # PINN_model.apply(PINN_model.init_weights)
    PINN_model = PINN_model.to(DEVICE)

    regular_model = MLP(3, [128, 128, 128, 128, 128], 1, nn.ReLU)
    # regular_model.load_state_dict(torch.load("checkpoints/NN_model_best_L4_20k_silu_5x256.ckp"), strict=True)
    regular_model = regular_model.to(DEVICE)

    PINN_optimizer = torch.optim.Adam(PINN_model.parameters(), lr = LR)
    regular_optimizer = torch.optim.Adam(regular_model.parameters(), lr = LR)


    # Create the dataset
    train_indexes = [15, 25]
    train_dataset = PaperDataset(snapshots[train_indexes], t_offsets=train_indexes)
    print("Train dataset points:")
    train_dataset.print_info()
    # save_field_animation(train_dataset.snapshots.reshape((-1, 200, 200)), None, interval=50, bound_mult_factor=0.0005)
    # frame_15ns = train_dataset.get_frame(1)
    #show_field(frame_15ns)
    scaler = train_dataset.scaler
    val_indexes = [20]
    val_dataset = PaperDataset(snapshots[val_indexes], t_offsets=val_indexes, scaler=scaler)
    print("Validation dataset points:")
    val_dataset.print_info()

    test_indexes = [40]
    test_dataset = PaperDataset(snapshots[test_indexes], t_offsets=test_indexes, scaler=scaler)
    print("Test dataset points:")
    test_dataset.print_info()
    # save_field_animation(val_dataset.snapshots.reshape((-1, 5, 5)), None, interval=50)
    # frame_60ns = val_dataset.get_frame(0)
    # show_field(frame_60ns)


    # collocation points
    collocation_points = RNG.uniform(size=(3, N_COLLOCATION_POINTS)) * np.array(COLLOCATION_DOMAIN_SIZE).reshape(3, -1)
    tc, yc, xc = collocation_points
    tc = tc + 1.5e-8
    collocation_points = np.stack([xc, yc, tc])
    collocation_points = torch.tensor(collocation_points, dtype=torch.float32)
    # collocation_points, _ = scaler.transform(collocation_points.T, None)
    collocation_points = collocation_points.to(DEVICE)

    print("collocation points:", collocation_points.shape)
    print("min:", collocation_points.min(axis=1).values)
    print("max:", collocation_points.max(axis=1).values)

    boundary_points = None

    # get the derivative functions
    f_PINN = get_f(PINN_model, scaler)
    PINN_loss_fn_L4 = get_PINN_warmup_loss_fn(L4loss)
    PINN_loss_fn_L2 = get_PINN_uniform_loss_fn(nn.MSELoss())
    regular_loss_fn = nn.MSELoss()

    # get f for regular model
    f_regular = get_f(regular_model, scaler)
    #plot_data_histogram(train_dataset.data, train_dataset.labels)

    # generate dataset
    print("building train dataset...")
    train_samples = torch.cat([train_dataset.data, train_dataset.labels[:, None]], dim=1).T.to(DEVICE)

    print("building validation dataset...")
    val_samples = torch.cat([val_dataset.data, val_dataset.labels[:, None]], dim=1).T.to(DEVICE)


    print("building test dataset...")
    test_samples = torch.cat([test_dataset.data, test_dataset.labels[:, None]], dim=1).T.to(DEVICE)

    if warmup:
        best_PINN_model, last_PINN_model, best_regular_model, last_regular_model = train(PINN_model,
                                                    f_PINN,
                                                    PINN_optimizer,
                                                    PINN_loss_fn_L4,
                                                    regular_model,
                                                    f_regular,
                                                    regular_optimizer,
                                                    L4loss,
                                                    train_samples,
                                                    val_samples,
                                                    boundary_points,
                                                    collocation_points,
                                                    geometry,
                                                    EPOCHS_WARMUP)

        best_PINN_model = best_PINN_model.to(DEVICE)
        best_regular_model = best_regular_model.to(DEVICE)
        regular_optimizer = torch.optim.Adam(best_regular_model.parameters(), lr = LR)

        f_PINN = get_f(best_PINN_model, scaler)
        f_regular = get_f(best_regular_model, scaler)
        PINN_optimizer = torch.optim.Adam(best_PINN_model.parameters(), lr = LR)
    else:
        best_PINN_model = PINN_model
        best_regular_model = regular_model

    best_PINN_model, last_PINN_model, best_regular_model, last_regular_model = train(best_PINN_model,
                                                f_PINN,
                                                PINN_optimizer,
                                                PINN_loss_fn_L2,
                                                best_regular_model,
                                                f_regular,
                                                regular_optimizer,
                                                regular_loss_fn,
                                                train_samples,
                                                val_samples,
                                                boundary_points,
                                                collocation_points,
                                                geometry,
                                                EPOCHS)

    torch.save(last_PINN_model.state_dict(), "checkpoints/PINN_model_last.ckp")
    torch.save(best_PINN_model.state_dict(), "checkpoints/PINN_model_best.ckp")
    torch.save(last_regular_model.state_dict(), "checkpoints/NN_model_last.ckp")
    torch.save(best_regular_model.state_dict(), "checkpoints/NN_model_best.ckp")

    best_PINN_model = best_PINN_model.to(DEVICE)
    best_regular_model = best_regular_model.to(DEVICE)

    show_predictions(f_PINN, f_regular, val_samples)
    show_predictions(f_PINN, f_regular, test_samples)

if __name__ == "__main__":
    uniform_material(True)