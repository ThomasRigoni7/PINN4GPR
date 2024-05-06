"""
This module contains code to create figures for the thesis and paper, and save the predictions to disk.
"""

import torch
import torch.nn as nn
from src.pinns.paper.model import MLP, get_f, show_field, predict_functional
from src.pinns.models import Time2Image
import numpy as np
# from src.pinns.paper.dataset import PaperDataset
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

DEVICE = "cuda:1"

###############
#     MLP     #
###############

def save_image_and_colorbar(path, img, vmin=None, vmax=None, extent=None, xlabel="Distance (m)", ylabel="Depth (m)"):
    fig = plt.figure(num=1, clear=True)
    # plt.imshow(img, vmin=vmin, vmax=vmax, extent=extent, cmap="jet")
    plt.imshow(img, vmin=vmin, vmax=vmax, extent=extent)
    plt.colorbar(pad=0.01)
    plt.tight_layout(pad = 0, h_pad=0, w_pad=0)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close("all")

def plot_ax(ax, img, vmin=None, vmax=None, extent=None, xlabel="Distance (m)", ylabel="Depth (m)"):
    mappable =  ax.imshow(img, vmin=vmin, vmax=vmax, extent=extent)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return mappable

def save_fig_tight(path, fig):
    # fig.tight_layout(pad = 0, h_pad=0, w_pad=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close("all")

def save_4_5_fig(path, imgs, vmin, vmax, extent):

    fig, axs = plt.subplots(ncols=len(imgs), sharey=True)
    plot_ax(axs[0], imgs[0], vmin, vmax, extent)
    plot_ax(axs[1], imgs[1], vmin, vmax, extent, ylabel=None)
    plot_ax(axs[2], imgs[2], vmin, vmax, extent, ylabel=None)
    mappable = plot_ax(axs[3], imgs[3], vmin, vmax, extent, ylabel=None)
    if len(imgs) == 5:
        mappable = plot_ax(axs[4], imgs[4], vmin, vmax, extent, ylabel=None)

    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.83, 0.4, 0.012, 0.18])
    fig.colorbar(mappable, cax=cbar_ax)
    save_fig_tight(path, fig)

def mlp_uniform():
    from src.pinns.paper.dataset import PaperDataset
    path = Path("figures/mlp_uniform/")
    path.mkdir(exist_ok=True)
    snapshots = np.load("paper_data/uniform_wavefield.npz")["0000_E"]

    pinn_model = MLP(3, [128]*5, 1, nn.SiLU)
    pinn_model.load_state_dict(torch.load("checkpoints/PINN_model_best.ckp"))
    pinn_model = pinn_model.to(DEVICE)

    nn_model = MLP(3, [128]*5, 1, nn.ReLU)
    nn_model.load_state_dict(torch.load("checkpoints/NN_model_best.ckp"))
    nn_model = nn_model.to(DEVICE)

    train_indexes = [15, 25]
    train_dataset = PaperDataset(snapshots[train_indexes], t_offsets=train_indexes)
    scaler = train_dataset.scaler
    frame_15ns = train_dataset.get_frame(0)
    # save_image_and_colorbar(path / "field_15ns.png", frame_15ns[3], extent=[0, 20, 20, 0])
    frame_25ns = train_dataset.get_frame(1)
    # save_image_and_colorbar(path / "field_25ns.png", frame_25ns[3], extent=[0, 20, 20, 0])

    vmin_20 = -66.
    vmax_20 = 55.

    vmin_40 = -43.
    vmax_40 = 32
    
    val_indexes = [20]
    val_dataset = PaperDataset(snapshots[val_indexes], t_offsets=val_indexes, scaler=scaler)
    print("Validation dataset points:")
    val_dataset.print_info()
    frame_20ns = val_dataset.get_frame(0)
    # save_image_and_colorbar(path / "gt_20.png", frame_20ns[3], vmin=vmin_20, vmax=vmax_20, extent=[0, 20, 20, 0])
    
    test_indexes = [40]
    test_dataset = PaperDataset(snapshots[test_indexes], t_offsets=test_indexes, scaler=scaler)
    print("Test dataset points:")
    test_dataset.print_info()
    frame_40ns = test_dataset.get_frame(0)
    # save_image_and_colorbar(path / "gt_40.png", frame_40ns[3], vmin=vmin_40, vmax=vmax_40, extent=[0, 20, 20, 0])

    pred_indexes = list(range(15, 45, 1))
    pred_dataset = PaperDataset(snapshots[pred_indexes], pred_indexes, scaler)


    f_pinn = get_f(pinn_model, scaler)
    f_nn = get_f(nn_model, scaler)

    val_samples = torch.cat([val_dataset.data, val_dataset.labels[:, None]], dim=1).T.to(DEVICE)
    test_samples = torch.cat([test_dataset.data, test_dataset.labels[:, None]], dim=1).T.to(DEVICE)
    pred_samples = torch.cat([pred_dataset.data, pred_dataset.labels[:, None]], dim=1).T.to(DEVICE)

    pinn_predictions = predict_functional(f_pinn, pred_samples).reshape(-1, 200, 200)
    nn_predictions = predict_functional(f_nn, pred_samples).reshape(-1, 200, 200)

    np.savez(predictions_dir / "uniform_mlp.npz", **{"nn_15_45_1":nn_predictions, "pinn_15_45_1": pinn_predictions})

    pinn_predictions_20 = predict_functional(f_pinn, val_samples).reshape(200, 200)
    save_image_and_colorbar(path / "pinn_predictions_20.png", pinn_predictions_20, vmin=vmin_20, vmax=vmax_20, extent=[0, 20, 20, 0])
    save_image_and_colorbar(path / "diff_pinn_predictions_20.png", pinn_predictions_20 - frame_20ns[3], vmin=vmin_20, vmax=vmax_20, extent=[0, 20, 20, 0])
    pinn_predictions_40 = predict_functional(f_pinn, test_samples).reshape(200, 200)
    save_image_and_colorbar(path / "pinn_predictions_40.png", pinn_predictions_40, vmin=vmin_40, vmax=vmax_40, extent=[0, 20, 20, 0])
    save_image_and_colorbar(path / "diff_pinn_predictions_40.png", pinn_predictions_40 - frame_40ns[3], vmin=vmin_40, vmax=vmax_40, extent=[0, 20, 20, 0])


    nn_predictions_20 = predict_functional(f_nn, val_samples).reshape(200, 200)
    save_image_and_colorbar(path / "nn_predictions_20.png", nn_predictions_20, vmin=vmin_20, vmax=vmax_20, extent=[0, 20, 20, 0])
    nn_predictions_40 = predict_functional(f_nn, test_samples).reshape(200, 200)
    save_image_and_colorbar(path / "nn_predictions_40.png", nn_predictions_40, vmin=vmin_40, vmax=vmax_40, extent=[0, 20, 20, 0])

    save_4_5_fig(path / "final_20.png", [frame_20ns[3], pinn_predictions_20, nn_predictions_20, pinn_predictions_20 - frame_20ns[3]], vmin_20, vmax_20, extent=[0, 20, 20, 0])
    save_4_5_fig(path / "final_40.png", [frame_40ns[3], pinn_predictions_40, nn_predictions_40, pinn_predictions_40 - frame_40ns[3]], vmin_40, vmax_40, extent=[0, 20, 20, 0])

from src.pinns.paper.dataset import MyNormalizer
def save_figures(figures_path: str|Path, 
                 model_path: str|Path,
                 snapshots_path: str|Path,
                 geometry_path: str|Path,
                 train_indexes: list[int],
                 pred_indexes: list[int],
                 activation: nn.Module = nn.SiLU,
                 extent: list[float] = [0, 20, 20, 0], 
                 net_size: int = 256,
                 normalizer_class: type = MyNormalizer):
    """
    Loads snapshots, geometry and model, then saves ground truth and predicted E values.

    Parameters
    ----------
    figures_path : str | Path

    model_path : str | Path
        
    snapshots_path : str | Path
        
    geometry_path : str | Path
        
    train_indexes : list[int]
        
    val_indexes : list[int]
        
    test_indexes : list[int]
        
    """
    figures_path = Path(figures_path)
    figures_path.mkdir(exist_ok=True, parents=True)

    snapshots = np.load(snapshots_path)["00000_E"]

    geometry = np.load(geometry_path)

    save_image_and_colorbar(figures_path / "geometry.png", geometry[0], extent=extent)

    model = MLP(3, [net_size]*5, 1, activation)
    model.load_state_dict(torch.load(model_path))
    model = model.to(DEVICE)

    from src.pinns.rail_sample_mlp import PaperDataset
    # from src.pinns.paper.dataset import PaperDataset

    train_dataset = PaperDataset(snapshots[train_indexes], t_offsets=train_indexes, scaler_class=normalizer_class)
    print("Train dataset:")
    train_dataset.print_info()
    scaler = train_dataset.scaler

    f_pinn = get_f(model, scaler)

    pred_dataset = PaperDataset(snapshots[pred_indexes], t_offsets=pred_indexes, scaler=scaler)
    
    pred_samples = torch.cat([pred_dataset.data, pred_dataset.labels[:, None]], dim=1).T.to(DEVICE)

    img_size = snapshots.shape[1:]
    model_predictions = predict_functional(f_pinn, pred_samples).reshape(-1, *img_size)

    labels = []
    for i, pred in enumerate(tqdm(model_predictions)):
        label = pred_dataset.get_frame(i)[3]
        labels.append(label)
        vmin = label.min()
        vmax = label.max()

        save_image_and_colorbar(figures_path / ("gt_" + str(pred_indexes[i]) + "ns.png"), label, vmin=vmin, vmax=vmax, extent=extent)
        save_image_and_colorbar(figures_path / ("pred_" + str(pred_indexes[i]) + "ns.png"), pred, vmin=vmin, vmax=vmax, extent=extent)
        save_image_and_colorbar(figures_path / ("diff_pred_" + str(pred_indexes[i]) + "ns.png"), pred - label, vmin=vmin, vmax=vmax, extent=extent)

    plt.close("all")

    labels = np.asarray(labels)
    return model_predictions, labels

def mlp_2layer():
    snapshots_path = "paper_data/2layer_wavefield.npz"
    geometry_path = "paper_data/2layer_geometry.npy"

    pred_indexes = list(range(15, 50, 1))

    pinn_0_pred, labels = save_figures("figures/mlp_2layer/0/PINN", 
                 "results/two_layer_0/PINN_model_best.ckp", 
                 snapshots_path, geometry_path,
                 [15, 19, 23, 27, 31, 35], pred_indexes)


    nn_0_pred, _ = save_figures("figures/mlp_2layer/0/NN", 
                 "results/two_layer_0/NN_model_best.ckp", 
                 snapshots_path, geometry_path,
                 [15, 19, 23, 27, 31, 35], pred_indexes, nn.ReLU)
    
    vmin = labels[0].min()
    vmax = labels[0].max()
    save_4_5_fig("figures/mlp_2layer/0/final_45.png", [labels[0], pinn_0_pred[0], nn_0_pred[0], pinn_0_pred[0] - labels[0]], vmin, vmax, [0, 20, 20, 0])
    

    pinn_1_pred, _ = save_figures("figures/mlp_2layer/1/PINN", 
                 "results/two_layer_1/PINN_model_best.ckp", 
                 snapshots_path, geometry_path,
                 [15, 19, 23], pred_indexes)
    nn_1_pred, _ = save_figures("figures/mlp_2layer/1/NN", 
                 "results/two_layer_1/NN_model_best.ckp", 
                 snapshots_path, geometry_path,
                 [15, 19, 23], pred_indexes, nn.ReLU)

    save_4_5_fig("figures/mlp_2layer/1/final_45.png", [labels[0], pinn_1_pred[0], nn_1_pred[0], pinn_1_pred[0] - labels[0]], vmin, vmax, [0, 20, 20, 0])
    
    
    pinn_2_pred, _ = save_figures("figures/mlp_2layer/2/PINN", 
                 "results/two_layer_2/PINN_model_best.ckp", 
                 snapshots_path, geometry_path,
                 [15, 19, 23], pred_indexes)
    nn_2_pred, _ = save_figures("figures/mlp_2layer/2/NN", 
                 "results/two_layer_2/NN_model_best.ckp", 
                 snapshots_path, geometry_path,
                 [15, 19, 23], pred_indexes, nn.ReLU)
    
    save_4_5_fig("figures/mlp_2layer/2/final_45.png", [labels[0], pinn_2_pred[0], nn_2_pred[0], pinn_2_pred[0] - labels[0]], vmin, vmax, [0, 20, 20, 0])

    data_to_save = {
        "nn0_15_50_1": nn_0_pred,
        "nn1_15_50_1": nn_1_pred,
        "nn2_15_50_1": nn_2_pred,
        "pinn0_15_50_1": pinn_0_pred,
        "pinn1_15_50_1": pinn_1_pred,
        "pinn2_15_50_1": pinn_2_pred,
    }
    
    np.savez(predictions_dir / "mlp_2layer.npz", **data_to_save)


def mlp_rail_sample():
    figures_path = Path("figures/mlp_rail_sample/")
    snapshots_path = "munnezza/output/scan_00000/snapshots.npz"
    geometry_path = "munnezza/output/scan_00000/scan_00000_geometry.npy"

    from src.pinns.rail_sample_mlp import MyNormalizer, PowNormalizer

    pred_indexes = [35, 85, 145, 195]
    pred_indexes = list(range(20, 200, 5))

    pinn_preds, labels = save_figures(figures_path / "1ns_all", 
                 "results/rail_sample_1ns_all/PINN_model_best.ckp", 
                 snapshots_path, geometry_path,
                 list(range(20, 200, 10)), pred_indexes, extent=[0, 1.5, 1.7, 0], net_size=512, normalizer_class=MyNormalizer)

    nn_preds, labels = save_figures(figures_path / "02ns_all_pow", 
                 "results/rail_sample_02ns_all_pow/NN_model_best.ckp", 
                 snapshots_path, geometry_path,
                 list(range(20, 200, 2)), pred_indexes, nn.ReLU, extent=[0, 1.5, 1.7, 0], net_size=512, normalizer_class=PowNormalizer)
    

    np.savez(predictions_dir / "mlp_rail_sample.npz", **{"nn_2_20_05":nn_preds, "pinn_2_20_05": pinn_preds})

    for i, (pinn_pred, nn_pred, label) in enumerate(zip(pinn_preds, nn_preds, labels)):
        vmin = label.min()
        vmax = label.max()
        print("label:", vmin, vmax)
        print("NN:", nn_pred.min(), nn_pred.max())
        print("PINN:", pinn_pred.min(), pinn_pred.max())
        save_4_5_fig(figures_path / f"final_{i}.png", [label, pinn_pred, nn_pred, pinn_pred - label, nn_pred - label],
                   vmin, vmax, [0, 1.5, 1.7, 0])
        

def time2image():
    figures_path = Path("figures/time2image/")
    figures_path.mkdir(parents=True, exist_ok=True)
    snapshots = np.load("munnezza/output/scan_00000/snapshots.npz")["00000_E"]

    model_checkpoint = Path("checkpoints/time2image_relu_warmup_1ns_20_100.ckp")
    NN_model = Time2Image([1, 64, 256, 512, 4608], [72, 64], cnn_layers=[1, 64, 64], activations=nn.ReLU)
    NN_model.load_state_dict(torch.load(model_checkpoint))
    NN_model = NN_model.to(DEVICE)

    PINN_model_checkpoint = Path("checkpoints/time2image_gelu_1ns_20_100.ckp")
    PINN_model = Time2Image([1, 64, 256, 512, 4608], [72, 64], cnn_layers=[1, 64, 64], activations=nn.GELU)
    PINN_model.load_state_dict(torch.load(PINN_model_checkpoint))
    PINN_model = PINN_model.to(DEVICE)

    times = np.load("munnezza/output/scan_00000/snapshots.npz")["00000_times"]
    times = torch.from_numpy(times).to(DEVICE, dtype=torch.float32)
    pred_times = times.unsqueeze(1)

    from src.pinns.time2image import get_f
    f_NN = get_f(NN_model, input_scale=1e8, output_scale=500.)
    f_PINN = get_f(PINN_model, input_scale=1e8, output_scale=500.)

    with torch.no_grad():
        preds_NN = f_NN(pred_times).cpu().numpy()
        preds_PINN = f_PINN(pred_times).cpu().numpy()

    np.savez(predictions_dir / "time2image.npz", **{"nn_all":preds_NN, "pinn_all": preds_PINN})

    from src.visualization.field import save_field_animation

    save_field_animation(preds_PINN[20:], "figures/time2image_PINN_preds.gif", interval=200)
    save_field_animation(preds_NN[20:], "figures/time2image_NN_preds.gif", interval=200)
    save_field_animation(snapshots[20:], "figures/time2image_ground_truth.gif", interval=200)


    from src.visualization.field import save_field_animation
    save_field_animation(preds_PINN, None)

    save_image_and_colorbar(figures_path / "gt_20.png", snapshots[20])
    save_image_and_colorbar(figures_path / "gt_100.png", snapshots[100])
    
    for i in [60, 65, 70, 120]:
        save_image_and_colorbar(figures_path / f"gt_{i}.png", snapshots[i])
        vmin, vmax = snapshots[i].min(), snapshots[i].max()

        save_image_and_colorbar(figures_path / f"PINN_{i}.png", preds_PINN[i], vmin=vmin, vmax=vmax, extent=[0, 1.5, 1.7, 0])
        save_image_and_colorbar(figures_path / f"NN_{i}.png", preds_NN[i], vmin=vmin, vmax=vmax, extent=[0, 1.5, 1.7, 0])

        save_image_and_colorbar(figures_path / f"diff_PINN_{i}.png", preds_PINN[i] - snapshots[i], vmin=vmin, vmax=vmax, extent=[0, 1.5, 1.7, 0])
        save_image_and_colorbar(figures_path / f"diff_NN_{i}.png", preds_NN[i] - snapshots[i], vmin=vmin, vmax=vmax, extent=[0, 1.5, 1.7, 0])

        save_4_5_fig(figures_path / f"final_{i}.png", [snapshots[i], preds_PINN[i], preds_NN[i], preds_PINN[i] - snapshots[i], preds_NN[i] - snapshots[i]],
                   vmin, vmax, [0, 1.5, 1.7, 0])


def wavefield1D():

    figures_path = Path("figures/time2sequence/")
    figures_path.mkdir(exist_ok=True, parents=True)

    T1 = 2.5
    T2 = 20
    WAVE_SPEED = 16

    def build_labels(t1, t2):
        snapshots = np.load("paper_data/uniform_wavefield.npz")["0000_E"]
        labels = np.zeros((2, 512), dtype=np.float32)
        wave = snapshots[30][100:, 100]
        wave /= -wave.min()
        wave = wave.squeeze()

        wave = np.interp(np.arange(200), np.arange(100) * (200 / len(wave)), wave)
        labels[0][int(t1*WAVE_SPEED):200 + int(t1*WAVE_SPEED)] = wave
        if t2 > 15:
            labels[1][int(t2*WAVE_SPEED): 200 + int(t2*WAVE_SPEED)] = wave[:192]
        else:
            labels[1][int(t2*WAVE_SPEED): 200 + int(t2*WAVE_SPEED)] = wave

        from scipy.ndimage import gaussian_filter1d
        labels[0] = gaussian_filter1d(labels[0], 2)
        labels[1] = gaussian_filter1d(labels[1], 2)

        return labels

    from src.pinns.time2sequence import MLP, Time2Sequence
    mlp_nn_model_path = Path(f"results/time2sequence/new/mlp/NN_512_200_{WAVE_SPEED}/model.ckp")
    mlp_pinn_model_path = Path(f"results/time2sequence/new/mlp/PINN_512_200_{WAVE_SPEED}/model.ckp")

    cnn_nn_model_path = Path(f"results/time2sequence/new/cnn/NN_512_200_{WAVE_SPEED}/model.ckp")
    cnn_pinn_model_path = Path(f"results/time2sequence/new/cnn/PINN_512_200_{WAVE_SPEED}/model.ckp")
    expanding_mlp_nn_model_path = Path(f"results/time2sequence/new/expanding_mlp/NN_512_200_{WAVE_SPEED}/model.ckp")
    expanding_mlp_pinn_model_path = Path(f"results/time2sequence/new/expanding_mlp/PINN_512_200_{WAVE_SPEED}/model.ckp")

    mlp_nn_model = MLP(2, [256]*5, 1, nn.SiLU)
    mlp_nn_model.load_state_dict(torch.load(mlp_nn_model_path))
    mlp_nn_model = mlp_nn_model.to(DEVICE)

    mlp_pinn_model = MLP(2, [256]*5, 1, nn.SiLU)
    mlp_pinn_model.load_state_dict(torch.load(mlp_pinn_model_path))
    mlp_pinn_model = mlp_pinn_model.to(DEVICE)

    cnn_nn_model = Time2Sequence(cnn_layers=[1, 64], activations=nn.SiLU)
    cnn_nn_model.load_state_dict(torch.load(cnn_nn_model_path))
    cnn_nn_model = cnn_nn_model.to(DEVICE)
    cnn_nn_model.gaussian_kernel = cnn_nn_model.gaussian_kernel.to(DEVICE)

    cnn_pinn_model = Time2Sequence(cnn_layers=[1, 64], activations=nn.SiLU)
    cnn_pinn_model.load_state_dict(torch.load(cnn_pinn_model_path))
    cnn_pinn_model = cnn_pinn_model.to(DEVICE)
    cnn_pinn_model.gaussian_kernel = cnn_pinn_model.gaussian_kernel.to(DEVICE)

    expanding_mlp_nn_model = MLP(1, [64, 256], 512, nn.SiLU)
    expanding_mlp_nn_model.load_state_dict(torch.load(expanding_mlp_nn_model_path))
    expanding_mlp_nn_model = expanding_mlp_nn_model.to(DEVICE)

    expanding_mlp_pinn_model = MLP(1, [64, 256], 512, nn.SiLU)
    expanding_mlp_pinn_model.load_state_dict(torch.load(expanding_mlp_pinn_model_path))
    expanding_mlp_pinn_model = expanding_mlp_pinn_model.to(DEVICE)

    #######
    # MLP #
    #######

    def f_mlp(model, x:torch.Tensor, t:torch.Tensor):
        x = x / 500
        t = t / 10

        input = torch.stack([x, t], dim=-1)
        out = model(input)
        
        return out.squeeze()
    
    def f_other(model, t: torch.Tensor):
        t = t / 10
        out = model(t)
        return out.squeeze()

    x_test = torch.arange(0., 512, 1.0)
    t_test = torch.arange(0., 25., 0.1)
    # t_test = torch.tensor([T1, T2])
    mlp_t_test, mlp_x_test = torch.meshgrid([t_test, x_test], indexing="ij")

    mlp_x_test = mlp_x_test.to(DEVICE)
    mlp_t_test = mlp_t_test.to(DEVICE)
    t_test = t_test.unsqueeze(1).to(DEVICE)

    mlp_nn_preds = f_mlp(mlp_nn_model, mlp_x_test, mlp_t_test).cpu().detach()
    mlp_pinn_preds = f_mlp(mlp_pinn_model, mlp_x_test, mlp_t_test).cpu().detach()

    # from time2sequence import show_wave_evolution
    # show_wave_evolution(mlp_pinn_preds, "figures/1D_mlp_pinn_preds.gif")
    

    cnn_nn_preds = f_other(cnn_nn_model, t_test).cpu().detach()
    cnn_pinn_preds = f_other(cnn_pinn_model, t_test).cpu().detach()

    expanding_mlp_nn_preds = f_other(expanding_mlp_nn_model, t_test).cpu().detach()
    expanding_mlp_pinn_preds = f_other(expanding_mlp_pinn_model, t_test).cpu().detach()
    labels = build_labels(T1, T2)

    data_to_save = {
        "mlp_nn_0_25_01": mlp_nn_preds,
        "mlp_pinn_0_25_01": mlp_pinn_preds,
        "cnn_nn_0_25_01": cnn_nn_preds,
        "cnn_pinn_0_25_01": cnn_pinn_preds,
        "empl_nn_0_25_01": expanding_mlp_nn_preds,
        "empl_pinn_0_25_01": expanding_mlp_pinn_preds,
    }
    np.savez(predictions_dir / "1D_wavefield.npz", **data_to_save)

    plt.imsave("figures/1D_mlp_pinn_preds.png", mlp_pinn_preds)
    plt.imsave("figures/1D_cnn_pinn_preds.png", cnn_pinn_preds)
    plt.imsave("figures/1D_emlp_pinn_preds.png", expanding_mlp_pinn_preds)

    def plot(path, label, nn_pred, pinn_pred):
        plt.figure(figsize=(10, 6))
        if label is not None:
            plt.plot(label, "black", label="ground truth")
        if nn_pred is not None:
            plt.plot(nn_pred, label="NN prediction")
        if pinn_pred is not None:
            plt.plot(pinn_pred, label="PINN prediction")
        ax = plt.gca()
        ax.set_ylim(-1, 0.8)
        plt.xlabel("Distance (m)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight", pad_inches=0.1)
        plt.close()

    # plot(figures_path / "train", None, train_labels[0], train_labels[1])
    plot(figures_path / "mlp0", labels[0], mlp_nn_preds[0], mlp_pinn_preds[0])
    plot(figures_path / "mlp1", labels[1], mlp_nn_preds[1], mlp_pinn_preds[1])
    plot(figures_path / "cnn0", labels[0], cnn_nn_preds[0], cnn_pinn_preds[0])
    plot(figures_path / "cnn1", labels[1], cnn_nn_preds[1], cnn_pinn_preds[1])
    plot(figures_path / "emlp0", labels[0], expanding_mlp_nn_preds[0], expanding_mlp_pinn_preds[0])
    plot(figures_path / "emlp1", labels[1], expanding_mlp_nn_preds[1], expanding_mlp_pinn_preds[1])


def geom2bscan():
    from src.dataset_creation.geom2bscan import load_dataset, split_dataset, filter_initial_wave, build_network

    figures_path = Path("figures/geom2bscan")
    figures_path.mkdir(exist_ok=True, parents=True)
    
    geometries, bscans, _ = load_dataset()
    train_data, test_data, train_labels, test_labels = split_dataset(geometries, bscans)
    train_labels, test_labels, median = filter_initial_wave(train_labels, test_labels)

    test_data1 = np.expand_dims(test_data[:, :, :, 0], -1)
    test_data2 = np.expand_dims(test_data[:, :, :, 1], -1)
    test_mask = test_labels

    model = build_network()
    model_path = "results/geom2bscan_filtered_dataset2/model_backup.h5"

    model.load_weights(model_path)
    # model.evaluate(x=[test_data1,test_data2], y=test_mask)

    vmin, vmax = test_mask.min(), test_mask.max()
    vmax = max(np.absolute(vmin), vmax)
    vmin = -vmax

    vmin = vmin / 3
    vmax = vmax / 3

    test_pred = model.predict([test_data1,test_data2])
    test_pred = np.asarray(test_pred)


    save_image_and_colorbar(figures_path / "median.png", median, xlabel="", ylabel="")

    print("Saving test set predictions...")
    for i, (e, s, p, l) in tqdm(enumerate(zip(test_data1, test_data2, test_pred, test_mask)), total=len(test_mask)):
        results_subfolder = figures_path / "figs" / str(i).zfill(4)
        results_subfolder.mkdir(exist_ok=True, parents=True)
        save_image_and_colorbar(results_subfolder / "epsilon_r.png", e.squeeze(), vmin=1, vmax=20, extent=[0, 1.5, 1.7, 0])
        save_image_and_colorbar(results_subfolder / "sigma.png", s.squeeze(), vmin=0, vmax=0.05, extent=[0, 1.5, 1.7, 0])
        save_image_and_colorbar(results_subfolder / "prediction.png", p.squeeze(), vmin=vmin, vmax=vmax, xlabel="", ylabel="")
        save_image_and_colorbar(results_subfolder / "label.png", l.squeeze(), vmin=vmin, vmax=vmax, xlabel="", ylabel="")
        save_image_and_colorbar(results_subfolder / "diff.png", p.squeeze() - l.squeeze(), vmin=vmin, vmax=vmax, xlabel="", ylabel="")


def wavefield_gif():
    from src.visualization.field import save_field_animation
    snapshots = np.load("munnezza/output/scan_00000/snapshots.npz")["00000_E"]
    save_field_animation(snapshots[20:199:2], "figures/rail_sample_field.gif")

def show_saved_predictions(predictions_dir: Path):

    from src.visualization.field import save_field_animation

    files = list(predictions_dir.glob("*.npz"))
    for file_path in files:
        experiment = file_path.name
        print("**************")
        print(experiment.upper())
        print("**************")
        data = np.load(file_path)

        for model in data.keys():
            pred = data[model]
            print(model, "-->", pred.shape)
            
            if pred.ndim == 2:
                plt.imshow(pred, vmin=-2, vmax=2)
                plt.show()
            else:
                save_field_animation(pred, None)

if __name__ == "__main__":

    import matplotlib.pylab as pylab
    size = "large"

    # params = {'legend.fontsize': size,
    #         'figure.figsize': (16, 12),
    #         'axes.labelsize': size,
    #         'axes.titlesize': size,
    #         'xtick.labelsize': size,
    #         'ytick.labelsize': size}
    # pylab.rcParams.update(params)

    predictions_dir = Path("predictions/")

    # geom2bscan()
    # mlp_uniform()
    # mlp_2layer()
    # mlp_rail_sample()
    # time2image()
    # wavefield1D()

    show_saved_predictions(predictions_dir)