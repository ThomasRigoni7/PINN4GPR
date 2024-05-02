import torch
from torchvision.transforms import Pad
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.visualization.field import save_field_animation
from src.pinns.models import RecurrentCNN, HiddenStateRecurrentCNN

DEVICE = "cuda:3"

class SequenceDataset(Dataset):
    def __init__(self, snapshots: torch.Tensor, max_seq_len: int = 5) -> None:
        pass
        

"""
HIDDEN STATE
"""

def predict_sequence_hidden_state(model: HiddenStateRecurrentCNN, snapshots: torch.Tensor, geometry: torch.Tensor, initial_hidden_state: torch.Tensor, seq_len: int, use_predictions: bool = False) -> torch.Tensor:
    padding = Pad((3, 2), padding_mode="reflect")

    snapshots = padding(snapshots)

    initial_snapshot = snapshots[0]
    initial_snapshot = initial_snapshot.unsqueeze(0)

    # initial_snapshot = padding(initial_snapshot)
    geometry = padding(geometry)
    
    def f(e_field, hidden, delta_t):
        e = e_field / 500
        dt = delta_t * 1e8
        pred, hidden = model(geometry, e, hidden, dt)
        pred = pred * 500
        return pred, hidden

    delta_t = torch.tensor([5e-10], dtype=torch.float32, device=DEVICE, requires_grad=True)
    predictions = []
    preds, hidden = f(initial_snapshot, initial_hidden_state, delta_t)
    predictions.append(preds.squeeze())
    for i in range(1, seq_len):
        if not use_predictions:
            preds, hidden = f(snapshots[i].unsqueeze(0), hidden, delta_t)
        else:
            preds, hidden = f(preds, hidden, delta_t)
        # fig, axs = plt.subplots(ncols=2)
        # axs[0].imshow(preds.cpu().detach().squeeze())
        # axs[1].imshow(preds.cpu().detach().squeeze())
        # plt.show()
        predictions.append(preds.squeeze())
    
    predictions = torch.stack(predictions)
    predictions = predictions[:, 2:-2, 3:-3]
    return predictions

def train_hidden_state(model: HiddenStateRecurrentCNN, snapshots: torch.Tensor, geometry: torch.Tensor, initial_hidden_state: torch.Tensor, sequence_len: int = -1):
    
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    loss_fn = torch.nn.MSELoss()

    for i in tqdm(range(1000)):
        optimizer.zero_grad()
        predictions = predict_sequence_hidden_state(model, snapshots, geometry, initial_hidden_state, len(snapshots) - 1)
        loss = loss_fn(predictions[:, ], snapshots[1:])
        print(loss.item())
        loss.backward()
        optimizer.step()

    return model

def recurrentCNN_hidden_state():

    geometry = np.load("munnezza/output/scan_00000/scan_00000_geometry.npy")
    snapshots = np.load("munnezza/output/scan_00000/snapshots.npz")["00000_E"]

    geometry = block_reduce(geometry, block_size=(1, 3, 3), func=np.mean)
    geometry = torch.from_numpy(geometry[:2]).to(DEVICE)

    snapshots = torch.from_numpy(snapshots).to(DEVICE, dtype=torch.float32)
    times = np.load("munnezza/output/scan_00000/snapshots.npz")["00000_times"]
    times = torch.from_numpy(times).to(DEVICE, dtype=torch.float32).view(-1, 1)

    frame_start = 20
    frame_end = 100
    frame_step = 5
    train_set = snapshots[frame_start:frame_end:frame_step]
    save_field_animation(train_set.cpu(), None)

    print("geometry shape:", geometry.shape)
    print("field shape:", snapshots.shape)
    print("times shape:", times.shape)
    # save_field_animation(snapshots.cpu(), None, bound_mult_factor=0.1)

    model = HiddenStateRecurrentCNN(down_channels=[16, 32, 64, 128, 128],
                         bottleneck_size=(9, 8),
                         fc_layers=[1, 64, 72],
                         up_cnn_layers=[512, 256, 128, 64, 32, 16])
    model = model.to(DEVICE)
    initial_hidden_state = torch.zeros((128, 9, 8), device=DEVICE)

    model = train_hidden_state(model, train_set, geometry, initial_hidden_state)


    with torch.no_grad():
        predictions1 = predict_sequence_hidden_state(model, train_set[0].unsqueeze(0), geometry, initial_hidden_state, 20, use_predictions=True)
        predictions2 = predict_sequence_hidden_state(model, train_set[0].unsqueeze(0), geometry, initial_hidden_state, 20, use_predictions=True)

    save_field_animation(train_set.cpu(), None)
    save_field_animation(predictions1.cpu(), None)
    save_field_animation(predictions2.cpu(), None)



"""
RECURRENT_DT
"""


def predict_sequence(model: RecurrentCNN, snapshots: torch.Tensor, geometry: torch.Tensor, initial_de_dt: torch.Tensor, seq_len: int, use_predictions: bool = False) -> torch.Tensor:
    padding = Pad((3, 2), padding_mode="reflect")

    snapshots = padding(snapshots)

    initial_snapshot = snapshots[0]
    initial_snapshot = initial_snapshot.unsqueeze(0)
    initial_de_dt = initial_de_dt.unsqueeze(0)

    # initial_snapshot = padding(initial_snapshot)
    geometry = padding(geometry)
    initial_de_dt = padding(initial_de_dt)
    
    def f(e_field, de_dt, delta_t):
        e = e_field / 500
        # de = de_dt / 1e12
        de = de_dt * 0
        dt = delta_t * 1e8
        # print("Stats:")
        # print("e:", e.max(), e.min())
        # print("de:", de.max(), de.min())
        # print("dt:", dt.max(), dt.min())
        pred = model(geometry, e, de, dt)
        pred = pred * 500
        return pred

    delta_t = torch.tensor([5e-10], dtype=torch.float32, device=DEVICE, requires_grad=True)
    predictions = []
    preds, dft = torch.func.jvp(f, (initial_snapshot, initial_de_dt, delta_t), (torch.zeros_like(initial_snapshot), torch.zeros_like(initial_de_dt), torch.ones_like(delta_t)))
    predictions.append(preds.squeeze())
    for i in range(1, seq_len):
        if not use_predictions:
            preds, dft = torch.func.jvp(f, (snapshots[i].unsqueeze(0), dft, delta_t), (torch.zeros_like(initial_snapshot), torch.zeros_like(initial_de_dt), torch.ones_like(delta_t)))
        else:
            preds, dft = torch.func.jvp(f, (preds, dft, delta_t), (torch.zeros_like(initial_snapshot), torch.zeros_like(initial_de_dt), torch.ones_like(delta_t)))
        # fig, axs = plt.subplots(ncols=2)
        # axs[0].imshow(preds.cpu().detach().squeeze())
        # axs[1].imshow(preds.cpu().detach().squeeze())
        # plt.show()
        predictions.append(preds.squeeze())
    
    predictions = torch.stack(predictions)
    predictions = predictions[:, 2:-2, 3:-3]
    return predictions

def train(model: RecurrentCNN, snapshots: torch.Tensor, geometry: torch.Tensor, initial_de_dt: torch.Tensor, sequence_len: int = -1):
    
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    loss_fn = torch.nn.MSELoss()

    for i in tqdm(range(1000)):
        optimizer.zero_grad()
        predictions = predict_sequence(model, snapshots, geometry, initial_de_dt, len(snapshots) - 1)
        loss = loss_fn(predictions[:, ], snapshots[1:])
        print(loss.item())
        loss.backward()
        optimizer.step()

    return model




def recurrentCNN():

    geometry = np.load("munnezza/output/scan_00000/scan_00000_geometry.npy")
    snapshots = np.load("munnezza/output/scan_00000/snapshots.npz")["00000_E"]

    geometry = block_reduce(geometry, block_size=(1, 3, 3), func=np.mean)
    geometry = torch.from_numpy(geometry[:2]).to(DEVICE)

    snapshots = torch.from_numpy(snapshots).to(DEVICE, dtype=torch.float32)
    times = np.load("munnezza/output/scan_00000/snapshots.npz")["00000_times"]
    times = torch.from_numpy(times).to(DEVICE, dtype=torch.float32).view(-1, 1)

    frame_start = 20
    frame_end = 100
    frame_step = 5
    train_set = snapshots[frame_start:frame_end:frame_step]
    save_field_animation(train_set.cpu(), None)

    print("geometry shape:", geometry.shape)
    print("field shape:", snapshots.shape)
    print("times shape:", times.shape)
    # save_field_animation(snapshots.cpu(), None, bound_mult_factor=0.1)
    initial_de_dt = (snapshots[frame_start] - snapshots[frame_start - 1]) * 1e10

    model = RecurrentCNN(down_channels=[16, 32, 64, 128, 128],
                         bottleneck_size=(9, 8),
                         fc_layers=[1, 64, 72],
                         up_cnn_layers=[512, 256, 128, 64, 32, 16])
    model = model.to(DEVICE)

    model = train(model, train_set, geometry, initial_de_dt)
    
    with torch.no_grad():
        predictions1 = predict_sequence(model, train_set[0].unsqueeze(0), geometry, initial_de_dt, 20, use_predictions=True)
        predictions2 = predict_sequence(model, train_set, geometry, initial_de_dt, 20, use_predictions=False)

    save_field_animation(train_set.cpu(), None)
    save_field_animation(predictions1.cpu(), "figures/recurrentCNN_predictions.gif")
    save_field_animation(predictions2.cpu(), None)


if __name__ == "__main__":
    recurrentCNN()
