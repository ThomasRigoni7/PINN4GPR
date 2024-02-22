import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.visualization.misc import save_field_animation
from models import Time2Image

DEVICE = "cuda:2"
EPOCHS = 5000


def time2Image():

    snapshots = np.load("munnezza/output/scan_00000/snapshots.npz")["00000_E"]
    # snapshots = np.load("paper_data/2layer_wavefield.npz")["00000_E"]
    #snapshots = np.load("paper_data/uniform_wavefield.npz")["0000_E"]

    snapshots = torch.from_numpy(snapshots).to(DEVICE, dtype=torch.float32)
    #times = np.load("paper_data/uniform_wavefield.npz")["0000_times"]
    times = np.load("munnezza/output/scan_00000/snapshots.npz")["00000_times"]
    times = torch.from_numpy(times).to(DEVICE, dtype=torch.float32)

    train_set = snapshots[20:150:10]
    save_field_animation(train_set.cpu().numpy(), "figures/time2image_rail_label.gif", interval=100)
    train_times = times[20:150:10]

    model = Time2Image([1, 64, 256, 512, 4608], [72, 64], cnn_layers=[1, 64, 64])

    model = model.to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.001)


    t = (train_times * 1e8).unsqueeze(1)
    print(t.shape)


    for e in tqdm(range(EPOCHS)):
        optimizer.zero_grad()
        
        prediction = model(t)
        prediction *= 500
        prediction = prediction.squeeze()
        prediction = prediction[:, 2:-2, 3:-3]

        loss = loss_fn(prediction, train_set)
        loss.backward()
        optimizer.step()

        if (e+1) % 100 == 0:
            print(loss.item())
        
    preds = model(t)
    preds = preds.squeeze()
    preds = preds[:, 2:-2, 3:-3]
    
    save_field_animation(preds.cpu().detach().numpy(), "figures/time2image_rail.gif")

if __name__ == "__main__":
    time2Image()