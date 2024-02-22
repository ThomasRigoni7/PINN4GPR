import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.visualization.misc import save_field_animation
from models import Time2Image

DEVICE = "cuda:0"
EPOCHS = 5000


def time2Image():

    snapshots = np.load("paper_data/uniform_wavefield.npz")["0000_E"]

    snapshots = torch.from_numpy(snapshots).to(DEVICE, dtype=torch.float32)
    times = np.load("paper_data/uniform_wavefield.npz")["0000_times"]
    times = torch.from_numpy(times).to(DEVICE, dtype=torch.float32)

    train_set = snapshots[10:50]
    save_field_animation(train_set.cpu().numpy(), "figures/time2image_label.gif")
    train_times = times[10:50]

    model = Time2Image([1, 64, 256, 512, 2500], [50, 50], cnn_layers=[1, 64, 64])

    model = model.to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.001)


    t = (train_times * 1e8).unsqueeze(1)
    print(t.shape)


    for e in tqdm(range(EPOCHS)):
        optimizer.zero_grad()
        
        prediction = model(t)
        prediction *= 500
        loss = loss_fn(prediction.squeeze(), train_set)
        loss.backward()
        optimizer.step()
        

    preds = []
    for t, snap in zip(train_times, train_set):
        optimizer.zero_grad()
        
        t = (t * 1e8).unsqueeze(0)
        prediction = model(t)
        prediction *= 500
        loss = loss_fn(prediction.squeeze(), snap)
        print(loss)
        preds.append(prediction.squeeze().cpu().detach().numpy())
    
    save_field_animation(preds, "figures/time2image.gif")

if __name__ == "__main__":
    time2Image()