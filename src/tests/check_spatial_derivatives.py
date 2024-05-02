import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from kornia.filters import spatial_gradient

DEVICE = "cuda:0"

snapshots = np.load("dataset_ascan/gprmax_output_files/scan_00001/snapshots.npz")["00000_E"]
snapshots = torch.from_numpy(snapshots).to(DEVICE, dtype=torch.float32)

image = snapshots[30]

spatial_gradients = spatial_gradient(image.unsqueeze(0).unsqueeze(0), mode="sobel", order=2, normalized=True)
# scale the spatial gradients so that they express the real measurements in meters.
# 1 cell is 6mm -> 2nd order gradients need to be multiplied by 1/(0.006)**2
spatial_gradients = spatial_gradients
print(spatial_gradients.shape)
dfxx_old = spatial_gradients[:, :, 0, :, :].squeeze()
dfxy_old = spatial_gradients[:, :, 1, :, :].squeeze()
dfyy_old = spatial_gradients[:, :, 2, :, :].squeeze()

spatial_gradients1 = spatial_gradient(image.unsqueeze(0).unsqueeze(0), mode="sobel", order=1, normalized=True)
dfx = spatial_gradients1[:, :, 0, :, :]
dfy = spatial_gradients1[:, :, 1, :, :]
dfxx = spatial_gradient(dfx, mode="sobel", order=1, normalized=True)[:, :, 0, :, :].squeeze()
dfyy = spatial_gradient(dfy, mode="sobel", order=1, normalized=True)[:, :, 1, :, :].squeeze()

sobel_y = -torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], dtype=torch.float32, device=DEVICE)
sobel_x = -torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], dtype=torch.float32, device=DEVICE)
print(sobel_y.shape)
sobel_dfy = F.conv2d(image.unsqueeze(0), sobel_y, stride=1, padding=1) / 8.
sobel_dfx = F.conv2d(image.unsqueeze(0), sobel_x, stride=1, padding=1) / 8.
sobel_dfyy = F.conv2d(sobel_dfy, sobel_y, stride=1, padding=1) / 8.
sobel_dfxx = F.conv2d(sobel_dfx, sobel_x, stride=1, padding=1) / 8.

cut_sobel_x = sobel_dfx.squeeze()[1:-1, 1:-1]
cut_sobel_y = sobel_dfy.squeeze()[1:-1, 1:-1]
cut_x = dfx.squeeze()[1:-1, 1:-1]
cut_y = dfy.squeeze()[1:-1, 1:-1]

cut_sobel_xx = sobel_dfxx.squeeze()[2:-2, 2:-2]
cut_sobel_yy = sobel_dfyy.squeeze()[2:-2, 2:-2]
cut_xx = dfxx.squeeze()[2:-2, 2:-2]
cut_yy = dfyy.squeeze()[2:-2, 2:-2]

print("Close dfx:", torch.allclose(cut_x, cut_sobel_x))
print("Close dfy:", torch.allclose(cut_y, cut_sobel_y))
print("Close dfxx:", torch.allclose(cut_xx, cut_sobel_xx))
print("Close dfyy:", torch.allclose(cut_yy, cut_sobel_yy))

fig, axs = plt.subplots(ncols=2)
axs[0].imshow(image.cpu().squeeze())
axs[1].imshow(dfx.cpu().squeeze())
plt.show()


fig, axs = plt.subplots(ncols=5)
axs[0].imshow(dfxx.cpu().squeeze())
axs[1].imshow(-dfxx_old.cpu().squeeze())
axs[2].imshow(dfyy.cpu().squeeze())
axs[3].imshow(-dfyy_old.cpu().squeeze())
axs[4].imshow((dfxx + dfxx_old).cpu().squeeze())
plt.show()

cut_xx_old = dfxx_old.squeeze()[2:-2, 2:-2]
cut_yy_old = dfyy_old.squeeze()[2:-2, 2:-2]

print("Close old dfxx:", torch.allclose(cut_xx, -cut_xx_old))
print("Close old dfyy:", torch.allclose(cut_yy, -cut_yy_old))