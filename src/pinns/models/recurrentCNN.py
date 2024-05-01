"""
This module contains implementations for two types of recurrent CNNs.

The first model, :class:`.RecurrentCNN`, implements the 
recurrency by requiring as inputs the previous electric field map and its
derivative w.r.t. time.

The second model, :class:`.HiddenStateRecurrentCNN`, uses
instead the electric field map and an hidden state.
"""


import torch
import torch.nn as nn

from src.pinns.models.base import MLP
from src.pinns.models.time2image import UpCNN
from src.pinns.models.geom2bscan import DownBlock

class RecurrentCNN(nn.Module):
    def __init__(self, down_channels: list[int], bottleneck_size: tuple[int, int], fc_layers: list[int], up_cnn_layers: list[int], activations=nn.ReLU):

        super().__init__()

        geometry_down_blocks = [DownBlock(2, down_channels[0]),
                       DownBlock(down_channels[0], down_channels[1]),
                       DownBlock(down_channels[1], down_channels[2]),
                       DownBlock(down_channels[2], down_channels[3]),
                       DownBlock(down_channels[3], down_channels[4])]
        self.geometry_encoder = nn.Sequential(*geometry_down_blocks)

        e_field_down_blocks = [DownBlock(1, down_channels[0]),
                       DownBlock(down_channels[0], down_channels[1]),
                       DownBlock(down_channels[1], down_channels[2]),
                       DownBlock(down_channels[2], down_channels[3]),
                       DownBlock(down_channels[3], down_channels[4])]
        self.e_field_encoder = nn.Sequential(*e_field_down_blocks)

        dt_down_blocks = [DownBlock(1, down_channels[0]),
                       DownBlock(down_channels[0], down_channels[1]),
                       DownBlock(down_channels[1], down_channels[2]),
                       DownBlock(down_channels[2], down_channels[3]),
                       DownBlock(down_channels[3], down_channels[4])]
        self.dt_encoder = nn.Sequential(*dt_down_blocks)

        # dxy_down_blocks = [DownBlock(1, channels[0]),
        #                DownBlock(channels[0], channels[1]),
        #                DownBlock(channels[1], channels[2]),
        #                DownBlock(channels[2], channels[3]),
        #                DownBlock(channels[3], channels[4])]
        # self.dxy_encoder = nn.Sequential(*dxy_down_blocks)

        self.bottleneck_size = bottleneck_size
        self.time_mlp = MLP(1, fc_layers[1:-1], fc_layers[-1], activations)
        self.time_1x1 = nn.Conv2d(1, down_channels[4], 1)
        self.up_cnn = UpCNN(up_cnn_layers, activations)
        # some kind of fusion


    def forward(self, geometry: torch.Tensor, e_field: torch.Tensor, de_dt: torch.Tensor, delta_t: torch.Tensor):
        if geometry.ndim == 3:
            time_shape = (1, *self.bottleneck_size)
            embedding_concat_dim = 0
        elif geometry.ndim == 4:
            time_shape = (-1, 1, *self.bottleneck_size)
            embedding_concat_dim = 1
        else:
            raise ValueError("Expected input with 3 or 4 dims, got ", geometry.ndim)

        geom_embeddings = self.geometry_encoder(geometry)
        e_embeddings = self.e_field_encoder(e_field)
        de_dt_embeddings = self.dt_encoder(de_dt)
        t_1d_emb = self.time_mlp(delta_t)

        t_1d_emb = t_1d_emb.view(time_shape)
        t_embeddings = self.time_1x1(t_1d_emb)
        # print("Geometry embeddings:", geom_embeddings.shape)
        # print("E field embeddings:", e_embeddings.shape)
        # print("dE/dt embeddings:", de_dt_embeddings.shape)
        # print("time embeddings:", t_1d_emb.shape)

        bottleneck_embeddings = torch.cat([geom_embeddings, e_embeddings, de_dt_embeddings, t_embeddings], dim=embedding_concat_dim)

        x = self.up_cnn(bottleneck_embeddings)
        return x
    

class HiddenStateRecurrentCNN(nn.Module):
    def __init__(self, down_channels: list[int], bottleneck_size: tuple[int, int], fc_layers: list[int], up_cnn_layers: list[int], activations=nn.ReLU):

        super().__init__()

        geometry_down_blocks = [DownBlock(2, down_channels[0]),
                       DownBlock(down_channels[0], down_channels[1]),
                       DownBlock(down_channels[1], down_channels[2]),
                       DownBlock(down_channels[2], down_channels[3]),
                       DownBlock(down_channels[3], down_channels[4])]
        self.geometry_encoder = nn.Sequential(*geometry_down_blocks)

        e_field_down_blocks = [DownBlock(1, down_channels[0]),
                       DownBlock(down_channels[0], down_channels[1]),
                       DownBlock(down_channels[1], down_channels[2]),
                       DownBlock(down_channels[2], down_channels[3]),
                       DownBlock(down_channels[3], down_channels[4])]
        self.e_field_encoder = nn.Sequential(*e_field_down_blocks)


        self.bottleneck_size = bottleneck_size
        self.time_mlp = MLP(1, fc_layers[1:-1], fc_layers[-1], activations)
        self.time_1x1 = nn.Conv2d(1, down_channels[4], 1)

        self.bottleneck2hidden = nn.Conv2d(up_cnn_layers[0], down_channels[-1], 3, padding=1)

        self.up_cnn = UpCNN(up_cnn_layers, activations)

        self.activation = activations()


    def forward(self, geometry: torch.Tensor, e_field: torch.Tensor, hidden_state: torch.Tensor, delta_t: torch.Tensor):
        if geometry.ndim == 3:
            time_shape = (1, *self.bottleneck_size)
            embedding_concat_dim = 0
        elif geometry.ndim == 4:
            time_shape = (-1, 1, *self.bottleneck_size)
            embedding_concat_dim = 1
        else:
            raise ValueError("Expected input with 3 or 4 dims, got ", geometry.ndim)

        geom_embeddings = self.geometry_encoder(geometry)
        e_embeddings = self.e_field_encoder(e_field)
        
        t_1d_emb = self.time_mlp(delta_t)

        t_1d_emb = t_1d_emb.view(time_shape)
        t_embeddings = self.time_1x1(t_1d_emb)

        bottleneck_embeddings = torch.cat([geom_embeddings, e_embeddings, hidden_state, t_embeddings], dim=embedding_concat_dim)
        hidden = self.bottleneck2hidden(bottleneck_embeddings)
        hidden = self.activation(hidden)

        # print()
        # print("FORWARD")
        # print("Geometry embeddings:", geom_embeddings.shape)
        # print("E field embeddings:", e_embeddings.shape)
        # print("hidden_state:", hidden_state.shape)
        # print("time embeddings:", t_1d_emb.shape)
        # print("new hidden:", hidden.shape)

        x = self.up_cnn(bottleneck_embeddings)
        return x, hidden

if __name__ == "__main__":
    geom = torch.randn((5, 2, 288, 256))
    e_field = torch.randn((5, 1, 288, 256))
    de_dt = torch.randn((5, 1, 288, 256))
    delta_t = torch.randn((5, 1))

    model = RecurrentCNN(down_channels=[16, 32, 64, 128, 128],
                         bottleneck_size=(9, 8),
                         fc_layers=[1, 64, 72],
                         up_cnn_layers=[512, 256, 128, 64, 32, 16])

    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters:", total_params)

    out = model(geom, e_field, de_dt, delta_t)
    print(out.shape)