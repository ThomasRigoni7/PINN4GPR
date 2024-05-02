import torch
import torch.nn as nn
from torchvision.transforms import GaussianBlur

class MLP(nn.Module):
    def __init__(self, num_inputs: int, hidden_layer_sizes: list[int], num_outputs: int, activation: nn.Module):
        super().__init__()
        if len(hidden_layer_sizes) < 1:
            raise Exception("MLP needs to have at least 1 hidden layer!")
        
        self.activation = activation()
        self.input_layer = nn.Linear(num_inputs, hidden_layer_sizes[0])
        self.hidden_layers = []
        for i in range(len(hidden_layer_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], num_outputs)

    def forward(self, x):
        if x.ndim == 1:
            x = x[None, :]
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x
