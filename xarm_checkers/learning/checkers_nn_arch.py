import torch
import torch.nn as nn
from typing import Tuple

class CheckersNN(nn.Module):

    def __init__(self, args):
        self.args = args
        super(CheckersNN, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.top_conv_layer = nn.Conv2d(self.args.num_channels, 256, 3, padding=1)
        self.top_batch_norm = nn.BatchNorm2d(256)
        # Make unique layers for each residual block, we can then index the blocks to specify which residual layer
        # to use
        self.residual_pieces = nn.ModuleList([nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256)
        ) for _ in range(args.residual_block_count)])

        self.policy_conv_layer = nn.Conv2d(256, 2, 1)
        self.policy_batch_norm = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2*8*8, 32*4)

        self.value_conv_layer = nn.Conv2d(256, 1, 1)
        self.value_batch_norm = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1*8*8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_layer(x)
        for i in range(self.args.residual_block_count):
            x = self.residual_layer(x, i)
        pi = self.policy_head(x)
        v = self.value_head(x)
        return pi, v

    def conv_layer(self, x: torch.Tensor) -> torch.Tensor:
        """
        The convolutional layer of the network
        """
        x = self.top_conv_layer(x)
        x = self.top_batch_norm(x)
        x = self.relu(x)
        return x

    def residual_layer(self, x: torch.Tensor, residual_layer_index: int) -> torch.Tensor:
        """
        Use a residual layer, given the index of the layer
        """
        residual = x
        out = self.residual_pieces[residual_layer_index](x)
        out += residual
        out = self.relu(out)
        return out

    def policy_head(self, x: torch.Tensor) -> torch.Tensor:
        """
        Policy head for the network
        """
        x = self.policy_conv_layer(x)
        x = self.policy_batch_norm(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.policy_fc(x)
        return x

    def value_head(self, x: torch.Tensor) -> torch.Tensor:
        """
        Value head for the network
        """
        x = self.value_conv_layer(x)
        x = self.value_batch_norm(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.value_fc1(x)
        x = self.relu(x)
        x = self.value_fc2(x)
        return self.tanh(x)
