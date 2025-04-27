import torch
import torch.nn as nn
from .plinear import PoincareLinear 
from manifolds import poincareball_factory
import torch.nn.functional as F
import math

class HypLayer(nn.Module):
    def __init__(
        self,
        features: int,
        act,
        conv,
        manifold,
        c
    ):
        super(HypLayer, self).__init__()
        self.features = features
        self.c = c
        self.act = act
        self.conv = conv
        if self.act:
            self.act = act()
        self.manifold = manifold
        if self.conv:
            self.conv = conv
        self.norm = nn.BatchNorm2d(features, momentum=0.1)

    def forward(self, x: torch.Tensor):
        x_space = x[..., 1:]
        x_space = x_space.permute(0, 3, 2, 1)
        if self.act: 
            x_space = self.act(x_space)
        
        elif self.conv:
            x_space = self.conv(x_space)

        else:
            x_space = self.norm(x_space)
        x_space = x_space.permute(0, 2, 3, 1)
        x_time = ((x_space ** 2).sum(-1, keepdims=True) + self.c).clamp_min(1e-6).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        return x

class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(
        self,
        in_features,
        out_features,
        manifold,
        c,
        stride = 1,
    ):
        super(ResidualBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.stride = stride
        self.c = c
        self.manifold = manifold
        self.x_i = nn.Parameter(torch.tensor(1.0)) # can be size of 1 parameter for all x
        self.y_i = nn.Parameter(torch.tensor(1.0)) # but can also be size of n for entry of x
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.conv1 = HypLayer(
            out_features,
            act = None,
            conv = nn.Conv2d(in_features, out_features, (3,3), stride=self.stride, padding=1),
            manifold = self.manifold,
            c = self.c
        )

        self.ln1 = HypLayer(
            out_features,
            act = None,
            conv = None,
            manifold = self.manifold,
            c = self.c
        )
        self.conv2 = HypLayer(
            out_features,
            act = None,
            conv = nn.Conv2d(out_features, out_features, (3,3), stride=1, padding=1),
            manifold = self.manifold,
            c = self.c
        )

        self.ln2 = HypLayer(
            out_features,
            act = None,
            conv = None,
            manifold = self.manifold,
            c = self.c
        )
        self.act1 = HypLayer(
            out_features,
            act = nn.ReLU,
            conv = None,
            manifold = self.manifold,
            c = self.c
        )
        self.act2 = HypLayer(
            out_features,
            act = nn.ReLU,
            conv = None,
            manifold = self.manifold,
            c = self.c
        )
        self.downsample = nn.Sequential()
        if stride != 1:
            self.downsample = HypLayer(
                    out_features,
                    act = None,
                    conv = nn.Conv2d(in_features, out_features, (1,1), stride=stride, padding=0),
                    manifold = self.manifold,
                    c = self.c
                )

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.ln2(x)
        residual = self.downsample(residual)
        # Skip connection with LRN
        ave = x * self.x_i + residual * self.y_i
        denom = (-self.manifold.inner(ave, ave, keepdim=True)).abs().clamp_min(1e-8).sqrt() * self.c.sqrt()
        x = ave / denom
        x_space = self.scale * x[..., 1:]
        x_time = ((x_space ** 2).sum(-1, keepdims=True) + self.c).clamp_min(1e-6).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)

        x = self.act2(x)

        return x    