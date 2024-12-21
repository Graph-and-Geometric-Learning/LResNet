import torch
import torch.nn as nn
import numpy as np
import math

from manifolds import PoincareBall
"""
Implementation of Poincare MLR, adapted from 2023 ICCV  paper Poincare ResNet
"""

class PoincareLinear(nn.Module):
    """Poincare fully connected linear layer"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ball: PoincareBall,
        bias: bool = True,
        id_init: bool = True,
    ) -> None:
        super(PoincareLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ball = ball
        self.has_bias = bias
        self.id_init = id_init

        self.z = nn.Parameter(torch.empty(in_features, out_features))
        if self.has_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.id_init:
            self.z = nn.Parameter(
                1 / 2 * torch.eye(self.in_features, self.out_features)
            )
        else:
            nn.init.normal_(
                self.z, mean=0, std=(2 * self.in_features * self.out_features) ** -0.5
            )
        if self.has_bias:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ball.expmap0(x, dim = -1)

        y = self.ball.fully_connected(
            x=x,
            z=self.z,
            bias=self.bias,
        )
        logits = self.ball.logmap0(y, dim=-1)
        return logits