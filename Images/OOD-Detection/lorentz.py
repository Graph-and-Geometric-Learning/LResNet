import torch
from manifolds import Lorentz
from LRN.LRN import *

class LRNResNet(nn.Module):
    """Lorentz Residual Networks
    """

    def __init__(
        self,
        classes: int,
        channel_dims: list[int],
        depths: list[int],
        act_layer = None,
        c: float = 0.1,
    ):
        super(LRNResNet, self).__init__()
        self.classes = classes
        self.channel_dims = channel_dims
        self.depths = depths
        self.act_layer = act_layer
        self.c = torch.tensor(c).cuda()
        self.manifold = Lorentz(k = self.c)

        self.conv = HypLayer(
            0,
            act = None,
            conv = nn.Conv2d(3, channel_dims[0], (3,3), stride=1, padding=1),
            manifold = self.manifold,
            c = self.c
        )
        self.ln = HypLayer(
            channel_dims[0], None, None, self.manifold, self.c
        )

        self.act = HypLayer(
            0,
            act = nn.ReLU,
            conv=None,
            manifold = self.manifold,
            c = self.c
        )

        self.group1 = self._make_group(
            in_features=channel_dims[0],
            out_features=channel_dims[0],
            depth=depths[0],
        )

        self.group2 = self._make_group(
            in_features=channel_dims[0],
            out_features=channel_dims[1],
            depth=depths[1],
            stride=2,
        )

        self.group3 = self._make_group(
            in_features=channel_dims[1],
            out_features=channel_dims[2],
            depth=depths[2],
            stride=2,
        )

        self.avg_pool = nn.AvgPool2d(8)

        self.linear_ball = poincareball_factory(
            c=self.c, custom_autograd=True, learnable=False
        )
        self.fc = PoincareLinear(
            in_features=channel_dims[2],
            out_features=classes,
            ball=self.linear_ball,
        )

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1) #channel last
        x = torch.cat([torch.ones_like(x)[..., 0:1], x], dim=-1)
        x = self.manifold.expmap0(x)
        x = self.conv(x)
        x = self.ln(x)
        x = self.act(x)
        
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)

        #apply pooling layer
        x_space = x[..., 1:]
        x_space = x_space.permute(0, 3, 2, 1)
        x_space = self.avg_pool(x_space)
        x_space = x_space.permute(0, 2, 3, 1)
        x_time = ((x_space ** 2).sum(-1, keepdims=True) + self.c).clamp_min(1e-6).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        #map to poincare ball model for mlr
        x = self.linear_ball.logmap0(self.manifold.lorentz_to_poincare(x))
        # x = x.permute(0, 3, 2, 1)
        # x = self.avg_pool(x)
        # # x = x.view(x.size(0), -1)
        # x = x.permute(0, 2, 3, 1)
        x = self.fc(x.squeeze())
        return x

    def _make_group(
        self,
        in_features,
        out_features,
        depth: int,
        stride: int = 1,
    ):
        layers = [
            ResidualBlock(
                in_features=in_features,
                out_features=out_features,
                manifold=self.manifold,
                c=self.c,
                stride=stride,
            )
        ]

        for _ in range(1, depth):
            layers.append(
                ResidualBlock(
                    in_features=out_features,
                    out_features=out_features,
                    manifold=self.manifold,
                    c=self.c,
                )
            )

        return nn.Sequential(*layers)