import torch.nn as nn
import torch

from LRN.LRN import ResidualBlock, MidpointInputBloack, MidpointGlobalAvgPool2d, MidpointMLR
from manifolds.lorentz_manifold import Lorentz

__all__ = ["resnet10", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


class ResNet(nn.Module):
    """ Implementation of ResNet models on manifolds. """

    def __init__(
        self,
        block,
        num_blocks,
        manifold,
        img_dim=[3,32,32],
        embed_dim=512,
        num_classes=100,
        bias=True,
        remove_linear=False,
    ):
        super(ResNet, self).__init__()

        self.img_dim = img_dim[0]
        self.in_channels = 64
        self.conv3_dim = 128
        self.conv4_dim = 256
        self.embed_dim = embed_dim

        self.bias = bias
        self.block = block

        self.c = torch.Tensor([0.1]).cuda()
        self.manifold = manifold

        self.conv1 = self._get_inConv()
        self.conv2_x = self._make_layer(block, out_channels=self.in_channels, num_blocks=num_blocks[0], stride=1)
        self.conv3_x = self._make_layer(block, out_channels=self.conv3_dim, num_blocks=num_blocks[1], stride=2)
        self.conv4_x = self._make_layer(block, out_channels=self.conv4_dim, num_blocks=num_blocks[2], stride=2)
        self.conv5_x = self._make_layer(block, out_channels=self.embed_dim, num_blocks=num_blocks[3], stride=2)
        self.avg_pool = self._get_GlobalAveragePooling()

        if remove_linear:
            self.predictor = None
        else:
            self.predictor = self._get_predictor(self.embed_dim*block.expansion, num_classes)

    def forward(self, x):
        out = self.conv1(x)

        out_1 = self.conv2_x(out)
        out_2 = self.conv3_x(out_1)
        out_3 = self.conv4_x(out_2)
        out_4 = self.conv5_x(out_3)
        out = self.avg_pool(out_4)
        out = out.view(out.size(0), -1)

        if self.predictor is not None:
            out = self.predictor(out)
        return out

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            if type(self.manifold) is Lorentz:
                self.manifold = Lorentz(self.c)
                layers.append(
                    block(
                        self.in_channels,
                        out_channels,
                        self.manifold,
                        self.c,
                        stride,
                    )
                )
            else:
                raise RuntimeError(
                    f"Manifold {type(self.manifold)} not supported in ResNet."
                )

            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def _get_inConv(self):
        if type(self.manifold) is Lorentz:
            return MidpointInputBloack(
                self.manifold,
                self.img_dim,
                self.in_channels,
                self.c
            )
        else:
            raise RuntimeError(
                f"Manifold {type(self.manifold)} not supported in ResNet."
            )

    def _get_predictor(self, in_features, num_classes):
        if type(self.manifold) is Lorentz:
            return MidpointMLR(self.manifold, in_features, num_classes, self.c)
        else:
            raise RuntimeError(f"Manifold {type(self.manifold)} not supported in ResNet.")

    def _get_GlobalAveragePooling(self):
        if type(self.manifold) is Lorentz:
            return MidpointGlobalAvgPool2d(self.manifold, self.c, keep_dim=True)
        else:
            raise RuntimeError(f"Manifold {type(self.manifold)} not supported in ResNet.")

#################################################
#       LRN
#################################################

def lrn_resnet18(k = 0.1, learn_k = False, manifold = None, **kwargs):
    """Constructs a ResNet-18 model"""
    if not manifold:
        manifold = Lorentz()
    model = ResNet(ResidualBlock, [2,2,2,2], manifold, **kwargs)
    return model