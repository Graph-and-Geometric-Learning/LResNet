"""Graph encoders."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import manifolds
from layers.att_layers import GraphAttentionLayer
import layers.hyp_layers as hyp_layers
from layers.layers import GraphConvolution, Linear, get_dim_act
import utils.math_utils as pmath
from manifolds.lmath import expmap0

from geoopt import ManifoldParameter


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output

class MLP(Encoder):
    """
    Multi-layer perceptron.
    """

    def __init__(self, c, args):
        super(MLP, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False


class HyboNet(Encoder):
    """
    HyboNet from Fully Hyperbolic Neural Networks by Chen et.al
    """

    def __init__(self, c, args):
        super(HyboNet, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)(c)
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        c = self.curvatures
        hgc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.LorentzGraphConvolution(
                            self.manifold, in_dim, out_dim, c, args.bias, args.dropout, args.use_att, args.local_agg, nonlin=act if i != 0 else None
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        o = torch.zeros_like(x)
        x = torch.cat([o[:, 0:1], x], dim=1)
        x = expmap0(x, k=self.c, dim=-1)
        return super(HyboNet, self).encode(x, adj)
    
class SkipHGNN(Encoder):
    """
    Skip connected HyboNet model
    """
    def __init__(self, c, args):
        super(SkipHGNN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)(c)
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        hgc_layers = []
        c = self.curvatures
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
            hyp_layers.SkipLayer(
                self.manifold, in_dim, out_dim, act, c, args.bias, args.dropout, args.use_att, args.local_agg, args.num_layers, nonlin=None
                        )
                )   
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        o = torch.zeros_like(x)
        x = torch.cat([o[:, 0:1], x], dim=1)
        x = expmap0(x, k=self.c, dim=-1)
        return super(SkipHGNN, self).encode(x, adj)
