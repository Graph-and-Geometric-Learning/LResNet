"""Hyperbolic layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

from layers.att_layers import DenseAtt


def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
    if args.c is None:
        curvatures = nn.Parameter(torch.Tensor([1.0]))
    else:
        curvatures = torch.tensor([args.c])
        if not args.cuda == -1:
            curvatures = curvatures.to(args.device)
    return dims, acts, curvatures

class LorentzGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer. Code Adapted from Fully Hyperbolic Neural Networks by Chen et.al
    """

    def __init__(self, manifold, in_features, out_features, c_in, use_bias, dropout, use_att, local_agg, nonlin=None):
        super(LorentzGraphConvolution, self).__init__()
        self.linear = LorentzLinear(manifold, in_features, out_features, c_in, use_bias, dropout, nonlin=nonlin)
        self.agg = LorentzAgg(manifold, out_features, c_in, dropout, use_att, local_agg)

    def forward(self, input):
        x, adj = input
        h = self.linear(x)
        h = self.agg(h, adj)
        output = h, adj
        return output


class LorentzLinear(nn.Module):
    """
    Code Adapted from Fully Hyperbolic Neural Networks by Chen et.al
    """
    def __init__(self,
                 manifold,
                 in_features,
                 out_features,
                 c,
                 bias=True,
                 dropout=0.1,
                 scale=10,
                 fixscale=False,
                 nonlin=None):
        super(LorentzLinear, self).__init__()
        self.manifold = manifold
        self.nonlin = nonlin
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.bias = bias
        self.weight = nn.Linear(
            self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=not fixscale)

    def forward(self, x):
        if self.nonlin is not None:
            x = self.nonlin(x)
        x = self.weight(self.dropout(x))
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + (self.c.sqrt() + 0.5)
        scale = (time * time - self.c) / \
            (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        x = torch.cat([time, x_narrow * scale.clamp_min(1e-8).sqrt()], dim=-1)
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        step = self.in_features
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)


class LorentzAgg(Module):
    """
    Code Adapted from Fully Hyperbolic Neural Networks by Chen et.al
    """
    def __init__(self, manifold, in_features, c, dropout, use_att, local_agg):
        super(LorentzAgg, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.c = c
        self.use_att = use_att
        if self.use_att:
            self.key_linear = LorentzLinear(manifold, in_features, in_features, self.c)
            self.query_linear = LorentzLinear(manifold, in_features, in_features, self.c)
            self.bias = nn.Parameter(torch.zeros(()) + 20)
            self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_features))

    def forward(self, x, adj):
        if self.use_att:
            if self.local_agg:
                query = self.query_linear(x)
                key = self.key_linear(x)
                att_adj = 2 + 2 * self.manifold.cinner(query, key)
                att_adj = att_adj / self.scale + self.bias
                att_adj = torch.sigmoid(att_adj)
                att_adj = torch.mul(adj.to_dense(), att_adj)
                support_t = torch.matmul(att_adj, x)
            else:
                adj_att = self.att(x, adj)
                support_t = torch.matmul(adj_att, x)
        else:
            support_t = torch.spmm(adj, x)
        denom = (-self.manifold.inner(None, support_t, keepdim=True))
        denom = denom.abs().clamp_min(1e-8).sqrt()
        denom = denom / self.c.sqrt()
        output = support_t / denom
        return output

    def attention(self, x, adj):
        pass

class SkipLayer(nn.Module):
    """
    Skip connection HyboNet
    """
    def __init__(self, manifold, in_dim, out_dim, act, c, use_bias, dropout, use_att, local_agg, num_layers, nonlin=None):
        super(SkipLayer, self).__init__()
        self.linear1 = LorentzLinear(manifold, in_dim, out_dim, c, use_bias, dropout, nonlin=act)
        self.agg1 = LorentzAgg(manifold, out_dim, c, dropout, use_att, local_agg)
        # self.x_i = nn.Parameter(torch.tensor(1.0))
        # self.y_i = nn.Parameter(torch.tensor(1.0))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.manifold = manifold
        self.c = c
    def forward(self, input):
        x, adj = input
        h1 = self.linear1(x)
        h1 = self.agg1(h1, adj)
        if self.in_dim == self.out_dim:
            # Apply residual connection
            ave = x + h1
            denom = (-self.manifold.inner(ave, ave, keepdim=True)).abs().clamp_min(1e-8).sqrt() * self.c.sqrt()
            ret = ave / denom
        else:
            ret = h1
        output = (ret, adj)
        return output