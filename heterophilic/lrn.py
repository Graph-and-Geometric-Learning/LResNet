import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

from geoopt import ManifoldParameter

class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def foward(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output

class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def forward(self, input):
        x, adj = input
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs,adj

class LorentzDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, manifold, input_dim, output_dim, use_bias):
        super(LorentzDecoder, self).__init__(c)
        self.manifold = manifold
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.cls = ManifoldParameter(self.manifold.random_normal((output_dim, input_dim), std=1./math.sqrt(input_dim)), manifold=self.manifold)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        self.decode_adj = False
        self.c = c

    def forward(self, input):
        x, adj = input
        return (2 * self.c + 2 * self.manifold.cinner(x, self.cls)) + self.bias, adj

class HyboNet(Encoder):
    """
    HyboNet.
    """

    def __init__(self, c, manifold, acts, dims, args):
        super(HyboNet, self).__init__(c)
        self.manifold = manifold
        assert args.num_layers > 1
        hgc_layers = []
        for i in range(len(dims) - 2):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    LorentzGraphConvolution(
                            self.manifold, in_dim, out_dim, c, True, args.dropout, 0, None, nonlin=act if i != 0 else None
                    )
            )
        hgc_layers.append(LorentzDecoder(c, manifold, dims[-2], dims[-1], True))
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def forward(self, data):
        x = data.graph['node_feat']
        o = torch.zeros_like(x)
        x = torch.cat([o[:, 0:1], x], dim=1)
        x = self.manifold.expmap0(x)
        adj = data.graph['edge_index']
        return super(HyboNet, self).foward(x, adj)
    
class LorentzGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
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
        time = x.narrow(-1, 0, 1).sigmoid() * (self.scale.exp()).clamp_max(10) + (self.c.sqrt() + 0.5)
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
    Lorentz aggregation layer.
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

class SkipHGNN(Encoder):
    """
    Skip connection HyboNet
    """
    def __init__(self, c, manifold, acts, dims, args):
        super(SkipHGNN, self).__init__(c)
        self.manifold = manifold
        assert args.num_layers > 1
        hgc_layers = []
        for i in range(len(dims) - 2):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    SkipLayer(
                            self.manifold, in_dim, out_dim, c, True, args.dropout, 0, None, args.num_layers, nonlin=act if i != 0 else None
                            )
            )
        
        hgc_layers.append(LorentzDecoder(c, manifold, dims[-2], dims[-1], True))

        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def forward(self, data):
        x = data.graph['node_feat']
        o = torch.zeros_like(x)
        x = torch.cat([o[:, 0:1], x], dim=1)
        x = self.manifold.expmap0(x)
        adj = data.graph['edge_index']
        return super(SkipHGNN, self).foward(x, adj)
    
    
class SkipLayer(nn.Module):
    def __init__(self, manifold, in_dim, out_dim, c, use_bias, dropout, use_att, local_agg, num_layers, nonlin=None):
        super(SkipLayer, self).__init__()
        self.linear = LorentzLinear(manifold, in_dim, out_dim, c, use_bias, dropout, nonlin=nonlin)
        self.agg = LorentzAgg(manifold, out_dim, c, dropout, use_att, local_agg)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.x_i = nn.Parameter(torch.tensor(0.5)) # can be size of 1 parameter for all x
        self.manifold = manifold
        self.c = c
    def forward(self, input):
        x, adj = input
        h = self.linear(x)
        h = self.agg(h, adj)
        output = h
        #residual connection
        if self.in_dim == self.out_dim:
            ave = x * (self.x_i) + h
            denom = (-self.manifold.inner(ave, ave, keepdim=True)).abs().clamp_min(1e-8).sqrt() * self.c.sqrt()
            output = ave / denom
        
        ret = output, adj
        return ret