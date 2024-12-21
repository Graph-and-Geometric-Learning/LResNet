"""Graph decoders."""
import math
import manifolds
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.att_layers import GraphAttentionLayer
from layers.layers import GraphConvolution, Linear

from geoopt import ManifoldParameter


class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
            g = open("prob.txt", "w")
            for y in probs:
                g.write(str(torch.max(y).detach().numpy()))
                g.write("\n")
            g.close()    
        else:
            probs = self.cls.forward(x)
        return probs

class FermiDiracDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, args):
        super(FermiDiracDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)(c.item())
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.use_bias = args.bias
        self.cls = ManifoldParameter(self.manifold.random_normal((args.n_classes, args.dim), std=1./math.sqrt(args.dim)), manifold=self.manifold)
        if args.bias:
            self.bias = nn.Parameter(torch.zeros(args.n_classes))
        self.decode_adj = False
        self.c = c

    def decode(self, x, adj):
        return (2 * self.c + 2 * self.manifold.cinner(x, self.cls)) + self.bias



model2decoder = {
    'HyboNet': FermiDiracDecoder,
    'SkipHGNN': FermiDiracDecoder
}

