import torch
import torch.nn as nn
import torch.nn.functional as F

from manifolds.geoopt.manifolds.stereographic import PoincareBall

from LRN.LRN import MidpointMLR
from manifolds.lorentz_manifold import Lorentz

from resnet import (
    lrn_resnet18
)

LRN_RESNET_MODEL = {
    18: lrn_resnet18
}

RESNET_MODEL = {
    "lrn" : LRN_RESNET_MODEL
}


MIDPOINT_DECODER = {
    'mlr' : MidpointMLR
}

class ResNetClassifier(nn.Module):
    """ Classifier based on ResNet encoder.
    """
    def __init__(self, 
            num_layers:int, 
            enc_type:str="lrn", 
            dec_type:str="lrn",
            enc_kwargs={},
            dec_kwargs={}
        ):
        super(ResNetClassifier, self).__init__()

        self.enc_type = enc_type
        self.dec_type = dec_type

        self.clip_r = dec_kwargs['clip_r']

        self.encoder = RESNET_MODEL[enc_type][num_layers](remove_linear=True, **enc_kwargs)
        self.enc_manifold = self.encoder.manifold

        self.dec_manifold = None
        dec_kwargs['embed_dim']*=self.encoder.block.expansion
        if dec_type == "lrn":
            k = torch.Tensor([0.1]).cuda()
            self.dec_manifold = Lorentz(k = k)
            self.decoder = MIDPOINT_DECODER[dec_kwargs['type']](self.dec_manifold, dec_kwargs['embed_dim'], dec_kwargs['num_classes'], k)
        else:
            raise RuntimeError(f"Decoder manifold {dec_type} not available...")
    
    def embed(self, x):
        x = self.encoder(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        

