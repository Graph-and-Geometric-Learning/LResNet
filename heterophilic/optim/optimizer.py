from geoopt import ManifoldParameter
import torch
from optim.radam import RiemannianAdam
class Optimizer(object):
    def __init__(self, model, euc_lr, hyp_lr, euc_weight_decay, hyp_weight_decay):
        euc_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and not isinstance(p, ManifoldParameter)]

        hyp_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and isinstance(p, ManifoldParameter)]
        
        optimizer_euc = torch.optim.Adam(euc_params, lr=euc_lr, weight_decay=euc_weight_decay)
        optimizer_hyp = RiemannianAdam(hyp_params, lr=hyp_lr, stabilize=10, weight_decay=hyp_weight_decay)
        self.optimizer = [optimizer_euc, optimizer_hyp]
    def step(self):
        for optimizer in self.optimizer:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizer:
            optimizer.zero_grad()