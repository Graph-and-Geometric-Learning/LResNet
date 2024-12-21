from geoopt import ManifoldParameter
from optim.radam import RiemannianAdam
import torch

def get_param_groups(model, weight_decay):
    no_decay = ['bias', 'scale']
    parameters = [{
        'params': [
            p for n, p in model.named_parameters()
            if p.requires_grad and not any(
                nd in n
                for nd in no_decay) and not isinstance(p, ManifoldParameter)
        ],
        'weight_decay':
        weight_decay
    }, {
        'params': [
            p for n, p in model.named_parameters() if p.requires_grad and any(
                nd in n
                for nd in no_decay) or isinstance(p, ManifoldParameter)
        ],
        'weight_decay':
        0.0
    }]
    return parameters

def select_optimizers(model, lr, weight_decay, lr_reduce_freq=5000, gamma=0.5):
    optimizer_grouped_parameters = get_param_groups(model, weight_decay)
    optimizer = None
    optimizer = RiemannianAdam(params=optimizer_grouped_parameters,
                                   lr=lr,
                                   stabilize=10)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=int(
    #                                                    lr_reduce_freq),
    #                                                gamma=float(gamma))
    return optimizer