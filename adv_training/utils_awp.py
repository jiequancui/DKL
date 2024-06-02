import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
EPS = 1E-20

def dkl_loss(logits_nat, logits_adv, weight, alpha, beta):
    num_classes = logits_nat.size(1)
    delta_n = logits_nat.view(-1, num_classes, 1) - logits_nat.view(-1, 1, num_classes)
    delta_a = logits_adv.view(-1, num_classes, 1) - logits_adv.view(-1, 1, num_classes)
    
    loss_mse = 0.25 * (torch.pow(delta_n - delta_a, 2) * weight).sum() / logits_nat.size(0)
    loss_sce = -(F.softmax(logits_nat, dim=1).detach() * F.log_softmax(logits_adv, dim=-1)).sum(1).mean()
    return beta * loss_mse + alpha * loss_sce 


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class TradesAWP(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(TradesAWP, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv, inputs_clean, targets, weight, alpha, beta):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        logits_nat = self.proxy(inputs_clean)
        logits_adv = self.proxy(inputs_adv)
        loss_natural = F.cross_entropy(logits_nat, targets)

        loss_robust = dkl_loss(logits_nat, logits_adv, weight, alpha, beta)
        loss = - 1.0 * (loss_natural + loss_robust)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)




