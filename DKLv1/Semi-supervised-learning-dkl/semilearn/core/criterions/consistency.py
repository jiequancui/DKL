
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch 
import torch.nn as nn 
from torch.nn import functional as F

from .cross_entropy import ce_loss


def dkl_loss(logits_student, logits_teacher, temperature, alpha, beta, CLASS_PRIOR=None, GI=False, T2=100.0):
    _, NUM_CLASSES = logits_student.shape
    delta_n = (logits_teacher.view(-1, NUM_CLASSES, 1) - logits_teacher.view(-1, 1, NUM_CLASSES))
    delta_a = (logits_student.view(-1, NUM_CLASSES, 1) - logits_student.view(-1, 1, NUM_CLASSES))

    if GI:
        assert CLASS_PRIOR is not None, 'CLASS_PRIOR information should be collected'
        with torch.no_grad():
            p_n = CLASS_PRIOR.view(-1, NUM_CLASSES, 1) @ CLASS_PRIOR.view(-1, 1, NUM_CLASSES)
    else:
        s_n = F.softmax(logits_teacher / T2, dim=1)
        p_n = s_n.view(-1, NUM_CLASSES, 1) @ s_n.view(-1, 1, NUM_CLASSES)

    loss_mse = 0.25 * (torch.pow(delta_n - delta_a, 2) * p_n).sum(dim=(1,2))
    loss_sce = -(F.softmax(logits_teacher / temperature, dim=-1).detach() * F.log_softmax(logits_student, dim=-1)).sum(dim=1) # no temperature for strong aug
    return beta * loss_mse + alpha * loss_sce


def consistency_loss(logits, targets, name='ce', mask=None, temperature=1.0, alpha=None, beta=None, CLASS_PRIOR=None, GI=False, T2=100.0):
    """
    wrapper for consistency regularization loss in semi-supervised learning.

    Args:
        logits: logit to calculate the loss on and back-propagion, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    assert name in ['ce', 'mse', 'kl', 'ikl']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    elif name == 'kl':
        loss = F.kl_div(F.log_softmax(logits / 0.5, dim=-1), F.softmax(targets / 0.5, dim=-1), reduction='none')
        loss = torch.sum(loss * (1.0 - mask).unsqueeze(dim=-1).repeat(1, torch.softmax(logits, dim=-1).shape[1]), dim=1)
    elif name == 'ikl':
        loss = dkl_loss(logits, targets, temperature, alpha, beta, CLASS_PRIOR, GI, T2)
    else:
        loss = ce_loss(logits, targets, reduction='none')

    if mask is not None and name != 'kl':
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean()



class ConsistencyLoss(nn.Module):
    """
    Wrapper for consistency loss
    """
    def forward(self, logits, targets, name='ce', mask=None, temperature=1.0, alpha=None, beta=None, CLASS_PRIOR=None, GI=False, T2=1.0):
        return consistency_loss(logits, targets, name, mask, temperature, alpha, beta, CLASS_PRIOR, GI, T2)
