import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


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

    loss_mse = 0.25 * (torch.pow(delta_n - delta_a, 2) * p_n).sum() / logits_student.size(0)
    loss_sce = -(F.softmax(logits_teacher / temperature, dim=-1).detach() * F.log_softmax(logits_student / temperature, dim=-1)).sum(1).mean()
    return beta * loss_mse + alpha * loss_sce


class IKL_KD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(IKL_KD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.IKL_KD.CE_WEIGHT
        self.num_classes = cfg.IKL_KD.NUM_CLASSES
        self.temperature = cfg.IKL_KD.T
        self.warmup = cfg.IKL_KD.WARMUP

        self.alpha = cfg.IKL_KD.ALPHA
        self.beta = cfg.IKL_KD.BETA
        self.temperature2 = cfg.IKL_KD.T2
        self.GI = cfg.IKL_KD.GI

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
    
        prior = kwargs['prior']

        onehot = F.one_hot(target, logits_student.size(-1)).float()
        CLASS_PRIOR = onehot @ prior
        loss_dkl = min(kwargs["epoch"] / self.warmup, 1.0) * dkl_loss(
            logits_student,
            logits_teacher,
            self.temperature,
            self.alpha,
            self.beta,
            CLASS_PRIOR,
            self.GI,
            self.temperature2,
        )
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkl,
        }
        return logits_student, losses_dict, logits_teacher.clone().detach() / self.temperature2
