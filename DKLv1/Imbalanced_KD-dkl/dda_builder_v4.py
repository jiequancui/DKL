import torch
import torch.nn as nn


class DDAModel(nn.Module):
    def __init__(self, base_encoder, num_classes=1000):
        super(DDAModel, self).__init__()
        self.encoder = base_encoder(num_classes=num_classes)

    def _forward_train(self, x1):
        logits_q = self.encoder(x1)
        return logits_q

    def _forward_eval(self, x):
        return self.encoder(x)

    def forward(self, x1):
        if not self.training:
            return self._forward_eval(x1)
        else:
            return self._forward_train(x1)
