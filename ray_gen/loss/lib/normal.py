import torch
import torch.nn as nn


class NormalLoss:
    def __init__(self) -> None:
        pass

    def get_loss(self, outputs):
        # criterion = nn.CrossEntropyLoss()
        # label = torch.ones(4, dtype=torch.long) * 2
        # loss = criterion(outputs, label)

        loss = torch.var(outputs, 1)
        loss = torch.sum(loss) + (outputs[0, 0] - 1) ** 2

        return loss
