from .lib.normal import NormalLoss


class LossBuilder:
    def get_loss_instance(self):
        return NormalLoss()
