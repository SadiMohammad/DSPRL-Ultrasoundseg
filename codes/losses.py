

import torch

# https://discuss.pytorch.org/t/flatten-layer-of-pytorch-build-by-sequential-container/5983/2


class Flatten:
    def forward(self, input):
        return input.view(input.size(0), -1)


class Loss:
    def __init__(self, y_true, y_pred):
        self.smooth = 1e-7
        self.y_true_f = Flatten().forward(y_true)
        self.y_pred_f = Flatten().forward(y_pred)
        self.y_pred_f_sigmoid = Flatten().forward(torch.sigmoid(y_pred))
        self.intersection = torch.sum(
            self.y_true_f * self.y_pred_f_sigmoid, dim=1)
        self.union = (torch.sum(self.y_true_f) +
                      torch.sum(self.y_pred_f_sigmoid)) - self.intersection
        self.sum = (torch.sum(self.y_true_f, dim=1) +
                    torch.sum(self.y_pred_f_sigmoid, dim=1))

    def dice_coeff(self):
        intersection = torch.sum(self.y_true_f * self.y_pred_f, dim=1)
        sum = (torch.sum(self.y_true_f, dim=1) +
               torch.sum(self.y_pred_f, dim=1))
        coeff = ((2. * intersection + self.smooth) / (sum + self.smooth))
        return coeff

    def dice_coeff_loss(self):
        loss = ((2. * self.intersection + self.smooth) /
                (self.sum + self.smooth))
        # loss = -torch.log(loss)
        return 1-loss  # sometimes only (-) used

    def iou_calc(self):
        return (self.intersection + self.smooth)/(self.union + self.smooth)

    def iou_calc_loss(self):
        return -((self.intersection + self.smooth) / (self.union + self.smooth))

    def bce_logit_loss(self):
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(self.y_pred_f, self.y_true_f)
        return loss

    def bce_logit_with_dice_loss(self):
        bce_loss = self.bce_logit_loss()
        dice_loss = self.dice_coeff_loss()
        loss = bce_loss + dice_loss
        return loss
