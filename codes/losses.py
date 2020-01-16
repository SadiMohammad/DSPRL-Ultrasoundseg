

import torch

# https://discuss.pytorch.org/t/flatten-layer-of-pytorch-build-by-sequential-container/5983/2


class Flatten:
    def forward(self, input):
        return input.view(input.size(0), -1)


class Loss:
    def __init__(self, y_true, y_pred):
        self.smooth = 1e-7
        self.y_pred = y_pred
        self.y_true = y_true
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
        loss = bce_loss + 3*dice_loss
        return loss

    def ac_loss(self):
        """
        lenth term
        """

        # horizontal and vertical directions
        x = self.y_pred[:, :, 1:, :] - self.y_pred[:, :, :-1, :]
        y = self.y_pred[:, :, :, 1:] - self.y_pred[:, :, :, :-1]

        delta_x = x[:, :, 1:, :-2]**2
        delta_y = y[:, :, :-2, 1:]**2
        delta_u = torch.abs(delta_x + delta_y)

        epsilon = 1e-8
        w = 1   # equ.(11) in the paper
        lenth = w * torch.sum(torch.sqrt(delta_u + epsilon))

        """
        region term
        """

        C_1 = torch.cuda.FloatTensor(224, 224).fill_(0)
        C_2 = torch.cuda.FloatTensor(224, 224).fill_(1)

        region_in = torch.abs(torch.sum(
            self.y_pred[:, 0, :, :] * ((self.y_true[:, 0, :, :] - C_1)**2)))  # equ.(12) in the paper
        region_out = torch.abs(torch.sum(
            (1-self.y_pred[:, 0, :, :]) * ((self.y_true[:, 0, :, :] - C_2)**2)))  # equ.(12) in the paper

        lambdaP = 1  # lambda parameter could be various.

        loss = lenth + lambdaP * (region_in + region_out)

        return loss
