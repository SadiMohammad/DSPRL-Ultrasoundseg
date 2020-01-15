import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class LungNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(LungNet, self).__init__()
        self.inputx = inputBlock(n_channels, n_channels)
        self.block1 = convBlock(n_channels, 32, 1)
        self.block2 = convBlock(32, 32, 1)
        self.block3 = convBlock(32, 32, 2)
        self.block4 = convBlock(32, 32, 3)
        self.block5 = convBlock(32, 32, 5)
        self.block6 = convBlock(32, 32, 7)
        self.block7 = convBlock(32, 32, 11)
        self.block8 = convBlock(32, 32, 13)
        self.block9 = convBlock(32, 32, 17)
        self.block10 = convBlock(32, 32, 23)
        self.block11 = convBlock(32, 32, 29)
        self.block12 = convBlock(32, 32, 31)
        self.block13 = convBlock(32, 32, 37)
        self.block14 = convBlock(32, 32, 41)
        self.block15 = convBlock(32, 32, 43)
        self.block16 = convBlock(32, 32, 47)
        self.block17 = convBlock(32, 32, 53)
        self.block18 = convBlock(545, 324, 1)
        self.block19 = convBlock(324, 224, 1)
        self.block20 = convBlock(224, 128, 1)
        self.block21 = convBlock(128, 32, 1)
        self.block22 = convBlock(32, n_classes, 1)

    def forward(self, x):
        x1 = self.inputx(x)
        x2 = self.block1(x1)
        x3 = self.block2(x2)
        x4 = self.block3(x3)
        x5 = self.block4(x4)
        x6 = self.block5(x5)
        x7 = self.block6(x6)
        x8 = self.block7(x7)
        x9 = self.block8(x8)
        x10 = self.block9(x9)
        x11 = self.block10(x10)
        x12 = self.block11(x11)
        x13 = self.block12(x12)
        x14 = self.block13(x13)
        x15 = self.block14(x14)
        x16 = self.block15(x15)
        x17 = self.block16(x16)
        x18 = self.block17(x17)
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18), dim=1)
        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)
        x = self.block21(x)
        x = self.block22(x)
        return torch.sigmoid(x)


class convBlock(nn.Module):
    def __init__(self, inCh, outCh, dialation_rate):
        super(convBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels = inCh,
                               out_channels = outCh,
                               kernel_size = 3,
                               padding = dialation_rate,
                               dilation = dialation_rate)
        self.batchnorm = nn.BatchNorm2d(num_features = outCh)

    def forward(self, x):
        return F.relu(torch.add(self.conv(x), self.batchnorm(self.conv(x))))

class inputBlock(nn.Module):
    def __init__(self, inCh, outCh):
        super(inputBlock, self).__init__()
        self.batchnorm = nn.BatchNorm2d(num_features = outCh)

    def forward(self, x):
    	return torch.add(x, self.batchnorm(x))
