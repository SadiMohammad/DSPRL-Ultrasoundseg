import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.input = inputConv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        self.up1 = up(1024, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.preOutput = outConv(64, 32)
        self.output = outConv(32, n_classes)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x = self.up4(x8, x1)
        x = self.preOutput(x)
        x = self.output(x)
        return x


class doubleConv(nn.Module):
    def __init__(self, inCh, outCh):
        super(doubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inCh,
                               out_channels=outCh,
                               kernel_size=3,
                               padding=1)
        self.batchnorm1 = nn.BatchNorm2d(num_features=outCh)
        self.conv2 = nn.Conv2d(in_channels=outCh,
                               out_channels=outCh,
                               kernel_size=3,
                               padding=1)
        self.batchnorm2 = nn.BatchNorm2d(num_features=outCh)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        return x


class inputConv(nn.Module):
    def __init__(self, inCh, outCh):
        super(inputConv, self).__init__()
        self.conv = doubleConv(inCh, outCh)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, inCh, outCh):
        super(down, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = doubleConv(inCh, outCh)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, inCh, outCh, bilinear=True):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(inCh, outCh, kernel_size=2, stride=2)
        self.conv = doubleConv(inCh, outCh)

    def forward(self, x1, x2):  # x2 is the skip connection
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outConv(nn.Module):
    def __init__(self, inCh, outCh):
        super(outConv, self).__init__()
        self.conv = nn.Conv2d(inCh, outCh, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(1, 1).to(device)
    print(model)
    # summary(UNet, (3, 128, 128))
