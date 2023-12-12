import torch

class NestedDoubleConv(torch.nn.Module):
    """
    Helper Class which implements the nested DoubleConvolution block
    """
    def __init__(self, in_channels, out_channels):

        super().__init__()
        self.step = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU())

    def forward(self, X):
        return self.step(X)


class UNetPlusPlus(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = NestedDoubleConv(1, 64)
        self.layer2 = NestedDoubleConv(64, 128)
        self.layer3 = NestedDoubleConv(128, 256)
        self.layer4 = NestedDoubleConv(256, 512)

        self.layer5_1 = NestedDoubleConv(512+256, 256)
        self.layer5_2 = NestedDoubleConv(256, 256)

        self.layer6_1 = NestedDoubleConv(256+128, 128)
        self.layer6_2 = NestedDoubleConv(128, 128)

        self.layer7_1 = NestedDoubleConv(128+64, 64)
        self.layer7_2 = NestedDoubleConv(64, 64)

        self.layer8 = torch.nn.Conv2d(64, 1, 1)

        self.maxpool = torch.nn.MaxPool2d(2)

    def forward(self, x):

        x1 = self.layer1(x)
        x1m = self.maxpool(x1)

        x2 = self.layer2(x1m)
        x2m = self.maxpool(x2)

        x3 = self.layer3(x2m)
        x3m = self.maxpool(x3)

        x4 = self.layer4(x3m)

        x5_1 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x4)
        x5_1 = torch.cat([x5_1, x3], dim=1)
        x5_2 = self.layer5_1(x5_1)
        x5_2 = self.layer5_2(x5_2)

        x6_1 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x5_2)
        x6_1 = torch.cat([x6_1, x2], dim=1)
        x6_2 = self.layer6_1(x6_1)
        x6_2 = self.layer6_2(x6_2)

        x7_1 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x6_2)
        x7_1 = torch.cat([x7_1, x1], dim=1)
        x7_2 = self.layer7_1(x7_1)
        x7_2 = self.layer7_2(x7_2)

        ret = self.layer8(x7_2)
        return ret


model = UNetPlusPlus()

random_input = torch.randn(1, 1, 256, 256)
output = model(random_input)
assert output.shape == torch.Size([1, 1, 256, 256])
