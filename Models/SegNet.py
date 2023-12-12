import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(64)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(128)
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(256)

        # Decoder
        self.dec_conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_bn3 = nn.BatchNorm2d(128)
        self.dec_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(64)
        self.dec_conv1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.enc_bn1(self.enc_conv1(x)))
        x2 = F.relu(self.enc_bn2(self.enc_conv2(x1)))
        x3 = F.relu(self.enc_bn3(self.enc_conv3(x2)))

        # Decoder
        x_dec = F.relu(self.dec_bn3(self.dec_conv3(x3)))
        x_dec = F.relu(self.dec_bn2(self.dec_conv2(x_dec)))
        x_dec = self.dec_conv1(x_dec)

        return x_dec

# Example usage:
segnet_model = SegNet()
random_input = torch.randn(1, 1, 256, 256)
segnet_output = segnet_model(random_input)
assert segnet_output.shape == torch.Size([1, 1, 256, 256])