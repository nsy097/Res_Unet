""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from .attention_model import _DAHead

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        # self.dahead = _DAHead(64, 2)
        # self.attention_out = AttentionConv(4,2)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x4_up = self.up1(x5, x4)
        x3_up = self.up2(x4_up, x3)
        x2_up = self.up3(x3_up, x2)
        x1_up = self.up4(x2_up, x1)
        logits = self.outc(x1_up)
        # attetion_out = self.dahead(x1_up)
        # logits_attention = self.attention_out(logits,attetion_out)
        # print(logits.size())
        return logits
