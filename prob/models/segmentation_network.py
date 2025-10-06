import torch.nn as nn
import torch

class SegmentationNetwork(nn.Module):
    def __init__(self, num_classes=3):
        super(SegmentationNetwork, self).__init__()
        self.num_classes=num_classes

        self.convblock_down1 = self.double_conv_block(1, 8) # 8x64x64
        self.pool1 = nn.MaxPool2d(2) # 8x32x32

        self.convblock_down2 = self.double_conv_block(8, 16) # 16x32x32
        self.pool2 = nn.MaxPool2d(2) # 16x16x16

        self.convblock_down3 = self.double_conv_block(16, 32) # 32x16x16
        self.pool3 = nn.MaxPool2d(2) # 32x8x8

        self.convblock_down4 = self.double_conv_block(32, 64) # 64x8x8
        self.pool4 = nn.MaxPool2d(2) # 64x4x4

        self.bottom = self.double_conv_block(64, 128) # 128x4x4

        self.conv_transpose4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.convblock_up4 = self.double_conv_block(128, 64)

        self.conv_transpose3 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.convblock_up3 = self.double_conv_block(64, 32)

        self.conv_transpose2 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.convblock_up2 = self.double_conv_block(32, 16)

        self.conv_transpose1 = nn.ConvTranspose2d(16, 8, 2, 2)
        self.convblock_up1 = self.double_conv_block(16, 8)

        self.classifier = nn.Conv2d(8, self.num_classes + 1, kernel_size=1)

    def double_conv_block(self, input, output):
            return nn.Sequential(
                nn.Conv2d(input, output, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(output, output, 3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        # ---- ENCODER ----
        down1 = self.convblock_down1(x)
        down2 = self.convblock_down2(self.pool1(down1))
        down3 = self.convblock_down3(self.pool2(down2))
        down4 = self.convblock_down4(self.pool3(down3))
        
        bottom = self.bottom(self.pool4(down4))

        # ---- DECODER ----
        transpose4 = self.conv_transpose4(bottom)
        up4 = self.convblock_up4(torch.cat([transpose4, down4], dim=1))

        transpose3 = self.conv_transpose3(up4)
        up3 = self.convblock_up3(torch.cat([transpose3, down3], dim=1))

        transpose2 = self.conv_transpose2(up3)
        up2 = self.convblock_up2(torch.cat([transpose2, down2], dim=1))

        transpose1 = self.conv_transpose1(up2)
        up1 = self.convblock_up1(torch.cat([transpose1, down1], dim=1))

        # ---- OUTPUT ----
        out = self.classifier(up1)  # (N, num_classes, 64, 64)
        return out


    @staticmethod
    def get_criterion():
        return nn.CrossEntropyLoss()