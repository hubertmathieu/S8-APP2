import torch.nn as nn
import torch

class SegmentationNetwork(nn.Module):
    def __init__(self, num_classes=3):
        super(SegmentationNetwork, self).__init__()
        self.num_classes=num_classes

        self.convblock_down1 = self.double_convblock(1, 16) # 8x64x64
        self.pool1 = nn.MaxPool2d(2) # 8x32x32

        self.convblock_down2 = self.double_convblock(16, 32) # 16x32x32
        self.pool2 = nn.MaxPool2d(2) # 16x16x16

        self.convblock_down3 = self.double_convblock(32, 64) # 32x16x16
        self.pool3 = nn.MaxPool2d(2) # 32x8x8

        self.convblock_down4 = self.double_convblock(64, 128) # 64x8x8
        self.pool4 = nn.MaxPool2d(2) # 64x4x4

        self.bottom = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True)
        )

        self.conv_transpose4 = nn.ConvTranspose2d(128, 128, kernel_size=2, padding=0, stride=2)
        self.convblock_up4 = self.double_convblock(256, 64)

        self.conv_transpose3 = nn.ConvTranspose2d(64, 64, kernel_size=2, padding=0, stride=2)
        self.convblock_up3 = self.double_convblock(128, 32)

        self.conv_transpose2 = nn.ConvTranspose2d(32, 32, kernel_size=2, padding=0, stride=2)
        self.convblock_up2 = self.double_convblock(64, 16)

        self.conv_transpose1 = nn.ConvTranspose2d(16, 16, kernel_size=2, padding=0, stride=2)
        self.convblock_up1 = self.double_convblock(32, 8)

        self.classifier = nn.Conv2d(8, self.num_classes + 1, kernel_size=1, padding=0, stride=1)

    def double_convblock(self, input, output):
        return nn.Sequential(
            nn.Conv2d(input, output, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(output),
            nn.ReLU(),
            nn.Conv2d(output, output, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
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
    def get_criterion(device='cpu'):
        class_weights = torch.tensor([0.3, 1.0, 1.0, 1.0], device=device)
        return nn.CrossEntropyLoss(weight=class_weights)