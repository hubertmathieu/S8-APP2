import torch.nn as nn
import torch


class UNet(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(UNet, self).__init__()
        # ------------------------ Laboratoire 2 - Question 5 - Début de la section à compléter ------------------------
        self.hidden = None


        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.maxpool = nn.MaxPool2d(2)

        # Down
        self.conv_1_1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.relu_1_1 = nn.ReLU(inplace=True)
        self.conv_1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.relu_1_2 = nn.ReLU(inplace=True)

        self.maxpool_2 = nn.MaxPool2d(2)
        self.conv_2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu_2_1 = nn.ReLU(inplace=True)
        self.conv_2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu_2_2 = nn.ReLU(inplace=True)

        self.maxpool_3 = nn.MaxPool2d(2)
        self.conv_3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu_3_1 = nn.ReLU(inplace=True)
        self.conv_3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu_3_2 = nn.ReLU(inplace=True)

        self.maxpool_4 = nn.MaxPool2d(2)
        self.conv_4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu_4_1 = nn.ReLU(inplace=True)
        self.conv_4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu_4_2 = nn.ReLU(inplace=True)

        self.maxpool_5 = nn.MaxPool2d(2)
        self.conv_5_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu_5_1 = nn.ReLU(inplace=True)
        self.conv_5_2 = nn.Conv2d(512, 256, 3, padding=1)
        self.relu_5_2 = nn.ReLU(inplace=True)

        # Up
        self.upsample_6 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.conv_6_1 = nn.Conv2d(512, 256, 3, padding=1)
        self.relu_6_1 = nn.ReLU(inplace=True)
        self.conv_6_2 = nn.Conv2d(256, 128, 3, padding=1)
        self.relu_6_2 = nn.ReLU(inplace=True)

        self.upsample_7 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.conv_7_1 = nn.Conv2d(256, 128, 3, padding=1)
        self.relu_7_1 = nn.ReLU(inplace=True)
        self.conv_7_2 = nn.Conv2d(128, 64, 3, padding=1)
        self.relu_7_2 = nn.ReLU(inplace=True)

        self.upsample_8 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv_8_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.relu_8_1 = nn.ReLU(inplace=True)
        self.conv_8_2 = nn.Conv2d(64, 32, 3, padding=1)
        self.relu_8_2 = nn.ReLU(inplace=True)

        self.upsample_9 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.conv_9_1 = nn.Conv2d(64, 32, 3, padding=1)
        self.relu_9_1 = nn.ReLU(inplace=True)
        self.conv_9_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.relu_9_2 = nn.ReLU(inplace=True)

        self.output_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Down
        c1 = self.relu_1_2(self.conv_1_2(self.relu_1_1(self.conv_1_1(x))))
        p1 = self.maxpool_2(c1)

        c2 = self.relu_2_2(self.conv_2_2(self.relu_2_1(self.conv_2_1(p1))))
        p2 = self.maxpool_3(c2)

        c3 = self.relu_3_2(self.conv_3_2(self.relu_3_1(self.conv_3_1(p2))))
        p3 = self.maxpool_4(c3)

        c4 = self.relu_4_2(self.conv_4_2(self.relu_4_1(self.conv_4_1(p3))))
        p4 = self.maxpool_5(c4)

        c5 = self.relu_5_2(self.conv_5_2(self.relu_5_1(self.conv_5_1(p4))))

        # Up
        u6 = self.upsample_6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.relu_6_2(self.conv_6_2(self.relu_6_1(self.conv_6_1(u6))))

        u7 = self.upsample_7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.relu_7_2(self.conv_7_2(self.relu_7_1(self.conv_7_1(u7))))

        u8 = self.upsample_8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.relu_8_2(self.conv_8_2(self.relu_8_1(self.conv_8_1(u8))))

        u9 = self.upsample_9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.relu_9_2(self.conv_9_2(self.relu_9_1(self.conv_9_1(u9))))

        out = self.output_conv(c9)
        return out