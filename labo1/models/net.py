import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        # Convolution qui « couvre » tout le 28x28 (kernel 28x28)
        # entrée: 1 canal (MNIST), sortie: 10 canaux
        # self.conv = nn.Conv2d(in_channels=1, out_channels=num_classes, kernel_size=3)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        # 2) MaxPool
        self.pool = nn.MaxPool2d(2, 2)  # -> 4x14x14

        # 3) Deuxième convolution 3x3, 2 filtres
        self.conv2 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1)  # -> 2x14x14
        self.bn2 = nn.BatchNorm2d(2)

        # 4) Fully connected sous forme de conv
        # après deuxième maxpool -> 2x7x7
        self.fc = nn.Conv2d(2, num_classes, kernel_size=7)  # -> 10x1x1

    def forward(self, x):
        # x est de forme (batch,1,28,28)
        # x = self.conv(x)
        # output = x.view(x.size(0), -1)
        # return output
        # Bloc 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Bloc 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # -> 2x7x7

        # Fully connected
        x = self.fc(x)    # -> 10x1x1
        x = x.view(x.size(0), -1)  # -> [batch_size, 10]
        return x