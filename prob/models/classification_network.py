import torch.nn as nn
import torch
import torch.nn.functional as F


class ClassificationNetwork(nn.Module):
    def __init__(self, num_classes=3):
        super(ClassificationNetwork, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),  # 1 -> 8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x64 -> 32x32

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),  # 8 -> 16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 16 -> 32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),  # 2048 -> 128
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)  # logits
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
        
    
    @staticmethod
    def get_criterion():
        return nn.CrossEntropyLoss()  # use CrossEntropy for multi-class classification
