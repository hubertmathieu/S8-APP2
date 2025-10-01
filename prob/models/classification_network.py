import torch.nn as nn
import torch
import torch.nn.functional as F


class ClassificationNetwork(nn.Module):
    def __init__(self, num_classes=3):
        super(ClassificationNetwork, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Conv1 32 x 64 x 64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> downsample by 2 32 x 32 x 32

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Conv2 64 x 32 x 32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> downsample by 2 64 x 16 x 16

            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),  # Conv3 64 x 16 x 16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> downsample by 2 16 x 8 x 8
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
        
    
    @staticmethod
    def get_criterion():
        return nn.CrossEntropyLoss()  # use CrossEntropy for multi-class classification
