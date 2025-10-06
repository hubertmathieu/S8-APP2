import torch.nn as nn
import torch
import torch.nn.functional as F


class DetectionNetwork(nn.Module):
    def __init__(self, num_classes=3, max_objects=3):
        super(DetectionNetwork, self).__init__()
        self.num_classes=num_classes
        self.max_objects = max_objects
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
            nn.Linear(128, 128),  # logits
            nn.ReLU(inplace=True)
        )
        
        self.fc_boxes = nn.Linear(128, self.max_objects * (1 + 3 + self.num_classes))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        boxe_predict = self.fc_boxes(x)
        preds = boxe_predict.view(x.size(0), self.max_objects, 1 + 3 + self.num_classes)
        
        # Sigmoid pour presence et bbox (x,y,size normalisé entre 0 et 1)
        preds[:, :, 0] = torch.sigmoid(preds[:, :, 0])  # presence
        preds[:, :, 1:4] = torch.sigmoid(preds[:, :, 1:4])  # x, y, size

        return preds

    @staticmethod
    def get_criterion(alpha=2, beta=5):
        def criterion(pred, target):
            """
            pred: (N, max_objects, 1+3+C)
            target: (N, max_objects, 5) -> (presence, x, y, size, class_id)
            """
            presence_pred = pred[:, :, 0]
            bbox_pred = pred[:, :, 1:4]
            class_pred = pred[:, :, 4:]  # logits

            presence_target = target[:, :, 0]
            bbox_target = target[:, :, 1:4]
            class_target = target[:, :, 4].long()  # int class_id

            # Loss présence (BCE)
            loss_presence = F.binary_cross_entropy(presence_pred, presence_target)

            # Masque pour bbox et classe (seulement si presence=1)
            mask = presence_target.unsqueeze(-1)

            # Loss bbox
            loss_bbox = F.mse_loss(bbox_pred * mask, bbox_target * mask)

            # Loss classe (CrossEntropy par bbox où presence=1)
            loss_class = 0
            for i in range(pred.size(0)):  # batch
                for j in range(pred.size(1)):  # bbox
                    if presence_target[i, j] > 0:
                        prediction_class = class_pred[i, j].unsqueeze(0)
                        target_class = class_target[i, j].unsqueeze(0)
                        loss_class += F.cross_entropy(prediction_class, target_class)
                        
            loss_class /= (mask.sum() + 1e-6)  # moyenne sur les objets présents

            return loss_presence + beta * loss_bbox + alpha * loss_class

        return criterion