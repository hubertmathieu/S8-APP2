import torch.nn as nn
import torch
import torch.nn.functional as F


class DetectionNetwork(nn.Module):
    def __init__(self, num_classes=3, max_objects=3):
        super(DetectionNetwork, self).__init__()
        self.num_classes=num_classes
        self.max_objects = max_objects
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),  # 2048 -> 128
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),  # logits
            nn.ReLU(inplace=True)
        )
        
        self.fc_boxes = nn.Linear(256, self.max_objects * (1 + 3 + self.num_classes))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        boxe_predict = self.fc_boxes(x)
        preds = boxe_predict.view(boxe_predict.size(0), self.max_objects, 1 + 3 + self.num_classes)
        
        # Sigmoid pour presence et bbox (x,y,size normalisé entre 0 et 1)
        preds[:, :, 0] = torch.sigmoid(preds[:, :, 0])  # presence
        preds[:, :, 1:4] = torch.sigmoid(preds[:, :, 1:4])  # x, y, size

        return preds

    @staticmethod
    def get_criterion(alpha=1, beta=10):
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
            
            mask = presence_target.unsqueeze(-1)

            pos_weight = (presence_target == 0).sum() / ((presence_target == 1).sum() + 1e-6)
            loss_presence = F.binary_cross_entropy(
                presence_pred, presence_target, weight=pos_weight
            )

            # Loss bbox
            loss_bbox = 0
            # Loss classe (CrossEntropy par bbox où presence=1)
            loss_class = 0
            
            valid = presence_target > 0
            if valid.sum() > 0:
                loss_bbox = F.mse_loss(bbox_pred[valid], bbox_target[valid])
                loss_class = F.cross_entropy(class_pred[valid], class_target[valid])
                
            #loss_class /= (mask.sum() + 1e-6)  # moyenne sur les objets présents

            return loss_presence + beta * loss_bbox + alpha * loss_class

        return criterion