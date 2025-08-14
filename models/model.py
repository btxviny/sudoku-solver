import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitAndHandwrittenCNN(nn.Module):
    def __init__(self):
        super(DigitAndHandwrittenCNN, self).__init__()
        
        # Encoder (Conv2D + BatchNorm + MaxPool)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # BatchNorm after conv1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # BatchNorm after conv2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # BatchNorm after conv3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.digit_classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 9)  # 9 classes (1-9)
        )

        self.handwritten_classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # 2 classes (handwritten vs. non-handwritten)
        )

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)

        digit_pred = self.digit_classifier(x)
        handwritten_pred = self.handwritten_classifier(x)

        return digit_pred, handwritten_pred
