import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),  # Input: 3x100x100
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),        # 64x48x48
            nn.Conv2d(64, 128, kernel_size=5), # 128x44x44
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),        # 128x22x22
            nn.Conv2d(128, 256, kernel_size=3),# 256x20x20
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)         # 256x10x10
        )
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256 * 10 * 10, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),  # Embedding size: 256
            nn.ReLU()
        )

    def forward_one(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        # Process both images through the same CNN
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Euclidean distance between embeddings
        euclidean_distance = F.pairwise_distance(output1, output2)
        # Contrastive loss
        loss_same = label * torch.pow(euclidean_distance, 2)
        loss_diff = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss = 0.5 * (loss_same + loss_diff)
        return loss.mean()