"""
モデル用
"""
from socket import CAN_BCM_RX_NO_AUTOTIMER
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class MODEL_CNN(nn.Module):
    """
    とりあえず適当にVGG likeなモデルで
    """
    def __init__(self, out_channels=32, in_channels=3, kernel_size=3, strides=1, padding=1):
        super(MODEL_CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, strides, padding)
        self.conv3 = nn.Conv2d(out_channels, out_channels*2, kernel_size, strides, padding)
        self.conv4 = nn.Conv2d(out_channels*2, out_channels*2, kernel_size, strides, padding)
        self.conv5 = nn.Conv2d(out_channels*2, out_channels*2*2, kernel_size, strides, padding)
        self.conv6 = nn.Conv2d(out_channels*2*2, out_channels*2*2, kernel_size, strides, padding)
        self.conv7 = nn.Conv2d(out_channels*2*2, out_channels*2*2*2, kernel_size, strides, padding)
        self.conv8 = nn.Conv2d(out_channels*2*2*2, out_channels*2*2*2, kernel_size, strides, padding)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.25)
        self.dropout4 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256*16*16, 15587)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.avg_pool2d(x, 2)
        x = self.dropout2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.avg_pool2d(x, 2)
        x = self.dropout3(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.avg_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    model1 = MODEL_CNN()
    summary(model1, input_size=(16, 3, 256, 256)),