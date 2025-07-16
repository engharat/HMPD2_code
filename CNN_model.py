import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_channels=24):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(512 * 5 * 5, 1024)  # Assuming input image size is 64x64
        self.fc2 = nn.Linear(1024, 512)
        
        self.output_channels = 3
        self.height = 80 // 16  # Due to 4 max-pooling layers
        self.width = 80 // 16   # Due to 4 max-pooling layers
        
        self.fc3 = nn.Linear(512, self.height * self.width * self.output_channels)

        # Additional layer for binary classification
        self.fc4 = nn.Linear(self.height * self.width * self.output_channels, 2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = x.view(-1, 512 * 5 * 5)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = torch.sigmoid(self.fc3(x))
        x = x.view(-1, self.height * self.width * self.output_channels)
        
        # Binary classification layer
        x = self.fc4(x)
        #import pdb; pdb.set_trace()
        #x = torch.unsqueeze(x)
        return x


