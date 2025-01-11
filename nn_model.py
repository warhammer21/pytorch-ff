import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import torchmetrics

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = nn.functional.elu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.elu(self.fc2(x))
        x = nn.functional.elu(self.fc3(x))
        x = nn.functional.sigmoid(self.fc4(x))
        return x
