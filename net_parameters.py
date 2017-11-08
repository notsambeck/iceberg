'''shared classes for ice.py and test_loader.py'''
import torch.nn as nn
import torch.nn.functional as F


class IceNet(nn.Module):
    def __init__(self, channels):
        super(IceNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)           # pool is not a layer
        self.conv1 = nn.Conv2d(channels, 6, 3)  # 60 > 58
        self.drop0 = nn.Dropout2d()               # NC
        self.conv2 = nn.Conv2d(6, 10, 3)         # 26 -> 24 ->12
        self.drop1 = nn.Dropout2d()               # NC
        self.conv3 = nn.Conv2d(10, 16, 3)         # 12 -> 10 -> 5
        self.drop2 = nn.Dropout2d()
        self.fc1 = nn.Linear(16 * 5 * 5, 256)
        self.drop_fc = nn.Dropout()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        self.best_xval_loss = 10
        self.training = True

    def forward(self, x, v=0):
        if v:
            print(0, x.size())
        x = self.pool(F.relu(self.conv1(x)))
        if v:
            print(1, x.size())
        x = F.dropout(x, training=self.training)
        if v:
            print(2, x.size())
        x = self.pool(F.relu(self.conv2(x)))
        x = F.dropout(x, training=self.training)
        if v:
            print(3, x.size())
        x = self.pool(F.relu(self.conv3(x)))
        if v:
            print(4, x.size())
        x = F.dropout(x, training=self.training)
        if v:
            print(5, x.size())
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        if v:
            print(6, x.size())
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BigNet(nn.Module):
    def __init__(self, channels):
        super(IceNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)           # pool is not a layer
        # inputs, num filters, kernel size (not batch)
        self.conv1a = nn.Conv2d(channels, 4, 3)  # 60 > 58
        self.conv1b = nn.Conv2d(4, 8, 3)        # > 56 > 28
        self.drop0 = nn.Dropout2d()               # NC
        self.conv2a = nn.Conv2d(8, 12, 3)         # 28 -> 26
        self.conv2b = nn.Conv2d(12, 16, 3)         # 26 -> 24 ->12
        self.drop1 = nn.Dropout2d()               # NC
        self.conv3a = nn.Conv2d(16, 16, 3)         # 12 -> 10 -> 5
        self.conv3b = nn.Conv2d(16, 32, 3)         # 12 -> 10 -> 5
        self.drop2 = nn.Dropout2d()
        self.fc1 = nn.Linear(32 * 3 * 3, 256)
        self.drop_fc = nn.Dropout()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        self.best_xval_loss = 10
        self.training = True

    def forward(self, x, v=0):
        x = F.relu(self.conv1a(x))
        if v:
            print(0, x.size())
        x = self.pool(F.relu(self.conv1b(x)))
        if v:
            print(1, x.size())
        x = F.dropout(x, training=self.training)
        if v:
            print(2, x.size())
        x = F.relu(self.conv2a(x))
        x = self.pool(F.relu(self.conv2b(x)))
        x = F.dropout(x, training=self.training)
        if v:
            print(3, x.size())
        x = F.relu(self.conv3a(x))
        x = self.pool(F.relu(self.conv3b(x)))
        if v:
            print(4, x.size())
        x = F.dropout(x, training=self.training)
        if v:
            print(5, x.size())
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        if v:
            print(6, x.size())
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
