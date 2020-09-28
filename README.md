# CIFAR-100 Wide CNN
My submission for the UMass ACM ML Club CIFAR-100 contest. Won 1st Place.

## CNN Architecture
```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)
        self.conv2 = nn.Conv2d(128, 256, 3)
        self.conv3 = nn.Conv2d(256, 512, 3)
        self.conv4 = nn.Conv2d(512, 1024, 3)
        self.batchnorm1 = nn.BatchNorm2d(128)
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.batchnorm3 = nn.BatchNorm2d(512)
        self.batchnorm4 = nn.BatchNorm2d(1024)
        self.batchnorm5 = nn.BatchNorm1d(1024)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1024*2*2, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 100)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.batchnorm1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.batchnorm2(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.lrelu(x)
        x = self.batchnorm3(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.lrelu(x)
        x = self.batchnorm4(x)
        x = self.pool(x)

        x = x.view(-1, 1024*2*2)

        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.batchnorm5(x)

        x = self.fc2(x)
        x = self.lrelu(x)
        x = self.batchnorm5(x)

        x = self.fc3(x)
        x = self.lrelu(x)
        return x
```
