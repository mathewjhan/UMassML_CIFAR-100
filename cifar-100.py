import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import widenet
from torch.autograd import Variable

torch.manual_seed(69420)

input_size = 3072 # 32*32*3=3072
num_classes = 100
num_epochs = 60
batch_size = 50
l_rate = 1e-3

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomResizedCrop(64, scale=(0.75, 1.0)),
     transforms.ToTensor(),
     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.229, 0.224, 0.225))])

train_data = dsets.CIFAR100(root = './data', train = True,
        transform = transform, download = 1)
test_data = dsets.CIFAR100(root = './data', train = False,
        transform = transform)

train_gen = torch.utils.data.DataLoader(dataset = train_data,
        batch_size = batch_size,
        shuffle = True)
test_gen = torch.utils.data.DataLoader(dataset = test_data,
        batch_size = batch_size,
        shuffle = False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)
        self.conv2 = nn.Conv2d(128, 256, 3)
        self.conv3 = nn.Conv2d(256, 512, 3)
        self.conv4 = nn.Conv2d(512, 1024, 3)
        self.dropout1 = nn.Dropout2d(p=0.1)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)
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

net = Net()
if(torch.cuda.is_available()):
    net.cuda()

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr = l_rate)
#optimizer = torch.optim.SGD(net.parameters(), lr=l_rate, momentum=0.9, weight_decay=0.0005)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 35, 45, 50, 55])


running_loss = 0.0
epoch_loss = 0.0
for epoch in range(num_epochs):
    correct = 0
    total = 0
    for i,(images,labels) in enumerate(train_gen):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item()
        epoch_loss += loss.item()
        if((i+1) % batch_size == 0):
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, i+1, len(train_data)/batch_size, running_loss/batch_size))
            running_loss = 0.0

    scheduler.step()
    print("Train accuracy: %.4f%%" %(correct*100/total))
    print("Epoch loss: %.4f\n" %(epoch_loss/(total/batch_size)))
    epoch_loss = 0.0

correct = 0
total = 0
for images,labels in test_gen:
    images = images.cuda()
    labels = labels.cuda()

    output = net(images)
    _, predicted = torch.max(output,1)
    correct += (predicted == labels).sum().item()
    total += labels.size(0)

print('Accuracy of the model: %.3f%%' %((100*correct)/(total)))
print("Saving model...")
torch.save(net, "/home/mathew/cifar-100/model.pth")
print("Done!")
