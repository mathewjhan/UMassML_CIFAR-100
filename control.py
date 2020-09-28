import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

torch.manual_seed(42069)

input_size = 3072 # 32*32*3=3072
num_classes = 100
num_epochs = 20
batch_size = 50
l_rate = 1e-3

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomResizedCrop(32, scale=(0.75, 1.0)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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
        self.batchnorm1 = nn.BatchNorm2d(128)
        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*15*15, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 100)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.batchnorm1(x)
        x = self.pool(x)

        x = x.view(-1, 128*15*15)

        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.batchnorm2(x)

        x = self.fc2(x)
        return x

net = Net()
if(torch.cuda.is_available()):
    net.cuda()

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr = l_rate)
#optimizer = torch.optim.SGD(net.parameters(), lr=l_rate, momentum=0.9, weight_decay=0.0005)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 35, 50])


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

