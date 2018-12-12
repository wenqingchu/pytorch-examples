import argparse
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from models import *

def adjust_learning_rate(optimizer, epoch, model_type):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if model_type == 1:
        if epoch < 80:
            lr = args.lr
        elif epoch < 120:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
    elif model_type == 2:
        if epoch < 60:
            lr = args.lr
        elif epoch < 120:
            lr = args.lr * 0.2
        elif epoch < 160:
            lr = args.lr * 0.04
        else:
            lr = args.lr * 0.008
    elif model_type == 3:
        if epoch < 150:
            lr = args.lr
        elif epoch < 225:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')

args = parser.parse_args()

# prepare data
root = '/home/chuwenqing/.torch/datasets/'
input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
trainset = torchvision.datasets.CIFAR10(root, train=True, transform=input_transform, download=True)
testset = torchvision.datasets.CIFAR10(root, train=False, transform=input_transform, download=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers = 4)

testloader = torch.utils.data.DataLoader(testset, batch_size=int(args.batch_size/2), shuffle=True, num_workers = 4)


# prepare model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#model = Net()
#model = resnet20_cifar()
model = resneXt_cifar(29, 16, 64)

# adjust the lr according to the model type
if isinstance(model, (ResNet_Cifar, PreAct_ResNet_Cifar)):
    model_type = 1
elif isinstance(model, Wide_ResNet_Cifar):
    model_type = 2
elif isinstance(model, (ResNeXt_Cifar, DenseNet_Cifar)):
    model_type = 3
else:
    print('model type unrecognized...')
    model_type = 1




#model = torchvision.models.resnet18(pretrained=False)
model = nn.DataParallel(model).cuda()
# optimize
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
num_epochs = args.epochs
total_step = len(trainloader)
for epoch in range(num_epochs):
    adjust_learning_rate(optimizer, epoch, model_type)
    for i, (images, labels) in enumerate(trainloader):
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader):
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
    model.train()
torch.save(model.state_dict(), 'cifar10_params.pth')




# test

