"""
ResNet20 with BatchNorm
Total number of params 275572
Total layers 20
Accuracy 63.28%
"""

# imports
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

# model
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = LambdaLayer(lambda x: F.pad(
                x[:, :, ::2, ::2], (0, 0, 0, 0, out_channels//4, out_channels//4), "constant", 0))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def _weights_init(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet20():
    model = ResNet(BasicBlock, [3, 3, 3])
    print('ResNet20')

    total_params = 0
    for x in filter(lambda p: p.requires_grad, model.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(
        lambda p: p.requires_grad and len(p.data.size()) > 1, model.parameters()))))

    return model


model = ResNet20()

# hyperparams
batch_size = 128
learning_rate = 0.1
momentum = 0.9
weight_decay = 0.0001
num_epochs = 160

# data
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761])
])

train_dataset = torchvision.datasets.CIFAR100(
    root='./data', download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR100(
    root='./data', download=True, train=False, transform=test_transform)

train_dl = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True)
test_dl = DataLoader(test_dataset, batch_size, pin_memory=True)


def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(make_grid(images, 10).permute(1, 2, 0))
        break


show_batch(train_dl)

# device
def get_default_device():
    return torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')


def to_device(entity, device):
    if isinstance(entity, (list, tuple)):
        return [to_device(elem, device) for elem in entity]
    return entity.to(device, non_blocking=True)


class DeviceDataLoader():
    # wrapper around dataloaders to transfer batches to devices
    def __init__(self, dataloader, device):
        self.dl = dataloader
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)
test_dl = DeviceDataLoader(test_dl, device)

# training
def accuracy(logits, labels):
    pred, predClassId = torch.max(logits, dim=1)
    return torch.tensor(torch.sum(predClassId == labels).item() / len(logits))


def evaluate(model, dl, loss_func):
    model.eval()
    batch_losses, batch_accs = [], []
    for images, labels in dl:
        with torch.no_grad():
            logits = model(images)
        batch_losses.append(loss_func(logits, labels))
        batch_accs.append(accuracy(logits, labels))
        epoch_avg_loss = torch.stack(batch_losses).mean()
        epoch_avg_acc = torch.stack(batch_accs).mean()
        return epoch_avg_loss, epoch_avg_acc


def train(model, train_dl, num_epochs, loss_func, optimizer):
    results = []
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_dl:
            logits = model(images)
            loss = loss_func(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    epoch_avg_loss, epoch_avg_acc = evaluate(model, train_dl, loss_func)
    results.append({'avg_loss': epoch_avg_loss, 'avg_acc': epoch_avg_acc})

    return results


model.cuda()
loss_func = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                            momentum=momentum,
                            weight_decay=weight_decay)

results = train(model, train_dl, num_epochs, loss_func, optimizer)

results

# save
if not os.path.exists('./model'):
    os.makedirs('./model')

torch.save(model.state_dict(), './model/resnet20batchNorm.pth')

# test
model1 = ResNet20().cuda()
model1.load_state_dict(torch.load('./model/resnet20batchNorm.pth'))
_, test_acc = evaluate(model1, test_dl, loss_func)
print(test_acc)
