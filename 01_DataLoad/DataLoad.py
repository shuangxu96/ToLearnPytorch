# -*- coding: utf-8 -*-
"""
@author: Shuang Xu
"""
import numpy as np
import gzip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def fmnist_loader():
    with gzip.open(r'D:\py\data\fashion-mnist\train-labels-idx1-ubyte.gz') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)
    with gzip.open(r'D:\py\data\fashion-mnist\train-images-idx3-ubyte.gz', 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape(
                len(labels), 1, 28, 28)
    with gzip.open(r'D:\py\data\fashion-mnist\t10k-labels-idx1-ubyte.gz', 'rb') as tlbpath:
        tlabels = np.frombuffer(tlbpath.read(), dtype=np.uint8,offset=8)
    with gzip.open(r'D:\py\data\fashion-mnist\t10k-images-idx3-ubyte.gz', 'rb') as timgpath:
        timages = np.frombuffer(timgpath.read(), dtype=np.uint8,offset=16).reshape(
                len(tlabels), 1, 28, 28)
        
    labels = torch.from_numpy(labels).type(torch.LongTensor)
    images = torch.from_numpy(images).type(torch.float32)
    tlabels = torch.from_numpy(tlabels).type(torch.LongTensor)
    timages = torch.from_numpy(timages).type(torch.float32)
    
    trainset = torch.utils.data.TensorDataset(images, labels)
    testset = torch.utils.data.TensorDataset(timages, tlabels)
    return trainset, testset

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x)) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

trainset, testset = fmnist_loader()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

net = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
num_epoch = 5
for epoch in range(num_epoch):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(correct/total)
