import torch
from torchvision.transforms import Lambda
import torch.nn as nn
import torch.nn.functional as F
from math import floor, log

class View(nn.Module):
  # From PyTorch issue #2486
  def __init__(self, shape):
    super(View, self).__init__()
    self.shape = shape
  def forward(self, x):
    return x.view(self.shape, x.size(0))

class LogSoftmax(nn.Module):
  def __init__(self):
     super(LogSoftmax, self).__init__()
     self.softmax = nn.Softmax()
  def forward(self, x):
     x = self.softmax(x)
     return log(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # CONSTANTS
        NUM_CLASSES = 6
        NUM_CHANNELS = 3
        scale = 1 # what's this for?
        # SETTING UP duh layers
        conv1Channels = floor(96 * scale)
        conv1Kernel = 11
        conv1Stride = 4
        conv1Pad = 2
        pool1Kernel = 3
        pool1Stride = 2
        pool1Pad = 0
        conv2Channels = floor(256 * scale)
        conv2Kernel = 5
        conv2Stride = 1
        conv2Pad = 2
        pool2Kernel = 3
        pool2Stride = 2
        pool2Pad = 0
        conv3Channels = floor(384 * scale)
        conv3Kernel = 3
        conv3Stride = 1
        conv3Pad = 1
        conv4Channels = floor(384 * scale)
        conv4Kernel = 3
        conv4Stride = 1
        conv4Pad = 1
        conv5Channels = floor(256 * scale)
        conv5Kernel = 3
        conv5Stride = 1
        conv5Pad = 1
        pool5Kernel = 3
        pool5Stride = 2
        pool5Pad = 0
        self.conv1 = nn.Conv2d(NUM_CHANNELS, conv1Channels, conv1Kernel, conv1Stride, conv1Pad)
        self.pool1 = nn.MaxPool2d(pool1Kernel, pool1Stride, pool1Pad)
        self.conv2 = nn.Conv2d(conv1Channels, conv2Channels, conv2Kernel, conv2Stride, conv2Pad)
        self.pool2 = nn.MaxPool2d(pool2Kernel, pool2Stride, pool2Pad)
        self.conv3 = nn.Conv2d(conv2Channels, conv3Channels, conv3Kernel, conv3Stride, conv3Pad)
        self.conv4 = nn.Conv2d(conv3Channels, conv4Channels, conv4Kernel, conv4Stride, conv4Pad)
        self.conv5 = nn.Conv2d(conv4Channels, conv5Channels, conv5Kernel, conv5Stride, conv5Pad)
        self.pool5 = nn.MaxPool2d(pool5Kernel, pool5Stride, pool5Pad)
        self.fc6Channels = floor(4096 * scale)
        self.fc7Channels = floor(4096 * scale)

        self.model = nn.Sequential()

        self.model.add_module("conv1", self.conv1)
        self.model.add_module("relu1", nn.ReLU())
        self.model.add_module("pool1", self.pool1)
        self.model.add_module("conv2", self.conv2)
        self.model.add_module("relu2", nn.ReLU())
        self.model.add_module("pool2", self.pool2)
        self.model.add_module("conv3", self.conv3)
        self.model.add_module("relu3", nn.ReLU())
        self.model.add_module("conv4", self.conv4)
        self.model.add_module("relu4", nn.ReLU())
        self.model.add_module("conv5", self.conv5)
        self.model.add_module("relu5", nn.ReLU())
        self.model.add_module("pool5", self.pool5)
        self.model.add_module("view1", View(conv5Channels * 7 * 7))
        self.model.add_module("dropout", nn.Dropout(0.5))
        self.model.add_module("linear1", nn.Linear(conv5Channels * 7 * 7, self.fc6Channels))
        self.model.add_module("threshold1", nn.Threshold(0, 1e-6))
        self.model.add_module("dropout2", nn.Dropout(0.5))
        self.model.add_module("linear2", nn.Linear(self.fc6Channels, self.fc7Channels))
        self.model.add_module("threshold3", nn.Threshold(0, 1e-6))
        self.model.add_module("linear4", nn.Linear(self.fc7Channels, NUM_CLASSES))
        self.model.add_module("logsoftmax1", LogSoftmax())

    def forward(self, x):
        return self.model(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

def dataset(path, size=(512, 384), batch=32):
    transform = Compose([Resize(size), ToTensor()])
    dataset = ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True)
    return loader

from torch import nn, optim

def train(model, data, epochs=10, lr=1e-3, momentum=9e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    for e in range(epochs):
        e_loss = 0.0
        for img, cls in data:
            #  img.to("cuda")
            #  cls.to("cuda")
            out = model(img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            e_loss += loss.item()
            if i % 2000 == 1999:
                print("epoch {} loss: {}".format(e, e_loss))
    return model

from os import listdir, mkdir
from os.path import join
from random import choice, random
from shutil import copy
from sys import argv

def split(data_dir):
  classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
  for s in ["train", "test"]:
    try:
      mkdir(s)
    except:
        pass
    for c in classes:
        try:
          mkdir(join(s, c))
        except:  # it probably already exists, that's fine
            pass
  for cls in classes:
    for fn in listdir(join(data_dir, cls)):
      src = join(data_dir, cls, fn)
      dest = join("train" if random() > 0.3 else "test", cls, fn)
      copy(src, dest)
