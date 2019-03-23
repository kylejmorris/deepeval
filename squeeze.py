import torch
from torchvision.datasets import ImageFolder
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision import datasets, models, transforms>

size = (512, 384)
transform = Compose([Resize(size), ToTensor()])
train_dataset = ImageFolder("train", transform=transform)
train_dataset = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = ImageFolder("test", transform=transform)
test_dataset = DataLoader(test_dataset, batch_size=32, shuffle=True)

classes = 6
model = models.squeezenet1_1(pretrained=True)
for p in model.parameters():
    p.requires_grad = False
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Conv2d(512, classes, kernel_size=1),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d(1),
)
model.num_classes = classes

epochs = 10

def train(model, dataset):
    model = model.cuda()
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for i in range(epochs):
        print(i)
        running_loss = 0
        running_corrects = 0

        for inputs, labels in dataset:
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)

        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = running_corrects.double() / len(loader.dataset)

        print(epoch_loss)
        print(epoch_acc.item())
        print()

def test(model, dataset):
    with torch.no_grad():
        model = model.cuda()
        correct = 0
        total = 0
        for inputs, labels in dataset:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            outputs = torch.Tensor([x.argmax() for x in outputs]).long().cuda()
            correct = (outputs == labels).sum().item()
            total = labels.size(0)
        print(correct/total)


# train(model, train_dataset)
# test(model, test_dataset)
