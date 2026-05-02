# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 21:05:20 2021

@author: AM4
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time

# Сначала определим на каком устройстве будем работать - GPU или CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
# Затем загружаем данные
# Создадим преобразования для собственного набора данных
data_transforms = transforms.Compose([
    transforms.Resize(68),
    transforms.CenterCrop(64),
    transforms.ToTensor()
])

# Обучающий набор
train_dataset = torchvision.datasets.ImageFolder(
    root='./data/train',
    transform=data_transforms
)

# Тестовый набор
test_dataset = torchvision.datasets.ImageFolder(
    root='./data/test',
    transform=data_transforms
)

# Посмотрим какие классы содержатся в наборе
class_names = train_dataset.classes
print(class_names)

# Список изображений можно получить следующим образом
train_set = train_dataset.samples
print(train_set[1])

# Посмотрим на размер нашего набора данных
print(len(train_set))

batch_size = 10

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

# Загрузим одну порцию данных
inputs, classes = next(iter(train_loader))
print(inputs.shape)

img = torchvision.utils.make_grid(inputs, nrow=5)
img = img.numpy().transpose((1, 2, 0))
plt.imshow(img)
plt.show()

# Теперь можно переходить к созданию сети
class CnNet(nn.Module):
    def __init__(self, num_classes=3):
        nn.Module.__init__(self)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(8 * 8 * 64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# Количество классов
num_classes = 3

# Создаем экземпляр сети
net = CnNet(num_classes).to(device)

# Задаем функцию потерь и алгоритм оптимизации
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# Создаем цикл обучения и замеряем время его выполнения
t = time.time()

num_epochs = 50
save_loss = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        loss = lossFn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        save_loss.append(loss.item())

        if i % 100 == 0:
            print('Эпоха ' + str(epoch) + ' из ' + str(num_epochs) + ' Шаг ' +
                  str(i) + ' Ошибка: ', loss.item())

print(time.time() - t)

plt.figure()
plt.plot(save_loss)
plt.show()

# Посчитаем точность модели
correct_predictions = 0
num_test_samples = len(test_dataset)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images)
        _, pred_class = torch.max(pred.data, 1)
        correct_predictions += (pred_class == labels).sum().item()

print('Точность модели: ' + str(100 * correct_predictions / num_test_samples) + '%')

# Сохраним модель
torch.save(net.state_dict(), 'CnNet_3classes.ckpt')

# Теперь улучшим результат с помощью предобученной сети
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.ImageFolder(
    root='./data/train',
    transform=data_transforms
)

test_dataset = torchvision.datasets.ImageFolder(
    root='./data/test',
    transform=data_transforms
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

# В качестве донора возьмем предобученную сеть AlexNet
net = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)

print(net)

# Замораживаем все веса
for param in net.parameters():
    param.requires_grad = False

# Меняем классификатор под 3 класса
new_classifier = net.classifier[:-1]
new_classifier.add_module('fc', nn.Linear(4096, num_classes))
net.classifier = new_classifier

net = net.to(device)

# Проверка до обучения
correct_predictions = 0
num_test_samples = len(test_dataset)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images)
        _, pred_class = torch.max(pred.data, 1)
        correct_predictions += (pred_class == labels).sum().item()

print('Точность модели: ' + str(100 * correct_predictions / num_test_samples) + '%')

# Обучение предобученной сети
num_epochs = 2
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

t = time.time()
save_loss = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        loss = lossFn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        save_loss.append(loss.item())

        if i % 100 == 0:
            print('Эпоха ' + str(epoch) + ' из ' + str(num_epochs) + ' Шаг ' +
                  str(i) + ' Ошибка: ', loss.item())

print(time.time() - t)

plt.figure()
plt.plot(save_loss)
plt.show()

# Еще раз посчитаем точность модели
correct_predictions = 0
num_test_samples = len(test_dataset)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images)
        _, pred_class = torch.max(pred.data, 1)
        correct_predictions += (pred_class == labels).sum().item()

print('Точность модели: ' + str(100 * correct_predictions / num_test_samples) + '%')

# Отображение картинок и их класса, предсказанного сетью
inputs, classes = next(iter(test_loader))
pred = net(inputs.to(device))
_, pred_class = torch.max(pred.data, 1)

for i, j in zip(inputs, pred_class):
    img = i.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(class_names[j])
    plt.pause(2)