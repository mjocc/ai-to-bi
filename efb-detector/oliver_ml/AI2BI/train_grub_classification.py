import timm
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import GNNModel
import os

model = GNNModel()

print(sum(p.numel() for p in model.parameters()))

train_dataset = ImageFolder("datasets/grubs/training", transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomCrop((48, 48)),
    transforms.RandomAdjustSharpness(2)
    #transforms.ColorJitter(0.1, 0.1, 0.1, 0.01)
    #transforms.RandomAutocontrast(),
    #transforms.RandomRotation(10)
]))

val_dataset = ImageFolder("datasets/grubs/validation", transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(10),
    transforms.RandomCrop((48, 48)),
    #transforms.RandomAdjustSharpness(2),
    #transforms.ColorJitter(0.1, 0.1, 0.1, 0.01)
    #transforms.RandomAutocontrast(),
    #transforms.RandomRotation(10)
]))


weights = [0] * len(train_dataset)
num_uninfected = len(os.listdir("datasets/grubs/training/uninfected"))
num_infected = len(os.listdir("datasets/grubs/training/efb"))

for idx, (data, label) in enumerate(train_dataset):
    weights[idx] = 1/num_uninfected if label == 1 else 1/num_infected

train_sampler = WeightedRandomSampler(weights, replacement=True, num_samples=len(weights))
train_dataloader = DataLoader(train_dataset, batch_size = 4, sampler=train_sampler)

weights = [0] * len(val_dataset)
num_uninfected = len(os.listdir("datasets/grubs/validation/uninfected"))
num_infected = len(os.listdir("datasets/grubs/validation/efb"))

for idx, (data, label) in enumerate(val_dataset):
    weights[idx] = 1/num_uninfected if label == 1 else 1/num_infected

test_sampler = WeightedRandomSampler(weights, replacement=True, num_samples=len(weights))
test_dataloader = DataLoader(val_dataset, batch_size = 1, sampler=test_sampler)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

model = model.to("cuda")

best_loss = 100
for epoch in range(200):
    model.train()
    for x, y in train_dataloader:
        x += torch.randn_like(x)/100
        #plt.imshow(x[0].permute((1, 2, 0)))
        #plt.show()
        x, y = x.to("cuda"), y.to("cuda")
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    correct = 0
    n = 0
    totloss = 0
    a = 0
    b = 0
    for x, y in test_dataloader:
        x, y = x.to("cuda"), y.to("cuda")
        y_pred = model(x)
        #print(y)
        #plt.imshow(x[0].permute((1, 2, 0)))
        #plt.show()
        loss = criterion(y_pred, y)
        y_pred = model(x)[0]
        y = y[0]
        if y == 0:
            a += 1
        else:
            b += 1
        chosen = torch.argmax(y_pred)
        if chosen == y:
            correct += 1
        n += 1
        totloss += loss.item()
    print(a, b)
    totloss /= len(test_dataloader)
    prop = correct/n
    if totloss < best_loss:
        torch.save(model.state_dict(), "disease_classifier.pt")
    best_loss = totloss
    print(prop, totloss, "test")
    correct = 0
    n = 0
    totloss = 0
    for x, y in train_dataloader:
        x, y = x.to("cuda"), y.to("cuda")
        y_pred = model(x)
        #print(y)
        #plt.imshow(x[0].permute((1, 2, 0)))
        #plt.show()
        loss = criterion(y_pred, y)
        y_pred = model(x)[0]
        y = y[0]
        chosen = torch.argmax(y_pred)
        if chosen == y:
            correct += 1
        n += 1
        totloss += loss.item()
    totloss /= len(test_dataloader)
    prop = correct/n
    print(prop, totloss, "train")
model = model.to("cpu")
correct = 0
n = 0
for x, y in tqdm(test_dataloader):
    y_pred = model(x)
    #print(y)
    #plt.imshow(x[0].permute((1, 2, 0)))
    #plt.show()
    loss = criterion(y_pred, y)
    y_pred = model(x)[0]
    y = y[0]
    chosen = torch.argmax(y_pred)
    if chosen == y:
        correct += 1
    n += 1