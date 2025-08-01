
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from attacks import *
import tools
import numpy as np
import time
from watermarks.qim import QIM

def benignWorker(model, train_loader, optimizer, args, watermark=False):
    device = args.device
    mask = None
    if watermark:
        Watermark = args.watermark
        mask = args.masks
        glbal_grad_w = tools.get_gradient_values(model)
        global_grad, m = Watermark.detect(glbal_grad_w)
        tools.set_gradient_values(model, global_grad)
    model.train()
    criterion = nn.CrossEntropyLoss()
    images, labels = next(train_loader)
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    user_grad = tools.get_gradient_values(model)
    # shuffle the gradient use the sign (m)
    # user_grad = lfea(m)
    return user_grad, loss.item()

def byzantineWorker(model, train_loader, optimizer, args, watermark=False):
    device = args.device
    attack = args.attack
    if watermark:
        Watermark = args.watermark
        mask = args.masks
        glbal_grad_w = tools.get_gradient_values(model)
        global_grad, m = Watermark.detect(glbal_grad_w)
        tools.set_gradient_values(model, global_grad)
    model.train()
    criterion = nn.CrossEntropyLoss()
    images, labels = next(train_loader)
    if attack=='label_flip':
        labels = 9 - labels
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    user_grad = tools.get_gradient_values(model)

    return user_grad, loss.item()

# define model testing function
def test_classification(device, model, test_loader):
    model.eval()
    correct = 0
    total = len(test_loader.dataset)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # pred = output.max(1, keepdim=True)[1]
            correct += (predicted == labels).sum().item()
    acc = 100.0*correct/total
    # print('Test Accuracy: %.2f %%' % (acc))

    return acc