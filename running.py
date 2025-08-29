
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from attacks import *
import tools
import numpy as np
import time
from watermarks.qim import QIM
import copy

class Worker():
    def __init__(self,init_alpha=0.5,init_delta=1,init_k=0):
        self.previous_grad = None
        self.init_alpha = init_alpha
        self.init_delta = init_delta
        self.init_k = init_k
        
    def benignWorker(self, model, train_loader, optimizer, args, watermark=False):
        device = args.device
        attack = args.attack
        global_grads = None
        # print(tools.get_parameter_values(model))
        if watermark:
            global_grads = tools.get_parameter_values(model)
            Watermark = args.watermark
            masks = args.masks if hasattr(args, 'masks') else [0,len(global_grads)]
            global_grads_w = copy.deepcopy(global_grads[masks[0]:masks[1]])
            alpha = self.init_alpha
            delta = self.init_delta
            k = self.init_k
            t_ = global_grads_w.cpu().detach().numpy()
            global_grads_w, m = Watermark.detect(t_, alpha=alpha, k=k)
            print('Extracted message:',m)
            global_grads_w = torch.tensor(global_grads_w,dtype=torch.float32).to(device)
            with torch.no_grad():
                global_grads[masks[0]:masks[1]].copy_(global_grads_w)
            tools.set_gradient_values(model, global_grads)
        model.train()
        criterion = nn.CrossEntropyLoss()
        images, labels = next(train_loader)

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        user_grad = tools.get_parameter_values(model)
        self.previous_grad = user_grad
        if watermark:
            alpha = self.init_alpha

            delta = self.init_delta
            k = self.init_k
            user_grad_w = Watermark.embed(user_grad[masks[0]:masks[1]], m,alpha=alpha, delta=delta, k=k)
            with torch.no_grad():
                user_grad[masks[0]:masks[1]].copy_(user_grad_w)
        return user_grad, loss.item(), alpha

    def byzantineWorker(self, model, train_loader, optimizer, args, watermark=False):
        device = args.device
        attack = args.attack
        global_grads = None
        # print(tools.get_parameter_values(model))
        if watermark:
            global_grads = tools.get_parameter_values(model)
            Watermark = args.watermark
            masks = args.masks if hasattr(args, 'masks') else [0,len(global_grads)]
            global_grads_w = copy.deepcopy(global_grads[masks[0]:masks[1]])
            alpha = self.init_alpha
            delta = self.init_delta
            k = self.init_k
            t_ = global_grads_w.cpu().detach().numpy()
            global_grads_w, m = Watermark.detect(t_, alpha=alpha, k=k)
            print('Extracted message:',m)
            global_grads_w = torch.tensor(global_grads_w,dtype=torch.float32).to(device)
            with torch.no_grad():
                global_grads[masks[0]:masks[1]].copy_(global_grads_w)
            tools.set_gradient_values(model, global_grads)
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
        user_grad = tools.get_parameter_values(model)
        self.previous_grad = user_grad
        if watermark:
            alpha = self.init_alpha

            delta = self.init_delta
            k = self.init_k
            user_grad_w = Watermark.embed(user_grad[masks[0]:masks[1]], m,alpha=alpha, delta=delta, k=k)
            with torch.no_grad():
                user_grad[masks[0]:masks[1]].copy_(user_grad_w)
        return user_grad, loss.item(), alpha

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