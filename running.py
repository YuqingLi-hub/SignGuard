
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from attacks import *
import tools
import numpy as np
import time
from watermarks.modi_qim import QIM
import copy
from torch.nn.utils import vector_to_parameters
from tools import embedding_watermark_on_position,detect_recover_on_position

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
        masks = args.masks if hasattr(args, 'masks') else [0,1000]
        # print(tools.get_parameter_values(model))
        # print('embedding locations Watermark masks: ',masks)
        Watermark = None
        # detecting and restoring watermark
        if watermark:
            global_grads = tools.get_parameter_values(model)
            _global_grads = copy.deepcopy(global_grads)
            Watermark = QIM(args.delta)
            ################# args.alpha for now #################
            _global_grads,message = detect_recover_on_position(masks=masks,whole_grads=_global_grads,Watermark=Watermark,args=args)
            print(global_grads[masks[0]:masks[1]][:10],'recovered',_global_grads[masks[0]:masks[1]][:10])
            # err = torch.mean(torch.abs(_global_grads[masks[0]:masks[1]] 
            #                - torch.tensor(Watermark.x, dtype=torch.float32).to(device)))
            # print("Average recovery error:", err.item())
            # print('Client Recovery completed: ', torch.allclose(_global_grads[masks[0]:masks[1]],torch.tensor(Watermark.x,dtype=torch.float32).to(device)))
            vector_to_parameters(_global_grads, model.parameters())
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

        # embedding watermark
        if watermark:
            _user_grad = copy.deepcopy(user_grad)
            user_grad_w = embedding_watermark_on_position(masks,_user_grad,Watermark,message,args)
        # print('Client Watermarking completed: ', torch.allclose(user_grad,user_grad_w))
        print()
        return user_grad_w, loss.item(), args.alpha

    def byzantineWorker(self, model, train_loader, optimizer, args, watermark=False):
        device = args.device
        attack = args.attack
        global_grads = None
        masks = args.masks if hasattr(args, 'masks') else [0,1000]
        # print('embedding locations Watermark masks: ',masks)
        # print(tools.get_parameter_values(model))
        Watermark = None
        if watermark:
            global_grads = tools.get_parameter_values(model)
            _global_grads = copy.deepcopy(global_grads)
            Watermark = QIM(args.delta)
            ################# args.alpha for now #################
            _global_grads,message = detect_recover_on_position(masks=masks,whole_grads=_global_grads,Watermark=Watermark,args=args)
            print(global_grads[masks[0]:masks[1]][:10],'recovered',_global_grads[masks[0]:masks[1]][:10])
            # print(global_grads[masks[0]:masks[1]][:10],Watermark.x[:10],'recovered',_global_grads[masks[0]:masks[1]][:10])
            # err = torch.mean(torch.abs(_global_grads[masks[0]:masks[1]] 
            #                - torch.tensor(Watermark.x, dtype=torch.float32).to(device)))
            # print("Average recovery error:", err.item())
            # print('Client Recovery completed: ', torch.allclose(_global_grads[masks[0]:masks[1]],torch.tensor(Watermark.x,dtype=torch.float32).to(device)))
            vector_to_parameters(_global_grads, model.parameters())
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
        # embedding watermark
        if watermark:
            _user_grad = copy.deepcopy(user_grad)
            user_grad_w = embedding_watermark_on_position(masks,_user_grad,Watermark,message,args)
        # print('Client Watermarking completed: ', torch.allclose(user_grad,user_grad_w))
        print()
        
        return user_grad_w, loss.item(), args.alpha

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