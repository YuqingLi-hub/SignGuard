
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
# from logger import get_logger
# logger = get_logger()
class Worker():
    def __init__(self,delta,alpha,k,secret_param='mean'):
        self.init_delta = delta
        self.init_alpha = alpha
        self.init_k = k
        # if secret_param == 'mean':
        #     self.
def all_worker(model, train_loader, optimizer, args, water=False,malicious=True):
    logger = args.logger
    device = args.device
    attack = args.attack
    masks = args.masks if hasattr(args, 'masks') else [0, 1000]
    alpha = args.alpha
    k = args.k
    delta = args.delta
    Qim = None
    # recovered_grads = None
    global_param = tools.get_parameter_values(model)
    logger.debug(f'Client Received parameters {global_param[masks[0]:masks[1]][:10]}')
    # ------------------ Updating model parameters - Watermark detection & recovery ------------------
    if water:
        # print('Received gradients',global_grads[masks[0]:masks[1]][:10])
        Qim = QIM(delta)

        # Recover watermark
        global_param, message = detect_recover_on_position(
            masks=masks,
            whole_grads=global_param,
            Watermark=Qim,
            alpha=alpha,
            k=k,
            model=model
        )
        logger.debug(f'Client Recovered params {global_param[masks[0]:masks[1]][:10]}')
        # print('Recovered gradients',recovered_grads[masks[0]:masks[1]][:10])
        # with torch.no_grad():
        #     start = 0
        #     for p in model.parameters():
        #         numel = p.numel()
        #         g = global_param[start:start+numel].view_as(p)
        #         p = g
        #         start += numel
        logger.debug(f'Updated recovered params {tools.get_parameter_values(model)[masks[0]:masks[1]][:10]}')
    # ------------------ Training step ------------------
    model.train()
    criterion = nn.CrossEntropyLoss()

    images, labels = next(train_loader)
    if attack == 'label_flip' and malicious:
        labels = 9 - labels
    images, labels = images.to(device), labels.to(device)

    # zero local grads
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    # optimizer.step()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()

    # Get updated grads/params
    updated_grads = tools.get_gradient_values(model)
    logger.debug(f'Updated grads (first 10): {updated_grads[masks[0]:masks[1]][:10]}')

    # ------------------ Embed watermark if required ------------------
    if water:
        _user_param = copy.deepcopy(updated_grads)
        user_grad_w = embedding_watermark_on_position(
            masks, _user_param, Qim, message, alpha=alpha,k=k, model=model
        )
        logger.debug(f'Watermarked (first 10): {user_grad_w[masks[0]:masks[1]][:10]}')
    else:
        user_grad_w = updated_grads
  
    return user_grad_w, loss.item(), alpha
    # # ------------------ Training step ------------------
    # model.train()
    
    # if recovered_grads is not None:
    #     tools.update_gradient(recovered_grads,model=model)
    # before_train = tools.get_gradient_values(model)
    # logger.info(f'Client Recovered gradients {before_train[masks[0]:masks[1]][:10]}')
    # criterion = nn.CrossEntropyLoss()
    # images, labels = next(train_loader)
    # if attack=='label_flip' and malicious:
    #     labels = 9 - labels
    # images, labels = images.to(device), labels.to(device)
    # optimizer.zero_grad()
    # outputs = model(images)
    # loss = criterion(outputs, labels)
    # loss.backward()
    # # user_grad = tools.get_gradient_values(model)

    # # Get updated parameters
    # updated_params = tools.get_gradient_values(model)
    # # print('Weights updated?', not torch.allclose(before_train, updated_params))
    # # print('Updated params (first 10):', updated_params[masks[0]:masks[1]][:10])
    # logger.info(f'Updated (first 10):{updated_params[masks[0]:masks[1]][:10]}')

    # # self.previous_grad = updated_params

    # # ------------------ Embed watermark if required ------------------
    # if water:
    #     _user_grad = copy.deepcopy(updated_params)
    #     user_grad_w = embedding_watermark_on_position(masks, _user_grad, Qim, message, args,model=model)
    #     logger.info(f'Watermarked (first 10):{user_grad_w[masks[0]:masks[1]][:10]}')
    #     # print()
    # else:
    #     user_grad_w = updated_params
    # return user_grad_w, loss.item(), args.alpha

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

