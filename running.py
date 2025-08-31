
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

        
def all_worker(model, train_loader, optimizer, args, water=False,malicious=True):
    device = args.device
    attack = args.attack
    masks = args.masks if hasattr(args, 'masks') else [0, 1000]
    Qim = None

    # ------------------ Watermark detection & recovery ------------------
    if water:
        # Get current parameters
        global_grads = tools.get_gradient_values(model)
        print('Received gradients',global_grads[masks[0]:masks[1]][:10])
        Qim = QIM(args.delta)

        # Recover watermark
        recovered_grads, message = detect_recover_on_position(
            masks=masks,
            whole_grads=global_grads,
            Watermark=Qim,
            args=args,
            model=model
        )
        print('Recovered gradients',recovered_grads[masks[0]:masks[1]][:10])

    # ------------------ Training step ------------------
    model.train()
    criterion = nn.CrossEntropyLoss()
    images, labels = next(train_loader)
    if attack=='label_flip' and malicious:
        labels = 9 - labels
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    # user_grad = tools.get_gradient_values(model)

    # Get updated parameters
    updated_params = tools.get_gradient_values(model)
    # print('Weights updated?', not torch.allclose(before_train, updated_params))
    # print('Updated params (first 10):', updated_params[masks[0]:masks[1]][:10])

    # self.previous_grad = updated_params

    # ------------------ Embed watermark if required ------------------
    if water:
        _user_grad = copy.deepcopy(updated_params)
        user_grad_w = embedding_watermark_on_position(masks, _user_grad, Qim, message, args,model=model)
        print('Watermarked (first 10):', user_grad_w[masks[0]:masks[1]][:10])
        print()
    else:
        user_grad_w = updated_params
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


def get_parameter_values(model):
    """ Get (optionally make each parameter's gradient) a reference to the flat gradient.
    Returns:
      Flat gradient (by reference: future calls to 'set_gradient' will modify it)
    """

    parameter = torch.cat([torch.reshape(param.data, (-1,)) for param in model.parameters()])
    return parameter