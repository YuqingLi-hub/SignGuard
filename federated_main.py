import numpy as np # linear algebra
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from data_loader import get_dataset
from running import test_classification, benignWorker, byzantineWorker
from models import CNN, ResNet18, ResNet9
from aggregators import aggregator
from attacks import attack
from options import args_parser
from torch.nn.utils import parameters_to_vector
import tools
import time
import copy
from watermarks.qim import QIM

# make sure that there exists CUDA，and show CUDA：
# print(device)
#
# attacks : non, random, noise, signflip, label_flip, byzMean.
#           lie, min_max, min_sum, *** adaptive (know defense) ***
#
# defense : Mean, TrMean, Median, GeoMed, Multi-Krum, Bulyan, DnC, SignGuard.

# set training hype-parameters
# arguments dict

# args = {
#     "epochs": 60,
#     "num_users": 50,
#     "num_byzs": 10,
#     "frac": 1.0,
#     "local_iter": 1,
#     "local_batch_size": 50,
#     "optimizer": 'sgd',
#     "agg_rule": 'SignCheck',
#     "attack": 'non',
#     "lr": 0.2,
#     "dataset": 'cifar',
#     "iid": True,
#     "unbalance": False,
#     "device": device
# }

###########################
'''
server send global model to clients
clients receive global model and train local model

how to let client only use m to recover the global model? (without alpha, )
'''
if __name__ == '__main__':
    # TODO: set random seed, numpy
    np.random.seed(2021)
    args = args_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(args.dataset)
    # load dataset and user groups
    train_loader, test_loader = get_dataset(args)
    # construct model
    if args.dataset == 'cifar':
        # if args.agg_rule == 'AlignIns':
        #     global_model = ResNet9()
        # else:
        global_model = ResNet18()
    elif args.dataset == 'fmnist':
        global_model = CNN().to(device)
    else:
        global_model = CNN().to(device)
    if device.type == 'cuda':
        global_model = global_model.cuda()
    else:
        global_model = global_model.cpu()

    # optimizer
    optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                       momentum=0.9, weight_decay=0.0005)
    scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # Training
    # number of iteration per epocj
    iteration = len(train_loader[0].dataset) // (args.local_bs*args.local_iter)
    train_loss, train_acc = [], []
    test_acc = []
    byz_rate = []
    benign_rate = []

    # attack method
    attack_list = ['random', 'sign_flip', 'noise', 'label_flip', 'lie', 'byzMean', 'min_max', 'min_sum', 'non']
    # attack_id = np.random.randint(9)
    #args.attack = attack_list[attack_id]
    Attack = attack(args.attack)

    # Gradient Aggregation Rule
    GAR = aggregator(args.agg_rule)()
    

    def train_parallel(args, model, train_loader, optimizer, epoch, scheduler):
        print(f'\n---- Global Training Epoch : {epoch+1} ----')
        num_users = args.num_users
        num_byzs = args.num_byzs
        # num_byzs = np.random.randint(1,20)
        device = args.device
        iter_loss = []
        data_loader = []
        mask = None
        for idx in range(num_users):
            data_loader.append(iter(train_loader[idx]))

        for it in range(iteration):

            m = max(int(args.frac * num_users), 1)
            idx_users = np.random.choice(range(num_users), m, replace=False)
            idx_users = sorted(idx_users)
            local_losses = []
            benign_grads = []
            byz_grads = []
            agent_data_sizes = {}
            
            for idx in idx_users[:num_byzs]:
                # if mask is not None:
                #     args.masks = mask
                #     args.watermark = Watermark
                #     grad, loss = byzantineWorker(model, data_loader[idx], optimizer, args, watermark=True)  
                grad, loss = byzantineWorker(model, data_loader[idx], optimizer, args)
                byz_grads.append(grad)
                agent_data_sizes[idx] = len(data_loader[idx])

            for idx in idx_users[num_byzs:]:
                # if mask is not None:
                #     args.masks = mask
                #     args.watermark = Watermark
                #     grad, loss = benignWorker(model, data_loader[idx], optimizer, device, args, watermark=True)
                grad, loss = benignWorker(model, data_loader[idx], optimizer, args)
                benign_grads.append(grad)
                local_losses.append(loss)
                agent_data_sizes[idx] = len(data_loader[idx])

            # get byzantine gradient
            byz_grads = Attack(byz_grads, benign_grads, GAR)
            # get all local gradient
            local_grads = byz_grads + benign_grads
            # for alignIns, we need to flatten the gradients
            # flatten_global_grad = parameters_to_vector(
            #     [model.state_dict()[name] for name in model.state_dict()]).detach()
            flatten_global_grad = tools.get_parameter_values(model)
            # get global gradient
            global_grad, selected_idx, isbyz = GAR.aggregate(local_grads, f=num_byzs, epoch=epoch, g0=flatten_global_grad, agent_data_sizes=agent_data_sizes, iteration=it)
            # masks = GAR.masks if hasattr(GAR, 'masks') else None
            # sign_message = torch.sign(global_grad[:,masks[0][0]:masks[0][1]]) if masks is not None else None
            # global_grad_w = Watermark.embed(global_grad, sign_message)
            byz_rate.append(isbyz)
            benign_rate.append((len(selected_idx)-isbyz*num_byzs)/(num_users-num_byzs))
            # update global model
            tools.set_gradient_values(model, global_grad)
            optimizer.step()

            loss_avg = sum(local_losses) / len(local_losses)
            iter_loss.append(loss_avg)

            if (it + 1) % 10 == 0:  # print every 10 local iterations
                print('[epoch %d, %.2f%%] loss: %.5f' %
                      (epoch + 1, 100 * ((it + 1)/iteration), loss_avg), "--- byz. attack succ. rate:", isbyz, '--- selected number:', len(selected_idx))

        if scheduler is not None:
            scheduler.step()

        return iter_loss

    for epoch in range(args.epochs):
        loss = train_parallel(args, global_model, train_loader, optimizer, epoch, scheduler)
        acc = test_classification(device, global_model, test_loader)
        print("Test Accuracy: {}%".format(acc))
        # test watermark etc.
        alpha = 0.7
        k = 0
        d = 20
        Watermark = QIM(delta=d)
        flatten_global_grad = tools.get_parameter_values(global_model).cpu().detach().numpy()
        #########################################################
        # test
        grad_test = copy.deepcopy(flatten_global_grad) + torch.randn_like(tools.get_parameter_values(global_model)) * 3
        print('distortion:',np.mean(np.abs(flatten_global_grad - global_w.cpu().detach().numpy())))
        tools.set_gradient_values(global_model, grad_test)
        acc_test = test_classification(device, global_model, test_loader)
        print("Test Accuracy with noise: {}%".format(acc_test))
        #########################################################
        # random message
        message = Watermark.random_msg(len(flatten_global_grad))
        # embedding watermark to whole global gradient
        global_w = torch.tensor(Watermark.embed(flatten_global_grad, message, alpha=alpha,k=k),dtype=torch.float32).to(device)
        print('distortion:',np.mean(np.abs(flatten_global_grad - global_w.cpu().detach().numpy())))
        tools.set_gradient_values(global_model, global_w)
        acc_w = test_classification(device, global_model, test_loader)
        # detect and recover watermark
        global_w = global_w.cpu().detach().numpy()
        reconstructed_grad, m = Watermark.detect(global_w, alpha=alpha,k=k)
        print("Watermark acc:", np.mean(message == m))
        print("Watermarked model Test Accuracy: {}%".format(acc_w))
        print("Reconstructed Gradient Distortion:", np.mean(np.abs(flatten_global_grad - reconstructed_grad)))
        reconstructed_grad = torch.tensor(reconstructed_grad, dtype=torch.float32).to(device)
        tools.set_gradient_values(global_model, reconstructed_grad)
        acc_recovered = test_classification(device, global_model, test_loader)
        print("Recovered model Test Accuracy: {}%".format(acc_recovered))
        