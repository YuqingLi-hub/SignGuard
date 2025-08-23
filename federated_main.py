import numpy as np # linear algebra
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from data_loader import get_dataset
from running import test_classification, benignWorker, byzantineWorker
from models import CNN, ResNet18, ResNet9, SmallVGG, MLP
from aggregators import aggregator
from attacks import attack
from options import args_parser
from torch.nn.utils import parameters_to_vector
import tools
import time
import copy
from watermarks.qim import QIM
from torch.nn.utils import vector_to_parameters
import os
from datetime import date
date = date.today().strftime("%Y-%m-%d_")
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
    #############################################################################
    # cover only necessary ones

def embedding_watermark_on_position(masks,whole_grads,args):
    alpha = args.alpha
    k = args.k
    ini_delta = args.delta
    # print(whole_grads.shape)
    grad_unwater = copy.deepcopy(whole_grads[masks[0]:masks[1]])
    print("grad_unwater mean:", grad_unwater.mean().item(),'grad_unwater std:', grad_unwater.std().item())
    sign_mask = torch.sign(grad_unwater) 
    num_spars = masks[1]-masks[0]
    # get the sign of the gradients, and sum the sign gradients
    # sign_grads = torch.sign(gradss)
    # print("sign_grads", sign_grads)
    sign_pos = (sign_mask.eq(1.0)).sum(dtype=torch.float32)/(num_spars)
    sign_zero = (sign_mask.eq(0.0)).sum(dtype=torch.float32)/(num_spars)
    sign_neg = (sign_mask.eq(-1.0)).sum(dtype=torch.float32)/(num_spars)
    print('+',sign_pos,'0',sign_zero, '-', sign_neg)
    print('masks:', masks)
    # if torch.isclose(sign_pos,0): 
    #     sign_pos = 1
    true_delta = float(ini_delta*sign_pos+1e-5)
    print("True delta:", true_delta)
    Watermark = QIM(delta=true_delta)
    message = Watermark.random_msg(num_spars)

    t_ = grad_unwater.cpu().detach().numpy()
    w_ = Watermark.embed(t_,m=message,alpha=alpha,k=k)

    r_w,mm = Watermark.detect(w_,alpha=alpha,k=k)
    print("Test Reconstructed Gradient Error:", np.mean(np.abs(t_ - r_w)))
    w_grad = torch.tensor(w_,dtype=torch.float32).to(device)
    m = mm
    reconstructed_grad = torch.tensor(r_w,dtype=torch.float32).to(device)
    with torch.no_grad():
        whole_grads[masks[0]:masks[1]].copy_(w_grad)
    # whole_grads[masks[0]:masks[1]] = w_grad
    print('Correctly update grads: ', torch.allclose(whole_grads[masks[0]:masks[1]],w_grad))
    print('Distortion wat v.s. ori:',torch.mean(torch.abs(grad_unwater - w_grad)))
    print("Watermark acc:", np.mean(message == m))
    print("Reconstructed Gradient Error (should be same as Test Reconstructed Gradient Error):", torch.mean(torch.abs(grad_unwater - reconstructed_grad)))
    return whole_grads

if __name__ == '__main__':
    # TODO: set random seed, numpy
    np.random.seed(2021)
    torch.manual_seed(2021)
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
        # global_model = SmallVGG()
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
            global_grad, selected_idx, isbyz = GAR.aggregate(local_grads, f=num_byzs, epoch=epoch, g0=flatten_global_grad, agent_data_sizes=agent_data_sizes, iteration=it,attack=args.attack)
            
            if args.random_watermark:
                if hasattr(GAR, 'masks'):
                    masks = GAR.masks if hasattr(GAR, 'masks') else None
                    if masks is not None:
                        print('---------------------------------------------------')
                        # print(masks,masks[0],masks[1])
                        watermarked_grad = embedding_watermark_on_position(masks=masks,whole_grads=global_grad,args=args)
                        model_watermark = copy.deepcopy(model)
                        vector_to_parameters(watermarked_grad, model_watermark.parameters())
                        acc_w = test_classification(device, model_watermark, test_loader)
                        print("Watermarked model Test Accuracy: {}%".format(acc_w))
                        print()

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
    alpha = args.alpha
    k = args.k
    d = args.delta
    args.random_watermark = True
    print("Watermarking with alpha: {}, delta: {}, k: {}".format(alpha, d, k))
    for epoch in range(args.epochs):
        loss = train_parallel(args, global_model, train_loader, optimizer, epoch, scheduler)
        acc = test_classification(device, global_model, test_loader)
        print("Test Accuracy: {}%".format(acc))

        epoch_watermark_check = True

        if epoch_watermark_check:
            #-------------------------------------------------------------
            # test watermark etc.
            Watermark = QIM(delta=d)
            flatten_global_grad = tools.get_parameter_values(global_model)
            #########################################################
            # print(flatten_global_grad.mean(), flatten_global_grad.std(), flatten_global_grad.min(), flatten_global_grad.max())
            # testnp.random.randn(*a.shape)
            print("Param mean:", flatten_global_grad.mean().item())
            print("Std:", flatten_global_grad.std().item())
            # grad_test = copy.deepcopy(flatten_global_grad) + torch.randn_like(flatten_global_grad) * 0.1
            # print('distortion:',torch.mean(torch.abs(flatten_global_grad - grad_test)))
            # model_copy = copy.deepcopy(global_model)
            # vector_to_parameters(torch.tensor(grad_test,dtype=torch.float32).to(device), model_copy.parameters())
            # acc_test = test_classification(device, model_copy, test_loader)
            # print("Test Accuracy with noise: {}%".format(acc_test))
            #########################################################
            # random message
            message = Watermark.random_msg(len(flatten_global_grad))
            #embedding watermark to whole global gradient
            # global_w = torch.tensor(Watermark.embed(flatten_global_grad, message, alpha=alpha,k=k),dtype=torch.float32).to(device)
            
            t_ =  copy.deepcopy(flatten_global_grad.cpu().detach().numpy())
            # print(len(t_))
            w_ = Watermark.embed(t_,m=message,alpha=alpha,k=k)
            r_w,mm = Watermark.detect(w_,alpha=alpha,k=k)
            print("Test Reconstructed Gradient Error:", np.mean(np.abs(t_ - r_w)))


            global_w = torch.tensor(w_,dtype=torch.float32).to(device)
            m = mm
            reconstructed_grad = torch.tensor(r_w,dtype=torch.float32).to(device)

            model_watermark = copy.deepcopy(global_model)
            vector_to_parameters(global_w, model_watermark.parameters())
            acc_w = test_classification(device, model_watermark, test_loader)
            # detect and recover watermark
            # global_w = global_w.cpu().detach().numpy()
            # reconstructed_grad, m = Watermark.detect(global_w, alpha=alpha,k=k)
            print('Distortion wat v.s. ori:',torch.mean(torch.abs(flatten_global_grad - global_w)))
            print("Watermark acc:", np.mean(message == m))
            print("Watermarked model Test Accuracy: {}%".format(acc_w))
            print("Reconstructed Gradient Error (should be same as Test Reconstructed Gradient Error):", torch.mean(torch.abs(flatten_global_grad - reconstructed_grad)))
            # reconstructed_grad = torch.tensor(reconstructed_grad, dtype=torch.float32).to(device)
            model_recover = copy.deepcopy(global_model)
            vector_to_parameters(reconstructed_grad, model_recover.parameters())
            acc_recovered = test_classification(device, model_recover, test_loader)
            print("Recovered model Test Accuracy: {}%".format(acc_recovered))
        if epoch%100 == 0:
            os.makedirs(f"./outputs/Rqim_model/", exist_ok=True)
            torch.save(global_model.state_dict(), f'./outputs/Rqim_model/model_{epoch}.pth')

