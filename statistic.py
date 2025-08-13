import re
import os
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import date
date = date.today().strftime("%Y-%m-%d_")
import torch
import os
from models import MLP, SmallVGG, ResNet18, ResNet9, CNN
from watermarks.qim import QIM
from watermarks.utils import RQIM
import tools
from data_loader import get_dataset
from torch.nn.utils import vector_to_parameters
from running import test_classification
import copy
from options import args_parser


def extract_statistics(log_file,model):
    # Load your log text
    with open(log_file, "r") as f:
        log_text = f.read()
    if model == 'Al':
        byz_rates = re.findall(r"byz\. attack succ\. rate: (\d+)", log_text)
    else:
    # Extract all byzantine attack success rates
        byz_rates = re.findall(r"byz\. attack succ\. rate: ([\d.]+)", log_text)
    byz_rates = [float(rate) for rate in byz_rates]

    # Extract all test accuracies
    test_accuracies = re.findall(r"Test Accuracy: ([\d.]+)%", log_text)
    test_accuracies = [float(acc) for acc in test_accuracies]

    # Extract all selected numbers
    selected_numbers = re.findall(r"--- selected number: (\d+)", log_text)
    selected_numbers = [int(num) for num in selected_numbers]

    # Get the most recent values (or all if you want)
    avg_byz_rate = np.mean(np.array(byz_rates)) if byz_rates else None
    best_test_acc = max(test_accuracies) if test_accuracies else None
    avg_selected_num = np.mean(np.array(selected_numbers)) if selected_numbers else None

    print("Average Byzantine Attack Success Rate:", avg_byz_rate)
    print("Best Test Accuracy:", best_test_acc)
    print("Selected Numbers:", avg_selected_num)
def extract_acc(log_file,job_name):
    # Load your log text
    with open(log_file, "r") as f:
        log_text = f.read()

    # Extract all test accuracies
    test_accuracies = re.findall(r"Recovered model Test Accuracy: ([\d.]+)%", log_text)
    test_accuracies = [float(acc) for acc in test_accuracies]
    watermark_accuracies = re.findall(r"Watermarked model Test Accuracy: ([\d.]+)%", log_text)
    watermark_accuracies = [float(acc) for acc in watermark_accuracies]
    std_change = re.findall(r"Std: ([\d.]+)", log_text)
    std_change = [float(cc) for cc in std_change]
    epic = range(len(test_accuracies))
    # Get the best test accuracy
    best_test_acc = max(test_accuracies) if test_accuracies else None
    plt.figure(figsize=(12, 6))
    plt.plot(epic, test_accuracies, marker='o', linestyle='-', markersize=4, color='green',label="Recovered Model")
    plt.plot(epic, watermark_accuracies, marker="o", color="red", label="Watermarked Model")
    # plt.axvline(x=delta, color='r', linestyle='--', label=f'True delta')
    plt.title(f'Watermarked Accuracy vs. Recovery Accuracy for {job_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs(f"./outputs/qim/{date}", exist_ok=True)
    plt.savefig(f"./outputs/qim/{date}/{job_name}_statistic_Re_Wa_acc.png")
    plt.close()
    print("Best Test Accuracy:", best_test_acc)

    plt.figure(figsize=(12, 6))
    plt.plot(epic, std_change, marker='o', linestyle='-', markersize=4, color='green',label="Recovered Model")
    # plt.plot(epic, watermark_accuracies, marker="o", color="red", label="Watermarked Model")
    # plt.axvline(x=delta, color='r', linestyle='--', label=f'True delta')
    plt.title(f'STD change in Epoches for {job_name}')
    plt.xlabel('Epoch')
    plt.ylabel('std')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs(f"./outputs/qim/{date}", exist_ok=True)
    plt.savefig(f"./outputs/qim/{date}/{job_name}_statistic_std.png")
    plt.close()
    # print("Best Test Accuracy:", best_test_acc)

def test_watermark_acc():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = args_parser()
    # args.dataset = 'cifar10'
    # args.iid = True
    # args.num_users = 50
    train_loader, test_loader = get_dataset(args)
    model = ResNet18()
    model.load_state_dict(torch.load("./outputs/Rqim_model/model_100.pth", map_location=torch.device("cpu")))
    model.eval()

    alpha = 0.8675
    k = 0
    delta = 0.1
    # alpha = 0.8675
    # k = 0
    # delta = 1


    
    flatten_global_grad = tools.get_parameter_values(model)
    num_spars = len(flatten_global_grad)* np.array([0.02,0.04,0.06,0.08,0.1])
    # num_spars = [396, 760, 1177, 1604, 1982]

    for n in num_spars:
        Watermark = QIM(delta=delta)
        n = int(n)
        idx = torch.randint(0, (len(flatten_global_grad) - int(n)),size=(1,)).item()
        print(n)
        gradss = copy.deepcopy(flatten_global_grad[idx:(idx+int(n))])
        print("grad_unwater mean:", gradss.mean().item(),'grad_unwater std:', gradss.std().item())


        message = np.random.choice((0, 1), (len(gradss)))
        #embedding watermark to whole global gradient
        # global_w = torch.tensor(Watermark.embed(flatten_global_grad, message, alpha=alpha,k=k),dtype=torch.float32).to(device)
        
        t_ =  copy.deepcopy(gradss.cpu().detach().numpy())
        # print(len(t_))
        w_ = Watermark.embed(t_,m=message,alpha=alpha,k=k)
        r_w,mm = Watermark.detect(w_,alpha=alpha,k=k)
        print("Test Reconstructed Gradient Error:", np.mean(np.abs(t_ - r_w)))


        global_w = torch.tensor(w_,dtype=torch.float32).to(device)
        flatten_global_grad[idx:(idx+n)] = global_w
        m = mm
        reconstructed_grad = torch.tensor(r_w,dtype=torch.float32).to(device)

        model_watermark = copy.deepcopy(model)
        vector_to_parameters(flatten_global_grad, model_watermark.parameters())
        acc_w = test_classification(device, model_watermark, test_loader)
        # detect and recover watermark
        # global_w = global_w.cpu().detach().numpy()
        # reconstructed_grad, m = Watermark.detect(global_w, alpha=alpha,k=k)
        print('Distortion wat v.s. ori:',torch.mean(torch.abs(gradss - global_w)))
        print("Watermark acc:", np.mean(message == m))
        print("Watermarked model Test Accuracy: {}%".format(acc_w))
        print("Reconstructed Gradient Error (should be same as Test Reconstructed Gradient Error):", torch.mean(torch.abs(gradss - reconstructed_grad)))
        # reconstructed_grad = torch.tensor(reconstructed_grad, dtype=torch.float32).to(device)
        flatten_global_grad[idx:(idx+n)] = reconstructed_grad
        model_recover = copy.deepcopy(model)
        vector_to_parameters(flatten_global_grad, model_recover.parameters())
        acc_recovered = test_classification(device, model_recover, test_loader)
        print("Recovered model Test Accuracy: {}%".format(acc_recovered))



if __name__ == '__main__':
    # attack_list = ['random', 'sign_flip', 'noise', 'label_flip', 'lie', 'byzMean', 'min_max', 'min_sum', 'non','nan','zero']
    # model = 'SG-SIM'
    
    # for attack in attack_list:
    #     print(f"Statistics for attack: {attack}")
    #     # logfile = f"./outputs/{model}_{attack}00001-out.txt"
    #     logfile = f"./outputs/{model}test00001-{attack}-out.txt"
    #     if os.path.exists(logfile):
    #     #     logfile = f"./outputs/{model}_{attack}00002-out.txt"
            
    #     # else:
    #         # logfile = f"./outputs/{model}_{attack}00001-out.txt"
            
    #         extract_statistics(logfile,model)
    #         print("\n")
    #     # extract_statistics("./outputs/SGtest00005-out.txt",model)
    job_name = 'RQimTest00007' #'RQimTest00006-7-2'
    extract_acc(f"./outputs/R-QIM/{job_name}-out.txt",job_name)
    test_watermark_acc()
 

