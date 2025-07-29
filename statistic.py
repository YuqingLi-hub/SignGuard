import re
import os
import numpy as np
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

if __name__ == '__main__':
    attack_list = ['random', 'sign_flip', 'noise', 'label_flip', 'lie', 'byzMean', 'min_max', 'min_sum', 'non','nan','zero']
    model = 'SG-SIM'
    
    for attack in attack_list:
        print(f"Statistics for attack: {attack}")
        # logfile = f"./outputs/{model}_{attack}00001-out.txt"
        logfile = f"./outputs/{model}test00001-{attack}-out.txt"
        if os.path.exists(logfile):
        #     logfile = f"./outputs/{model}_{attack}00002-out.txt"
            
        # else:
            # logfile = f"./outputs/{model}_{attack}00001-out.txt"
            
            extract_statistics(logfile,model)
            print("\n")
        # extract_statistics("./outputs/SGtest00005-out.txt",model)
