import torch
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
import os
from .Mean import mean
class FLGMM():
    def __init__(self):
        self.name = "FLGMM"
        self.r = []
        self.p = []
        self.f1 = []
    def aggregate(self,gradients,f=10,iteration=1,ccepochs=50):
        num_users = len(gradients)
        save_dir = 'outputs/FLGMM/'
        distances_matrix = [[] for _ in range(num_users)]
        distances_matrix_this_round = []
        normal_id = []
        # normal_clients_dis = []
        # normal_clients_dis_mean = []
        # normal_std = []
        normal_dis = [[] for _ in range(num_users)]
        FedAvg_0 = mean().aggregate
        w_glob = FedAvg_0(gradients)
        excluded = []
        noisy_this_round = []
        for idx, w_local in enumerate(gradients):
            distance = euclidean_distance(w_local, w_glob)
            distances_matrix_this_round.append(distance)

        distances = distances_matrix_this_round
        distances_array = np.array(distances).reshape(-1, 1)

        # Utilize GMM
        largest_cluster_data, bounds, means, covariances, weights = decompose_normal_distributions(distances_array)
        print(len(largest_cluster_data))
        mean = np.mean(largest_cluster_data)
        std = np.std(largest_cluster_data)

        for idx in range(num_users):
            distances_matrix[idx].append((distances[idx]-mean)/std)
            if distances_matrix_this_round[idx] in largest_cluster_data and iteration < ccepochs:
                normal_id.append(idx)

        # Plot GMM in round 10
        if iteration == 10:
            flat_distances3 = distances_matrix_this_round
            plt.figure(figsize=(10, 6))
            plt.hist(flat_distances3, bins=50, color='grey', edgecolor='black', density=True, alpha=0.6)
            x = np.linspace(min(distances_array), max(distances_array), 1000)
            pdf_1 = weights[0] * (1 / (np.sqrt(2 * np.pi * covariances[0]))) * np.exp(
                -0.5 * ((x - means[0]) ** 2) / covariances[0])
            pdf_2 = weights[1] * (1 / (np.sqrt(2 * np.pi * covariances[1]))) * np.exp(
                -0.5 * ((x - means[1]) ** 2) / covariances[1])
            pdf_1 = pdf_1.reshape(-1)
            pdf_2 = pdf_2.reshape(-1)
            plt.plot(x, pdf_1, color='red', linestyle='-.', label='GMM Component 1')
            plt.plot(x, pdf_2, color='green', linestyle='-.', label='GMM Component 2')
            plt.title('Initial Distance Distribution with GMM Components')
            plt.xlabel('Distance')
            plt.ylabel('Density')
            plt.legend()
            plt.savefig(os.path.join(save_dir, 'GMM_distance_distribution_with_GMM_10rounds.png'))
            upper_bound = bounds[1]
            lower_bound = bounds[0]
            for idx, client_distances in enumerate(distances_matrix):
                normal_dis[idx] = [d for d in client_distances if d <= upper_bound]
            flat_distances1 = [distance for client_list in normal_dis for distance in client_list]
            plt.figure(figsize=(10, 6))
            plt.hist(flat_distances1, bins=40, color='grey', edgecolor='black', density=True)
            sns.kdeplot(flat_distances1, color='red')
            plt.title(f'Final Distance Distribution ')
            plt.xlabel('Distance')
            plt.ylabel('Density')
            plt.savefig(os.path.join(save_dir, f'Final_distance_distribution_10round.png'))
            plt.close()

        if iteration == ccepochs:
            normalid = []
            f_distances_matrix = [[] for _ in range(num_users)]
            all_distances = [distance for client_distances in distances_matrix for distance in client_distances]
            distances_array = np.array(all_distances)

            # Utilize GMM again
            largest_cluster_data, bounds, means, covariances, weights = decompose_normal_distributions(distances_array)
            upper_bound = bounds[1]
            lower_bound = bounds[0]
            for idx, client_distances in enumerate(distances_matrix):
                normal_dis[idx] = [d for d in client_distances if d <= upper_bound ]
            flat_distances3 = [distance for client_list in distances_matrix for distance in client_list]
            plt.figure(figsize=(10, 6))
            plt.hist(flat_distances3, bins=200, color='grey', edgecolor='black', density=True, alpha=0.6)
            x = np.linspace(min(distances_array), max(distances_array), 1000)
            pdf_1 = weights[0] * (1 / (np.sqrt(2 * np.pi * covariances[0]))) * np.exp(
                -0.5 * ((x - means[0]) ** 2) / covariances[0])
            pdf_2 = weights[1] * (1 / (np.sqrt(2 * np.pi * covariances[1]))) * np.exp(
                -0.5 * ((x - means[1]) ** 2) / covariances[1])
            pdf_1 = pdf_1.reshape(-1)
            pdf_2 = pdf_2.reshape(-1)
            plt.plot(x, pdf_1, color='red', linestyle='-.', label='GMM Component 1')
            plt.plot(x, pdf_2, color='green', linestyle='-.', label='GMM Component 2')
            plt.title('Initial Distance Distribution with GMM Components')
            plt.xlabel('Distance')
            plt.ylabel('Density')
            plt.legend()
            plt.savefig(os.path.join(save_dir, 'GMM_distance_distribution_with_GMM.png'))

            flat_distances1 = [distance for client_list in normal_dis for distance in client_list]
            plt.figure(figsize=(10, 6))
            plt.hist(flat_distances1, bins=50, color='grey', edgecolor='black', density=True)
            sns.kdeplot(flat_distances1, color='red')
            plt.title(f'Final Distance Distribution ')
            plt.xlabel('Distance')
            plt.ylabel('Density')
            plt.savefig(os.path.join(save_dir, f'Final_distance_distribution_round.png'))
            plt.close()

            # Erase clients in the component with a lager mean
            largest_cluster_data_2, bounds_2, means, covariances, weights = decompose_normal_distributions(
                largest_cluster_data)
            upper_bound_2 = bounds[1]
            lower_bound_2 = bounds[0]
            for idx, client_distances in enumerate(distances_matrix):
                normal_dis[idx] = [d for d in client_distances if d <= upper_bound_2]

            client_means = [np.mean(distances) for distances in distances_matrix]
            excluded_clients, UCL, LCL = plot_control_chart(np.arange(len(client_means)), client_means, normal_dis, save_dir)
            print("GMM detects:", excluded_clients)
            noisy_clients = [i for i in range(f)]
            r.append(calculate_accuracy(excluded_clients, noisy_clients)[0])
            p.append(calculate_accuracy(excluded_clients, noisy_clients)[1])
            recall = calculate_accuracy(excluded_clients, noisy_clients)[0]
            pre = calculate_accuracy(excluded_clients, noisy_clients)[1]
            f = 2 * recall * pre / (recall + pre)
            f1.append(f)
            print("Initial recall:", r[0])
            print("Initial precision:", p[0])
            print("Initial f1score:", f1[0])

        if iteration > ccepochs:
            for idx, client_distances in enumerate(distances_matrix):
                if client_distances[-1] < UCL:
                    normal_id.append(idx)
                else:
                    excluded.append(idx)
            excluded_clients = excluded
            print("Anomaly:", excluded)
            r.append(calculate_accuracy(excluded_clients, noisy_this_round)[0])
            p.append(calculate_accuracy(excluded_clients, noisy_this_round)[1])
            recall = calculate_accuracy(excluded_clients, noisy_clients)[0]
            pre = calculate_accuracy(excluded_clients, noisy_clients)[1]
            f = 2 * recall * pre / (recall + pre)
            f1.append(f)
            print("Recall:", r[o])
            print("Precision:", p[o])
            print("f1score:", f1[o])
            o += 1

        # Update global model
        if iteration < ccepochs:
            gradients_used = [gradients[i] for i in range(len(gradients)) if i in normal_id]
            print('numbers of participants:',len(gradients_used))

        else:
            gradients_used = [gradients[i] for i in range(len(gradients)) if i not in excluded_clients]
            print('numbers of participants:', len(gradients_used))

        if len(gradients_used) > 0:
            w_glob = FedAvg_0(gradients_used)
        #     net_glob.load_state_dict(w_glob)
        #     global_acc, global_loss = test_img(net_glob, dataset_test, args)
        #     global_acctotal.append(global_acc)
        #     global_losses.append(global_loss)
        #     print('Round {:3d}, global acc {:.3f}'.format(iteration, global_acc))
        #     print('Round {:3d}, global loss {:.3f}'.format(iteration, global_loss))
        # else:
        #     print('Round {:3d}, No participant, skip'.format(iteration))
        #     if len(global_acctotal) > 0:
        #         global_acctotal.append(global_acctotal[-1])
        #         global_losses.append(global_losses[-1])
        #     else:
        #         global_acc, global_loss = test_img(net_glob, dataset_test, args)
        #         global_acctotal.append(global_acc)
        #         global_losses.append(global_loss)

        return w_glob,normal_id,excluded_clients/f
    
def euclidean_distance(local_weights, global_weights):
    distance = 0
    for key in global_weights.keys():
        distance += torch.pow(local_weights[key] - global_weights[key], 2).sum()
    distance = torch.sqrt(distance)
    return distance.item()
def calculate_accuracy(detected_noisy_clients, actual_noisy_clients):
    """
    Calculate the defense metrix of the anomaly detection algorithm.
    """

    detected_set = set(detected_noisy_clients)
    actual_set = set(actual_noisy_clients)

    correct_detections = detected_set.intersection(actual_set)

    # Calculate recall
    if len(actual_set) > 0:
        R = len(correct_detections) / len(actual_set)
    else:
        R = 0.0 

    # Calculate precision
    if len(detected_set) > 0:
        P = len(correct_detections) / len(detected_set)
    else:
        P = 0.0  

    return R,P
def decompose_normal_distributions(data, n_components=2):
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(data.reshape(-1, 1))
    labels = gmm.predict(data.reshape(-1, 1))
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_cluster_index = unique_labels[np.argmax(counts)]
    max_cluster_data = data[labels == max_cluster_index]
    bounds = (max_cluster_data.min(), max_cluster_data.max())
    return max_cluster_data, bounds, gmm.means_, gmm.covariances_, gmm.weights_
def plot_control_chart(client_id, client_means, distances_matrix, save_dir, L=3):
    """
    SPC-based anomaly detection algorithm.
    """
    ano = []
    distances = [distance for client_list in distances_matrix for distance in client_list]
    std = np.std(distances)
    mean = np.mean(distances)
    # the control limit to select clients, in our study LCL is not used
    UCL = mean + L *std
    LCL = mean - L *std
    
    plt.figure(figsize=(10, 6))
    plt.plot(client_id, client_means, marker='o', linestyle='-', color='blue', label='Average Distance')
    for idx, client_mean in zip(client_id, client_means):
        if client_mean > UCL:
            ano.append(idx)
            plt.plot(idx, client_mean, marker='o', color='red')

    plt.axhline(UCL, color='red', linestyle='--', label='UCL')
    plt.title('Control Chart for All Clients')
    plt.xlabel('Client ID')
    plt.ylabel('Average Distance')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'control_chart_all_clients.png'))
    plt.close()

    return ano, UCL, LCL