import numpy as np
import os
from datetime import date
date = date.today().strftime("%Y-%m-%d_m")
import sys

class QIM:
    def __init__(self, delta):
        # delta is the step size of quantization
        self.delta = delta
        # self.alpha = 0.51  # alpha is the weight of the original vector in the embedding

    def embed(self, x, m,alpha=0.51,k=0):
        """
        x is a vector of values to be quantized individually
        m is a binary vector of bits to be embeded
        returns: a quantized vector y
        """
        # make x type float
        x = x.astype(float)
        # state alpha
        scale = self.alpha_func(alpha=alpha)
        d = self.delta
        # get d_0 and d_1 according to m
        # dm = (-1)**(m+1) * d/2.
        dm = m*d/2.
        q_mk = quanti(x-dm-k, d) + dm + k
        # dis = np.round((x-dm-k) / d) - (x-dm-k)/d
        # print('Theory Embedding distortion: $a*delta*(frac((s-d_m-k)/delta))', alpha*d*dis)
        self.q_mk = q_mk
        self.x = x
        y = q_mk * scale + x * (1 - scale)
        return y
    
    def alpha_func(self, alpha):
        return alpha
    def detect(self, z,alpha=1,k=0,scale_delta=1):
        """
        z is the received vector, potentially modified
        returns: a detected vector z_detected and a detected message m_detected
        """
        d = self.delta *scale_delta
        M_cls = 2.
        shape = z.shape
        scale = self.alpha_func(alpha=alpha)
        z = z.flatten()
        z = z.astype(float)
        m_detected = np.zeros_like(z, dtype=float)
        dm_hat = (quanti(z-k,d/M_cls)+k)
        self.dm_hat = dm_hat
        self.m_c = self.selective_round(dm_hat/d)%1
        self.y_dm_hat = abs(z-self.q_mk)
        z_hat = (z-scale * dm_hat)/ (1-scale)
        d_values = [0,d/2.]
        rough_m = np.round((self.selective_round((dm_hat-k)/d)%1)*2)
        m_detected = rough_m
        m_detected = m_detected.reshape(shape)
        return z_hat, m_detected.astype(int)
    
    def selective_round(self,x, threshold=0.99):
        frac = x%1
        if np.allclose(frac,0.5):
            frac = 0.5
        if np.allclose(frac,0):
            frac = 0
        return np.floor(x) + np.where((x % 1) >= threshold, 1, (frac))
    def random_msg(self, l):
        """
        returns: a random binary sequence of length l
        """
        return np.random.choice((0, 1), l)
def quanti(x, delta):
    """
    quantizes the input x with step size delta
    """
    # the delta*floor[x/delta]
    # so floor will increase the distortion
    return np.round(x / delta) * delta










def test_qim_1(delta=1,embedding_alpha=0.99,k=0,plot=False,test=False):
    """
    tests the embed and detect methods of class QIM
    """
    np.random.seed(42)
    l = 10000 # binary message length
    # delta = 1.0 # quantization step (use float for consistency)
    qim = QIM(delta)

    # x = np.random.uniform(-500, 500, l).astype(float) # host sample
    x = np.random.randn(l)
    # x = np.linspace(-5, 5, l).astype(float)  # host sample
    print('Original x (first 5):', x[:5])

    msg = qim.random_msg(l)
    print('Watermark Message (first 10):', msg[:10])

    # --- Crucial: Generate secret_k_sequence ONCE ---
    # This must be the same sequence used for both embedding and detection.
    np.random.seed(123) # Use a different seed for reproducibility of k sequence
    
    
    true_k = k

    # --- Step 1: Embed the watermark ONCE with a specific, fixed alpha ---
    # embedding_alpha = embedding_alpha # This is the "true" alpha used for embedding
    print(f"\n--- Embedding watermark with fixed alpha = {embedding_alpha}, delta = {delta}, k = {true_k} ---")
    print(f"")
    y_watermarked = qim.embed(x, msg, alpha=embedding_alpha, k=true_k)
    print('Watermarked y (first 5):', y_watermarked[:5])
    print('Distortion: ',y_watermarked[:5]-x[:5])
    initial_distortion = np.mean(np.abs(x - y_watermarked))
    good_z, good_msg = qim.detect(y_watermarked, alpha=embedding_alpha, k=true_k, scale_delta=1)
    print(f"Initial Embedding Distortion (Abs Diff): {initial_distortion:.6f}")
    print(f"Detected Message Accuracy: {np.mean(msg == good_msg):.4f}")
    # print(qim.m_c[msg != good_msg])
    # # print((1-qim.m_c)[msg != good_msg])
    # # # print(qim.dm_hat[msg != good_msg]%1)
    # # print()
    # print(msg[msg != good_msg])
    # print(good_msg[msg != good_msg])
    print(f'Recovery error when all correct: {np.mean(np.abs(good_z - x)):.6f}')


    # --- Step 2: Loop through different 'a' values for DETECTION/RESTORATION ---
    # We are testing how well detection/restoration works if we GUESS 'a'
    # The 'y_watermarked' remains the same throughout this loop.
    # alphas_to_test_detection = np.arange(0.5, 1.00, 0.01) # Go up to 0.99 for range
    if test:
        recovery_errors = [] # This will store np.mean(np.abs(z_detected - x))
        message_accuracies = [] # This will store sum(msg==msg_detected)/len(msg)

        # print("\n--- Testing Detection/Restoration with Varying Alphas ---")
        # alphas_to_test_detection =np.linspace(-2, 2, 300)
        # for a_detect in alphas_to_test_detection:
        #     z_detected, msg_detected = qim.detect(y_watermarked, alpha=a_detect, k=true_k)
        # secret_k_sequence = np.linspace(-5, 5, 1000)
        # for k in secret_k_sequence:
        #     z_detected, msg_detected = qim.detect(y_watermarked, alpha=embedding_alpha, k=k)
        # scale_delta = np.concatenate((np.linspace(0.1, 5, 100),np.array([1,2,3,4,5])),axis=0)
        # scale_delta = np.concatenate((np.linspace(0.1, 5, 100),np.linspace(0.9, 1.1, 100)),axis=0)
        # scale_delta = np.linspace(0.1, 5, 100)
        middle = 1
        range_scale = 0.2
        scale_delta = (np.linspace(middle-range_scale, middle+range_scale, 200))
        theory = []
        dm_hat = []
        for sd in scale_delta:
            z_detected, msg_detected = qim.detect(y_watermarked, alpha=embedding_alpha, k=true_k, scale_delta=sd)
            theory.append(np.mean(np.abs(qim.q_mk - qim.dm_hat)*embedding_alpha/(1-embedding_alpha)))
            dm_hat.append(np.mean(np.abs(qim.dm_hat)))
            # print(f"Attempting to detect/restore with detection alpha = {a_detect:.2f}")
            # if np.isclose(a_detect, 1, atol=0.005):
            #     recovery_errors.append(np.inf)
            #     continue  # Skip detection at alpha=1 to avoid division by zero in theory error
            # Use the SAME y_watermarked from the single embedding above
            
            
            # Calculate errors

            current_recovery_error = np.mean(np.abs(z_detected - x))
            # thery = np.mean(np.abs(qim.y_dm_hat*((a_detect-embedding_alpha)/((1-embedding_alpha)*(1-a_detect)))))

            # print(f"  Detected error (first 5): {current_recovery_error}")
            # print(f"  Theory error (first 5): {thery}")
            # print(np.isclose(current_recovery_error, thery, atol=1e-7, rtol=1e-7))
            current_message_accuracy = np.mean(msg == msg_detected) # Already normalized
            # print(f"  Detected message accuracy: {current_message_accuracy:.4f}")
            recovery_errors.append(current_recovery_error)
            message_accuracies.append(current_message_accuracy)
            
            # print(f"  Recovery Error (vs original x): {current_recovery_error:.8f}")
            # print(f"  Message Accuracy: {current_message_accuracy:.4f}")
            
            # Optional: Assert for the perfect case (only when a_detect matches embedding_alpha)
            # if np.isclose(a_detect, embedding_alpha, atol=0.005): # Use a small tolerance for float comparison
            #     # print("  --- Approaching True Alpha ---")
            #     assert np.allclose(x, z_detected, atol=1e-7, rtol=1e-7), \
            #         f"Host not perfectly restored at a_detect={a_detect}!"
                # assert np.all(msg == msg_detected), \
                #     f"Watermark not perfectly detected at a_detect={a_detect}!"

    if plot:
        # # --- Plotting Results ---
        # from matplotlib import pyplot as plt
        # # print(abs((alphas_to_test_detection-embedding_alpha)/(embedding_alpha*(1-alphas_to_test_detection)))[:10])
        # scaled_alpha = qim.alpha_func(embedding_alpha)
        # y = np.abs((alphas_to_test_detection - scaled_alpha) / ((1 - scaled_alpha) * (1 - alphas_to_test_detection)))
        
        # # Plot Recovery Error
        # plt.figure(figsize=(12, 6))
        # plt.plot(alphas_to_test_detection, recovery_errors, marker='o', linestyle='-', markersize=4)
        # plt.plot(alphas_to_test_detection, y, label=r"$\left|\frac{x - 0.7}{0.3(1 - x)}\right|$")
        # # plt.plot(alphas_to_test_detection, abs(((alphas_to_test_detection-embedding_alpha)/((1-embedding_alpha)*(1-alphas_to_test_detection)))), markersize=4,label='Theoretical Error')
        # plt.axvline(x=np.sinc(embedding_alpha), color='r', linestyle='--', label=f'True Embedding Alpha ({embedding_alpha})')
        # plt.title(f'Recovery Error vs. Detection Alpha (Delta={delta}, True Embed Alpha={embedding_alpha})')
        # plt.xlabel('Detection Alpha ($a$)')
        # plt.ylabel('Mean Absolute Recovery Error ($|\\hat{s} - s|$ mean)')
        # plt.ylim(0, 10)  # Optional: limit y for better visualization
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # os.makedirs(f"./outputs/qim/{date}/", exist_ok=True)
        # plt.savefig(f"./outputs/qim/{date}/recovery_error_alpha_{embedding_alpha}_delta_{delta}.png")
        # plt.close()
        # plt.figure(figsize=(12, 6))
        # plt.plot(alphas_to_test_detection, message_accuracies, marker='o', linestyle='-', markersize=4, color='green')
        # plt.axvline(x=embedding_alpha, color='r', linestyle='--', label=f'True Embedding Alpha ({embedding_alpha})')
        # plt.title(f'Message Detection Accuracy vs. Detection Alpha (Delta={delta}, True Embed Alpha={embedding_alpha})')
        # plt.xlabel('Detection Alpha ($a$)')
        # plt.ylabel('Message Accuracy')
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(f"./outputs/qim/{date}/message_accuracy_alpha_{embedding_alpha}.png")
        # plt.close()
    #     #################################################
    #     from matplotlib import pyplot as plt
    #     # print(abs((alphas_to_test_detection-embedding_alpha)/(embedding_alpha*(1-alphas_to_test_detection)))[:10])
    #     # y = np.abs((alphas_to_test_detection - embedding_alpha) / ((1 - embedding_alpha) * (1 - alphas_to_test_detection)))
    #     # Plot Recovery Error
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(secret_k_sequence, recovery_errors, marker='o', linestyle='-', markersize=4)
    #     # plt.plot(secret_k_sequence, y, label=r"$\left|\frac{x - 0.7}{0.3(1 - x)}\right|$")
    #     # plt.plot(alphas_to_test_detection, abs(((alphas_to_test_detection-embedding_alpha)/((1-embedding_alpha)*(1-alphas_to_test_detection)))), markersize=4,label='Theoretical Error')
    #     plt.axvline(x=true_k, color='r', linestyle='--', label=f'True Embedding k ({true_k})')
    #     plt.title(f'Recovery Error vs. Detection secret k (Delta={delta}, True Embed k={true_k})')
    #     plt.xlabel('Detection secret k ($k$)')
    #     plt.ylabel('Mean Absolute Recovery Error ($|\\hat{s} - s|$ mean)')
    #     plt.ylim(0, 10)  # Optional: limit y for better visualization
    #     plt.grid(True)
    #     plt.legend()
    #     plt.tight_layout()
    #     os.makedirs(f"./outputs/qim/{date}/", exist_ok=True)
    #     plt.savefig(f"./outputs/qim/{date}/recovery_error_k_{true_k}_delta_{delta}.png")
    #     plt.close()
    # # Plot Message Accuracy
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(secret_k_sequence, message_accuracies, marker='o', linestyle='-', markersize=4, color='green')
    #     plt.axvline(x=true_k, color='r', linestyle='--', label=f'True Embedding Alpha ({true_k})')
    #     plt.title(f'Message Detection Accuracy vs. Detection secret k (Delta={delta}, True Embed k={true_k})')
    #     plt.xlabel('Detection Alpha ($a$)')
    #     plt.ylabel('Message Accuracy')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(f"./outputs/qim/{date}/message_accuracy_k_{true_k}_delta_{delta}.png")
    #     plt.close()

        # print("\n--- Testing complete. Check generated plots in ./outputs/ ---")
    #     ###########################################
        from matplotlib import pyplot as plt
        # print(abs((alphas_to_test_detection-embedding_alpha)/(embedding_alpha*(1-alphas_to_test_detection)))[:10])
        # y = np.abs((alphas_to_test_detection - embedding_alpha) / ((1 - embedding_alpha) * (1 - alphas_to_test_detection)))
        # Plot Recovery Error
        plt.figure(figsize=(12, 6))
        plt.plot(scale_delta*delta, recovery_errors, marker='o', linestyle='-', markersize=4)
        plt.plot(delta, np.mean(np.abs(good_z-x)), marker="o", color="red")
        plt.plot(scale_delta*delta, theory, label=r"a/(1-a) Q_m,k(s) - d_m")
        # unscale_theory = ((1-embedding_alpha)/embedding_alpha)*np.array(theory)
        # plt.plot(scale_delta*delta, unscale_theory, label=r"$Q_m,k(s) - d_m$")
        plt.plot(scale_delta*delta,dm_hat, label=r"$\hat{d}_m$")
        # plt.plot(secret_k_sequence, y, label=r"$\left|\frac{x - 0.7}{0.3(1 - x)}\right|$")
        # plt.plot(alphas_to_test_detection, abs(((alphas_to_test_detection-embedding_alpha)/((1-embedding_alpha)*(1-alphas_to_test_detection)))), markersize=4,label='Theoretical Error')
        plt.axvline(x=delta, color='r', linestyle='--', label=f'True delta')
        plt.title(f'Recovery Error vs. Detection Delta={delta}')
        plt.xlabel('Detection Delta ($d$)')
        plt.ylabel('Mean Absolute Recovery Error ($|\\hat{s} - s|$ mean)')
        plt.ylim(0, 10)  # Optional: limit y for better visualization
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        os.makedirs(f"./outputs/qim/{date}/", exist_ok=True)
        plt.savefig(f"./outputs/qim/{date}/recovery_error_delta_{delta}.png")
        plt.close()
    # Plot Message Accuracy
        plt.figure(figsize=(12, 6))
        plt.plot(scale_delta*delta, message_accuracies, marker='o', linestyle='-', markersize=4, color='green')
        plt.plot(delta, np.mean(msg == good_msg), marker="o", color="red")
        plt.axvline(x=delta, color='r', linestyle='--', label=f'True delta')
        plt.title(f'Message Detection Accuracy vs. Detection Delta={delta}')
        plt.xlabel('Detection Delta ($d$)')
        plt.ylabel('Message Accuracy')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./outputs/qim/{date}/message_accuracy_delta_{delta}.png")
        plt.close()

        print("\n--- Testing complete. Check generated plots in ./outputs/qim/ ---")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # federated arguments
    parser.add_argument('--a', type=float, default=0.7,
                        help="number of training epochs")
    parser.add_argument('--d', type=float, default=1,
                        help="number of users: n")
    parser.add_argument('--k', type=float, default=0,
                        help="number of users: n")
    args = parser.parse_args()
    sys.exit(test_qim_1(args))