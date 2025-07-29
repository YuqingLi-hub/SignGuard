"""Implementation of QIM method from Data Hiding Codes, Moulin and Koetter, 2005"""

from __future__ import print_function
import sys
import os
from datetime import date
date = date.today().strftime("%Y-%m-%d_")
HOME = os.environ["HOME"]

import numpy as np


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
        d = self.delta
        # get d_0 and d_1 according to m
        # dm = (-1)**(m) * d/2.
        dm = m*d/2.
        # if m ==1:
        #     dm = d/2.
        # print('dm', dm)
        # y = np.floor((x-dm)/d)*d + dm
        # q_mk = np.floor((x-dm-k)/d)*d + dm + k
        q_mk = quanti(x-dm-k, d) + dm + k
        # for i in quanti(x-dm-k, d):
        #     if i % d != 0:
        #         print('wrong quantization', i, d, i % d)
        y = q_mk * alpha + x * (1 - alpha)
        # print('rqim',y)
        # y = np.floor(x/d) * d + (-1)**(m+1) * d/4.
        # print('qim',y)
        return y
    def set_delta(self, x,m,k=0):
        d = self.delta
        dm = (-1)**(m+1) * d/4.
        return quanti(x-dm-k, d) + dm + k
    def detect(self, z,alpha=1,k=0):
        """
        z is the received vector, potentially modified
        returns: a detected vector z_detected and a detected message m_detected
        """

        M_cls = 2.
        shape = z.shape
        z = z.flatten()
        # alpha = self.alpha
        m_detected = np.zeros_like(z, dtype=float)
        # z_detected = np.zeros_like(z, dtype=float)

        # z0 = self.embed(z, 0)
        # z1 = self.embed(z, 1)
        dm_hat = (quanti(z-k,self.delta/M_cls)+k)
        # d0 = self.set_delta(z,0,k)
        # d1 = self.set_delta(z,1,k)
        # print('d0', d0)
        # print('d1', d1)
        # dm_val_pos = self.delta / 4.0   # For m=1, dm = delta/4
        # dm_val_neg = -self.delta / 4.0  # For m=0, dm = -delta/4
        # remainder = (z - k) % self.delta
        # dist_to_expected_pos_dm = np.abs(remainder - dm_val_pos)
        # dist_to_expected_neg_dm_mod = np.abs(remainder - (self.delta + dm_val_neg)) # (1 + (-0.25)) = 0.75
        # dm_hat = np.where(dist_to_expected_pos_dm < dist_to_expected_neg_dm_mod,
        #                   dm_val_pos,
        #                   dm_val_neg)
        # dm_hat = (quanti(z-k,self.delta/M_cls)+k)
        # gen = zip(range(len(z)), d0, d1)
        
        # for i, dd0, dd1 in gen:
        #     if dm_hat[i] == dd0:
        #         m_detected[i] = 0
        #     elif dm_hat[i] == dd1:
        #         m_detected[i] = 1
            # else:
                # print('Error in detection, dm_hat:', dm_hat[i], 'd0:', dd0, 'd1:', dd1)
                # m_detected[i] = -1
        #     if dd0 < dd1:
        #         m_detected[i] = 0
        #     else:
        #         m_detected[i] = 1
        # M_cls = 2.
        # # TODO: findout what should be the best modulo or representation of {delta}, modulo
        # # The points on the sets is d0/d1 + delta*integer
        # # print(np.allclose(quanti(z-k,self.delta/M_cls)+k,z))
        
        # dm_hat = (quanti(z-k,self.delta/M_cls)+k)
        # for i,d in zip(range(len(dm_hat)),dm_hat):
        #     # d -self.delta/4.
        #     # print((d-self.delta/4.)%self.delta)
        #     print(d%self.delta)

            # print()
            # if (d-self.delta/4.) in z0:
            #     # m_detected[i] = 0
            # else:
            #     m_detected[i] = 1
        
        # print('m1', m1)
        # print('m0', m0)
        # print(m1+ (m0))
        # print(dm_hat%self.delta)
        # d1 = + self.delta / 4., # d0 = - self.delta / 4.
        # print(dm_hat%self.delta)
        m_detected = np.array([1 if i>0 else 0 for i in dm_hat%self.delta])
        # print('m', m)
        # val_before_mod = (self.delta / M_cls) * np.floor(((z - k) * M_cls) / self.delta) + k

        # raw_dm_hat_remainder = val_before_mod % self.delta
        # dm_hat = raw_dm_hat_remainder
        # # dm_hat = ((np.floor(val_before_mod/self.delta)*self.delta)-val_before_mod)
        # dm_hat = np.where(np.isclose(raw_dm_hat_remainder, self.delta / 4),
        #               self.delta / 4,
        #               -self.delta / 4)
        # print(1.25%1,0.75%1)
        # print('dm_hat', dm_hat)
        # print('z', z)
        # print(alpha*dm_hat)
        # print((z-alpha * dm_hat))
        if alpha == 1:
            print('alpha is 1, no restoration')
            z_hat = z - alpha * dm_hat
        else:
            # print(f'alpha={alpha}, restoring original signal')
            z_hat = (z-alpha * dm_hat)/ (1-alpha)
        # print('z_hat', z_hat)
        # z_detected = z_detected.reshape(shape)
        m_detected = m_detected.reshape(shape)
        return z_hat, m_detected.astype(int)

    def random_msg(self, l):
        """
        returns: a random binary sequence of length l
        """
        return np.random.choice((0, 1), l)
    # def extract_and_restore_rqim_paper_method(self,watermarked_y, secret_k_sequence=0):
    #     """
    #     Extracts watermark bits and restores original weights from a sequence of watermarked weights.

    #     Args:
    #         watermarked_y (np.array): Array of watermarked weights.
    #         delta (float): Quantization step size used during embedding.
    #         alpha (float): Scaling factor used during embedding.
    #         secret_k_sequence (np.array): The sequence of secret keys (k values) used for each weight.

    #     Returns:
    #         tuple: (extracted_watermark_bits, restored_original_weights)
    #             extracted_watermark_bits (np.array): The extracted binary watermark bits.
    #             restored_original_weights (np.array): The perfectly restored original weights.
    #     """
    #     restored_original_weights = np.zeros_like(watermarked_y)
    #     extracted_watermark_bits = []
    #     alpha = self.alpha
    #     delta = self.delta
    #     # Calculate the receiver's quantizer step size (from Eq. 11)
    #     # Note: If alpha is 1, this becomes division by zero. Paper implies alpha < 1.
    #     receiver_delta_prime = delta / 2

    #     # Define the dithering offsets for extraction (e.g., 0 and delta/2)
    #     # This must match what was used during embedding!
    #     d_extraction_values = {0: 0, 1: delta / 2} # Assume this mapping was used

    #     for i, y_i in enumerate(watermarked_y):
    #         k_i = secret_k_sequence # Get the specific secret key used for this weight

    #         # Step 1: Extract the watermark bit (based on Eq. 11 logic)
    #         # This part is implicit in the paper's Eq 11, which shows d'_m.
    #         # We need to test which 'm' makes y_i align with its lattice.
    #         # We check distances to the two possible shifted lattices for 0 and 1
            
    #         # Candidate 1: Quantize as if bit 0 was embedded
    #         arg_q_0 = y_i - k_i - d_extraction_values[0] # Use d_0 for testing
    #         quantized_val_0 = quanti(arg_q_0, receiver_delta_prime)
    #         lattice_val_0 = quantized_val_0 + k_i + d_extraction_values[0] # Reconstruct receiver's lattice point

    #         # Candidate 2: Quantize as if bit 1 was embedded
    #         arg_q_1 = y_i - k_i - d_extraction_values[1] # Use d_1 for testing
    #         quantized_val_1 = quanti(arg_q_1, receiver_delta_prime)
    #         lattice_val_1 = quantized_val_1 + k_i + d_extraction_values[1] # Reconstruct receiver's lattice point

    #         # Determine which bit was likely embedded based on closeness
    #         # print(abs(y_i - lattice_val_0), abs(y_i - lattice_val_1))
    #         if abs(y_i - lattice_val_0) < abs(y_i - lattice_val_1):
    #             extracted_bit = 0
    #         else:
    #             extracted_bit = 1
    #         extracted_watermark_bits.append(extracted_bit)

    #         # Step 2: Restore the original signal (based on Eq. 13/14, assuming noiseless)
    #         # We need the Q_mk_s from embedding (which is lattice_val_extracted_bit)
    #         # From Eq. 9: s_R_QIM = alpha * Q_mk_s + (1 - alpha) * s
    #         # Rearranging for s (where s_R_QIM is y_i, and s is restored_original_weights[i]):
    #         # y_i - alpha * Q_mk_s = (1 - alpha) * s
    #         # s = (y_i - alpha * Q_mk_s) / (1 - alpha)

    #         Q_mk_s_extracted = quanti(y_i - d_extraction_values[extracted_bit] - k_i, receiver_delta_prime) + d_extraction_values[extracted_bit] + k_i
    #         # Using the receiver's quantizer based on the extracted bit, this reconstructs Q_mk_s_extracted

    #         # Using Eq. 13 without noise (n=0)
    #         # The term Q_Delta/(1-alpha) + k(y) in Eq 13 is the Q_mk_s_extracted
    #         restored_s_i = (y_i - alpha * Q_mk_s_extracted) / (1 - alpha)
    #         restored_original_weights[i] = restored_s_i

    #     return np.array(extracted_watermark_bits), restored_original_weights
def quanti(x, delta):
    """
    quantizes the input x with step size delta
    """
    # the delta*floor[x/delta]
    
    return np.round(x / delta) * delta
# def floor_to_delta(x, d):
    
#     # d0 = (-1)**(0+1) * d/4.
#     # d1 = (-1)**(1+1) * d/4.
#     x = x % d
# def test_qim(delta=1,alpha=0.51):
#     """
#     tests the embed and detect methods of class QIM
#     """
#     np.random.seed(42)
#     l = 10000 # binary message length
#     delta = 1 # quantization step
#     qim = QIM(delta)
#     # while True:
#     # x = np.random.randint(0, 255, l).astype(float) # host sample
#     x = np.random.uniform(-100, 255, l).astype(float) # host sample
#     print('x', x)
#     msg = qim.random_msg(l)
#     y = qim.embed(x, msg,alpha=alpha)
#     print('y', y)
#     z_detected, msg_detected = qim.detect(y,alpha=alpha)
#     print(np.mean(np.abs(z_detected - x)))

#     print('message accuracy', sum(msg==msg_detected)/len(msg))
#     # print()
#     # print(msg[:10], msg_detected[:10])
#     # print(np.unique(msg_detected))
#     # assert np.allclose(msg, msg_detected) # compare the original and detected messages
#     # assert np.allclose(x, z_detected)     # compare the original and detected vectors
def test_qim(delta=1,alpha=0.51,k=0):
    """
    tests the embed and detect methods of class QIM
    """
    np.random.seed(42)
    l = 10000 # binary message length
    # delta = 1 # quantization step
    qim = QIM(delta)
    # while True:
    # x = np.random.randint(0, 255, l).astype(float) # host sample
    x = np.random.uniform(-100, 255, l).astype(float) # host sample
    print('x', x)
    msg = qim.random_msg(l)

    # alpha = 0.71
    y = qim.embed(x, msg,alpha=alpha)
    print('y', y)
    z_detected, msg_detected = qim.detect(y,alpha=alpha)
    print(z_detected)
    print(np.mean(msg == msg_detected))
    # print(sum(np.allclose(msg_detected,msg)))
    alphas = np.arange(0.51, 1, 0.01)
    error = []
    msg_error = []
    # for a in alphas:
    #     z_detected, msg_detected = qim.detect(y,alpha=a)
    #     error.append(np.mean(np.abs(z_detected - x)))
    #     msg_error.append(sum(msg==msg_detected)/len(msg))
    # print(msg_error[:10],msg_error[-10:])
    # return alphas, error, msg_error
    # from matplotlib import pyplot as plt
    # plt.figure(figsize=(12, 6))
    # plt.plot(alphas, error, marker='o', linestyle='-', markersize=4)
    # plt.axvline(x=alpha, color='r', linestyle='--', label=f'True Embedding Alpha ({alpha})')
    # plt.title(f'Recovery Error vs. Detection Alpha (Delta={delta}, True Embed Alpha={alpha})')
    # plt.xlabel('Detection Alpha ($a$)')
    # plt.ylabel('Mean Absolute Recovery Error ($|\\hat{s} - s|$ mean)')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"./outputs/recovery_error_alpha_{alpha}.png")
    # plt.close()

    # # Plot Message Accuracy
    # plt.figure(figsize=(12, 6))
    # plt.plot(alphas, msg_error, marker='o', linestyle='-', markersize=4, color='green')
    # plt.axvline(x=alpha, color='r', linestyle='--', label=f'True Embedding Alpha ({alpha})')
    # plt.title(f'Message Detection Accuracy vs. Detection Alpha (Delta={delta}, True Embed Alpha={alpha})')
    # plt.xlabel('Detection Alpha ($a$)')
    # plt.ylabel('Message Accuracy')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"./outputs/message_accuracy_alpha_{alpha}.png")
    # plt.close()
    # plt.figure(figsize=(10, 5))
    # plt.plot(alphas, error)
    # plt.title(f'Alpha vs Recovery Error (delta={delta},true alpha={alpha})')
    # plt.xlabel('Alpha')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.savefig(f"./outputs/alpha={alpha}.png")
    # plt.close()




def test_qim_1(delta=1,embedding_alpha=0.99):
    """
    tests the embed and detect methods of class QIM
    """
    np.random.seed(42)
    l = 10000 # binary message length
    # delta = 1.0 # quantization step (use float for consistency)
    qim = QIM(delta)

    x = np.random.uniform(0, 255, l).astype(float) # host sample
    print('Original x (first 5):', x[:5])

    msg = qim.random_msg(l)
    print('Watermark Message (first 10):', msg[:10])

    # --- Crucial: Generate secret_k_sequence ONCE ---
    # This must be the same sequence used for both embedding and detection.
    np.random.seed(123) # Use a different seed for reproducibility of k sequence
    secret_k_sequence = 0


    # --- Step 1: Embed the watermark ONCE with a specific, fixed alpha ---
    # embedding_alpha = embedding_alpha # This is the "true" alpha used for embedding
    print(f"\n--- Embedding watermark with fixed alpha = {embedding_alpha} ---")
    y_watermarked = qim.embed(x, msg, alpha=embedding_alpha, k=secret_k_sequence)
    print('Watermarked y (first 5):', y_watermarked[:5])
    
    initial_distortion = np.mean(np.abs(x - y_watermarked))
    print(f"Initial Embedding Distortion (Abs Diff): {initial_distortion:.6f}")


    # --- Step 2: Loop through different 'a' values for DETECTION/RESTORATION ---
    # We are testing how well detection/restoration works if we GUESS 'a'
    # The 'y_watermarked' remains the same throughout this loop.
    alphas_to_test_detection = np.arange(0.5, 1.00, 0.01) # Go up to 0.99 for range
    
    recovery_errors = [] # This will store np.mean(np.abs(z_detected - x))
    message_accuracies = [] # This will store sum(msg==msg_detected)/len(msg)

    print("\n--- Testing Detection/Restoration with Varying Alphas ---")
    for a_detect in alphas_to_test_detection:
        print(f"Attempting to detect/restore with detection alpha = {a_detect:.2f}")
        
        # Use the SAME y_watermarked from the single embedding above
        z_detected, msg_detected = qim.detect(y_watermarked, alpha=a_detect, k=secret_k_sequence)
        
        # Calculate errors
        current_recovery_error = np.mean(np.abs(z_detected - x))
        current_message_accuracy = np.mean(msg == msg_detected) # Already normalized
        
        recovery_errors.append(current_recovery_error)
        message_accuracies.append(current_message_accuracy)
        
        print(f"  Recovery Error (vs original x): {current_recovery_error:.8f}")
        print(f"  Message Accuracy: {current_message_accuracy:.4f}")
        
        # Optional: Assert for the perfect case (only when a_detect matches embedding_alpha)
        if np.isclose(a_detect, embedding_alpha, atol=0.005): # Use a small tolerance for float comparison
            print("  --- Approaching True Alpha ---")
            assert np.allclose(x, z_detected, atol=1e-7, rtol=1e-7), \
                f"Host not perfectly restored at a_detect={a_detect}!"
            # assert np.all(msg == msg_detected), \
            #     f"Watermark not perfectly detected at a_detect={a_detect}!"


    # --- Plotting Results ---
    from matplotlib import pyplot as plt

    # Plot Recovery Error
    plt.figure(figsize=(12, 6))
    plt.plot(alphas_to_test_detection, recovery_errors, marker='o', linestyle='-', markersize=4)
    plt.axvline(x=embedding_alpha, color='r', linestyle='--', label=f'True Embedding Alpha ({embedding_alpha})')
    plt.title(f'Recovery Error vs. Detection Alpha (Delta={delta}, True Embed Alpha={embedding_alpha})')
    plt.xlabel('Detection Alpha ($a$)')
    plt.ylabel('Mean Absolute Recovery Error ($|\\hat{s} - s|$ mean)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs(f"./outputs/qim/{date}/", exist_ok=True)
    plt.savefig(f"./outputs/qim/{date}/recovery_error_alpha_{embedding_alpha}_delta_{delta}.png")
    plt.close()

    # Plot Message Accuracy
    # plt.figure(figsize=(12, 6))
    # plt.plot(alphas_to_test_detection, message_accuracies, marker='o', linestyle='-', markersize=4, color='green')
    # plt.axvline(x=embedding_alpha, color='r', linestyle='--', label=f'True Embedding Alpha ({embedding_alpha})')
    # plt.title(f'Message Detection Accuracy vs. Detection Alpha (Delta={delta}, True Embed Alpha={embedding_alpha})')
    # plt.xlabel('Detection Alpha ($a$)')
    # plt.ylabel('Message Accuracy')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"./outputs/message_accuracy_alpha_{embedding_alpha}.png")
    # plt.close()

    print("\n--- Testing complete. Check generated plots in ./outputs/ ---")


def main():
    # delta_values = [0.5, 1.0, 1.5, 2.0]
    # for d in delta_values:
    #     for a in np.arange(0.50, 1.01, 0.05):
    #         # print(f"Testing QIM with alpha={a}")
    #         # alphas, error, msg_error = test_qim_1(delta=d, embedding_alpha=a)
    #         test_qim_1(delta=d, embedding_alpha=a)
            # print(f"Alphas: {alphas[:5]}...")
            # Plot Message Accuracy
            # from matplotlib import pyplot as plt
            # plt.figure(figsize=(10, 5))
            # plt.plot(alphas, error, label=f'Alpha {a}, Delta {d}', marker='o')
            # plt.title(f'Alpha, Delta {d}')
            # plt.xlabel('Alpha')
            # plt.ylabel('Loss')
            # plt.legend()
            # plt.savefig(f"./outputs/delta={d}.png")
            # plt.close()
            # plt.figure(figsize=(12, 6))
            # plt.plot(alphas, msg_error, marker='o', linestyle='-', markersize=4, color='green')
            # plt.axvline(x=a, color='r', linestyle='--', label=f'True Embedding Alpha ({a})')
            # plt.title(f'Message Detection Accuracy vs. Detection Alpha (Delta={d}, True Embed Alpha={a})')
            # plt.xlabel('Detection Alpha ($a$)')
            # plt.ylabel('Message Accuracy')
            # plt.grid(True)
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig(f"./outputs/message_accuracy_alpha_{a}_delta_{d}.png")
            # plt.close()
            # plt.figure(figsize=(10, 5))
            # plt.plot(alphas, error)
            # plt.title(f'Alpha vs Recovery Error (delta={a},true alpha={a})')
            # plt.xlabel('Alpha')
            # plt.ylabel('Loss')
            # plt.legend()
            # plt.savefig(f"./outputs/alpha={a}_delta={d}.png")
            # plt.close()
    test_qim()


if __name__ == "__main__":
    sys.exit(main())