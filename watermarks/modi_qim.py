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

    def embed(self, x, m,alpha=0.51,k=0,beta=1.):
        """
        x is a vector of values to be quantized individually
        m is a binary vector of bits to be embeded
        returns: a quantized vector y
        """
        # make x type float
        x = x.astype(float)
        # state alpha
        scale = self.alpha_func(alpha=alpha,beta=beta)
        print(scale)
        d = self.delta
        # get d_0 and d_1 according to m
        # dm = (-1)**(m+1) * d/2.
        dm = m*d/2.
        q_mk = quanti(beta*(x-dm-k), d) + (dm + k)*beta
        # dis = np.round((x-dm-k) / d) - (x-dm-k)/d
        # print('Theory Embedding distortion: $a*delta*(frac((s-d_m-k)/delta))', alpha*d*dis)
        self.q_mk = q_mk
        self.x = x
        y = q_mk * scale + x * (1 - scale)
        return y
    
    def alpha_func(self, alpha,beta=1):
        return alpha/beta
    def detect(self, z,alpha=1,k=0,scale_delta=1,beta=1.):
        """
        z is the received vector, potentially modified
        returns: a detected vector z_detected and a detected message m_detected
        """
        d = self.delta *scale_delta
        M_cls = 2.
        shape = z.shape
        scale = self.alpha_func(alpha=alpha,beta=beta)
        z = z.flatten()
        z = z.astype(float)
        m_detected = np.zeros_like(z, dtype=float)
        dm_hat = (quanti(beta*(z-k),d/M_cls)+k*beta)
        self.dm_hat = dm_hat
        self.m_c = self.selective_round(dm_hat/d)%1
        self.y_dm_hat = abs(z-self.q_mk)
        z_hat = (z-scale * dm_hat)/ (1-scale)
        d_values = [0,d/2.]
        rough_m = np.round((self.selective_round((dm_hat-k*beta)/d)%1)*2)
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


    def make_alpha_pattern(self, L, alpha_c=0.7, delta_alpha=0.01, N=None,
                            use_cos=False, beta=0.2, f=1.0, clip_eps=1e-6):
        """
        Build per-index a_i pattern of length L.

        Options:
            - uniform grid around alpha_c (set N and delta_alpha; use_cos=False)
            - cosine carrier around alpha_c (use_cos=True, with amplitude beta)
        """
        if use_cos:
            # Cosine carrier over indices
            i = np.arange(L)
            alpha_i = alpha_c + beta * np.cos(2.0 * np.pi * f * (i / max(1, L)))
        else:
            # Uniform grid, repeated to length L
            if N is None:
                raise ValueError("For uniform a grid, set N (number of points).")
            grid = alpha_c + (np.arange(N) - (N - 1) / 2.0) * delta_alpha
            alpha_i = np.resize(grid, L)

        # Keep α_i strictly within (0,1) to avoid division issues
        alpha_i = np.clip(alpha_i, clip_eps, 1.0 - clip_eps)
        return alpha_i

    def embed_alpha_pattern(self, x, m, alpha_pattern, k=0):
        """
        Embed with per-index a_i pattern (Design A).
        """
        x = x.astype(float)
        d = self.delta
        dm = (m * d / 2.0)
        q_mk = quanti(x - dm - k, d) + dm + k
        self.q_mk = q_mk
        self.x = x
        self.alpha_pattern = alpha_pattern.astype(float)
        y = self.alpha_pattern * q_mk + (1.0 - self.alpha_pattern) * x
        return y

    def detect_alpha_score(self, z, alpha0, k=0, f=1.0, carrier='exp', weights=None):
        """
        Sweep trial a0: compute matched score with known a_i pattern.
        Returns a scalar score that shows a Dirichlet/sinc-like mainlobe vs a0.
        """
        if not hasattr(self, 'alpha_pattern'):
            raise RuntimeError("alpha_pattern not found. Call embed_alpha_pattern first.")

        d = self.delta
        z = z.astype(float).flatten()

        # Estimate anchor with α0
        qhat = quanti(z - k, d / 2.0) + k

        # Residual (unblend) with trial α0
        u = (z - alpha0 * qhat) / (1.0 - alpha0)
        v = u - self.x  # needs self.x from embed-time (as in your code)
        # return u
        # Carrier over α-pattern
        a = self.alpha_pattern
        if weights is None:
            weights = np.ones_like(a)

        if carrier == 'exp':
            ph = np.exp(-1j * 2.0 * np.pi * f * a)
            score = np.abs(np.sum(weights * v * ph))
        elif carrier == 'cos':
            ph = np.cos(2.0 * np.pi * f * a)
            score = np.abs(np.sum(weights * v * ph))
        else:
            raise ValueError("carrier must be 'exp' or 'cos'")
        return score
    def sweep_alpha0(self, z, alpha0_grid, **kwargs):
        return np.array([self.detect_alpha_score(z, a0, **kwargs) for a0 in alpha0_grid])
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
    true_f = 10
    # alpha_pattern = qim.make_alpha_pattern(len(x), alpha_c=0.7, delta_alpha=0.005, N=33, use_cos=True,f=true_f)
    alpha_pattern = qim.make_alpha_pattern(len(x), alpha_c=0.7, delta_alpha=0.005, N=33, use_cos=False)
    print('Alpha Pattern unique:', np.unique(alpha_pattern), len(np.unique(alpha_pattern)))
    y_watermarked = qim.embed_alpha_pattern(x, msg, alpha_pattern, k=0.0)
    good_z, good_msg = qim.detect(y_watermarked, alpha=alpha_pattern, k=0, scale_delta=1)
    # bad_alpha = qim.make_alpha_pattern(len(x), alpha_c=0.7, delta_alpha=0.005, N=33, use_cos=True,f=1000)
    # bad_z, bad_msg = qim.detect(y_watermarked, alpha=bad_alpha, k=0, scale_delta=1)
    # print('Alpha Pattern unique:', np.unique(bad_alpha), len(np.unique(bad_alpha)))
    # print(f"Detected Message Accuracy: {np.mean(msg == bad_msg):.4f}")
    # print(f'Recovery error when wrong: {np.mean(np.abs(bad_z - x)):.6f}')

    # Sweep detector α0
    alpha0s = np.linspace(0.5, 0.9, 81)

    # scores = qim.sweep_alpha0(y_watermarked, alpha0s, k=0.0, carrier='exp', f=1.0)
    # print(np.argmin(scores), np.min(scores),alpha0s[np.argmin(scores)])
    # print(embedding_alpha)
    # print("Scores for α0 sweep:", scores[:10])  # Print first 10 scores for brevity
    # from matplotlib import pyplot as plt
    # plt.figure(figsize=(12, 6))
    # plt.plot(alpha0s, scores, marker='o', linestyle='-', markersize=4)
    # # plt.plot(alphas_to_test_detection, y, label=r"$\left|\frac{x - 0.7}{0.3(1 - x)}\right|$")
    # # plt.plot(alphas_to_test_detection, abs(((alphas_to_test_detection-embedding_alpha)/((1-embedding_alpha)*(1-alphas_to_test_detection)))), markersize=4,label='Theoretical Error')
    # plt.axvline(x=embedding_alpha, color='r', linestyle='--', label=f'True Embedding Alpha ({embedding_alpha})')
    # plt.title(f'Sweep Score vs. Guessed Alpha (Delta={delta}, True Embed Alpha={embedding_alpha})')
    # plt.xlabel('Detection Alpha ($a$)')
    # plt.ylabel('Score ($|\\hat{s} - s|$ mean)')
    # plt.ylim(0, 10)  # Optional: limit y for better visualization
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # os.makedirs(f"./outputs/qim/{date}/", exist_ok=True)
    # plt.savefig(f"./outputs/qim/{date}/score_alpha_{embedding_alpha}_delta_{delta}.png")
    # plt.close()




    print('Watermarked y (first 5):', y_watermarked[:5])
    print('Distortion: ',y_watermarked[:5]-x[:5])
    initial_distortion = np.mean(np.abs(x - y_watermarked))
    print(f"Initial Embedding Distortion (Abs Diff): {initial_distortion:.6f}")
    print(f"Detected Message Accuracy: {np.mean(msg == good_msg):.4f}")
    print(f'Recovery error when all correct: {np.mean(np.abs(good_z - x)):.6f}')


    #######################################
    # # --- Step 1: Embed the watermark ONCE with a specific, fixed alpha ---
    # # embedding_alpha = embedding_alpha # This is the "true" alpha used for embedding
    # print(f"\n--- Embedding watermark with fixed alpha = {embedding_alpha}, delta = {delta}, k = {true_k} ---")
    # print(f"")
    # y_watermarked = qim.embed(x, msg, alpha=embedding_alpha, k=true_k)
    # print('Watermarked y (first 5):', y_watermarked[:5])
    # print('Distortion: ',y_watermarked[:5]-x[:5])
    # initial_distortion = np.mean(np.abs(x - y_watermarked))
    # good_z, good_msg = qim.detect(y_watermarked, alpha=embedding_alpha, k=true_k, scale_delta=1)
    # print(f"Initial Embedding Distortion (Abs Diff): {initial_distortion:.6f}")
    # print(f"Detected Message Accuracy: {np.mean(msg == good_msg):.4f}")
    # # print(qim.m_c[msg != good_msg])
    # # # print((1-qim.m_c)[msg != good_msg])
    # # # # print(qim.dm_hat[msg != good_msg]%1)
    # # # print()
    # # print(msg[msg != good_msg])
    # # print(good_msg[msg != good_msg])
    # print(f'Recovery error when all correct: {np.mean(np.abs(good_z - x)):.6f}')


    # --- Step 2: Loop through different 'a' values for DETECTION/RESTORATION ---
    # We are testing how well detection/restoration works if we GUESS 'a'
    # The 'y_watermarked' remains the same throughout this loop.
    # alphas_to_test_detection = np.arange(0.5, 1.00, 0.01) # Go up to 0.99 for range
    if test:
        recovery_errors = [] # This will store np.mean(np.abs(z_detected - x))
        message_accuracies = [] # This will store sum(msg==msg_detected)/len(msg)

        # print("\n--- Testing Detection/Restoration with Varying Alphas ---")
        # alphas_to_test_detection =np.linspace(true_f-1, true_f+1, 200)
        alphas_to_test_detection =  np.linspace(-2, 2,200)
        for a_detect in alphas_to_test_detection:
            # alpha_pattern = qim.make_alpha_pattern(len(x), alpha_c=embedding_alpha, delta_alpha=0.005, N=33, use_cos=True,f=a_detect)
            # print(np.unique(alpha_pattern))
            alpha_pattern = a_detect
            z_detected, msg_detected = qim.detect(y_watermarked, alpha=alpha_pattern, k=true_k)
        # secret_k_sequence = np.linspace(-5, 5, 1000)
        # for k in secret_k_sequence:
        #     z_detected, msg_detected = qim.detect(y_watermarked, alpha=embedding_alpha, k=k)
        # scale_delta = np.concatenate((np.linspace(0.1, 5, 100),np.array([1,2,3,4,5])),axis=0)
        # scale_delta = np.concatenate((np.linspace(0.1, 5, 100),np.linspace(0.9, 1.1, 100)),axis=0)
        # scale_delta = np.linspace(0.1, 5, 100)
        # middle = 1
        # range_scale = 0.2
        # scale_delta = (np.linspace(middle-range_scale, middle+range_scale, 200))
        # theory = []
        # dm_hat = []
        # for sd in scale_delta:
        #     z_detected, msg_detected = qim.detect(y_watermarked, alpha=embedding_alpha, k=true_k, scale_delta=sd)
        #     theory.append(np.mean(np.abs(qim.q_mk - qim.dm_hat)*embedding_alpha/(1-embedding_alpha)))
        #     dm_hat.append(np.mean(np.abs(qim.dm_hat)))
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
        print(alphas_to_test_detection[np.argmin(recovery_errors)],np.min(recovery_errors),embedding_alpha)
    if plot:
        # --- Plotting Results ---
        from matplotlib import pyplot as plt
        # print(abs((alphas_to_test_detection-embedding_alpha)/(embedding_alpha*(1-alphas_to_test_detection)))[:10])
        scaled_alpha = qim.alpha_func(embedding_alpha)
        # y = np.abs((alphas_to_test_detection - scaled_alpha) / ((1 - scaled_alpha) * (1 - alphas_to_test_detection)))
        
        # Plot Recovery Error
        plt.figure(figsize=(12, 6))
        plt.plot(alphas_to_test_detection, recovery_errors, marker='o', linestyle='-', markersize=4)
        # plt.plot(alphas_to_test_detection, y, label=r"$\left|\frac{x - 0.7}{0.3(1 - x)}\right|$")
        # plt.plot(alphas_to_test_detection, abs(((alphas_to_test_detection-embedding_alpha)/((1-embedding_alpha)*(1-alphas_to_test_detection)))), markersize=4,label='Theoretical Error')
        plt.axvline(x=np.sinc(embedding_alpha), color='r', linestyle='--', label=f'True Embedding Alpha ({embedding_alpha})')
        plt.title(f'Recovery Error vs. Detection Alpha (Delta={delta}, True Embed Alpha={embedding_alpha})')
        plt.xlabel('Detection Alpha ($a$)')
        plt.ylabel('Mean Absolute Recovery Error ($|\\hat{s} - s|$ mean)')
        plt.ylim(0, 10)  # Optional: limit y for better visualization
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        os.makedirs(f"./outputs/qim/{date}/", exist_ok=True)
        plt.savefig(f"./outputs/qim/{date}/recovery_error_alpha_{embedding_alpha}_delta_{delta}.png")
        plt.close()
        plt.figure(figsize=(12, 6))
        plt.plot(alphas_to_test_detection, message_accuracies, marker='o', linestyle='-', markersize=4, color='green')
        plt.axvline(x=embedding_alpha, color='r', linestyle='--', label=f'True Embedding Alpha ({embedding_alpha})')
        plt.title(f'Message Detection Accuracy vs. Detection Alpha (Delta={delta}, True Embed Alpha={embedding_alpha})')
        plt.xlabel('Detection Alpha ($a$)')
        plt.ylabel('Message Accuracy')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./outputs/qim/{date}/message_accuracy_alpha_{embedding_alpha}.png")
        plt.close()
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
    #     from matplotlib import pyplot as plt
    #     # print(abs((alphas_to_test_detection-embedding_alpha)/(embedding_alpha*(1-alphas_to_test_detection)))[:10])
    #     # y = np.abs((alphas_to_test_detection - embedding_alpha) / ((1 - embedding_alpha) * (1 - alphas_to_test_detection)))
    #     # Plot Recovery Error
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(scale_delta*delta, recovery_errors, marker='o', linestyle='-', markersize=4)
    #     plt.plot(delta, np.mean(np.abs(good_z-x)), marker="o", color="red")
    #     plt.plot(scale_delta*delta, theory, label=r"a/(1-a) Q_m,k(s) - d_m")
    #     # unscale_theory = ((1-embedding_alpha)/embedding_alpha)*np.array(theory)
    #     # plt.plot(scale_delta*delta, unscale_theory, label=r"$Q_m,k(s) - d_m$")
    #     plt.plot(scale_delta*delta,dm_hat, label=r"$\hat{d}_m$")
    #     # plt.plot(secret_k_sequence, y, label=r"$\left|\frac{x - 0.7}{0.3(1 - x)}\right|$")
    #     # plt.plot(alphas_to_test_detection, abs(((alphas_to_test_detection-embedding_alpha)/((1-embedding_alpha)*(1-alphas_to_test_detection)))), markersize=4,label='Theoretical Error')
    #     plt.axvline(x=delta, color='r', linestyle='--', label=f'True delta')
    #     plt.title(f'Recovery Error vs. Detection Delta={delta}')
    #     plt.xlabel('Detection Delta ($d$)')
    #     plt.ylabel('Mean Absolute Recovery Error ($|\\hat{s} - s|$ mean)')
    #     plt.ylim(0, 10)  # Optional: limit y for better visualization
    #     plt.grid(True)
    #     plt.legend()
    #     plt.tight_layout()
    #     os.makedirs(f"./outputs/qim/{date}/", exist_ok=True)
    #     plt.savefig(f"./outputs/qim/{date}/recovery_error_delta_{delta}.png")
    #     plt.close()
    # # Plot Message Accuracy
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(scale_delta*delta, message_accuracies, marker='o', linestyle='-', markersize=4, color='green')
    #     plt.plot(delta, np.mean(msg == good_msg), marker="o", color="red")
    #     plt.axvline(x=delta, color='r', linestyle='--', label=f'True delta')
    #     plt.title(f'Message Detection Accuracy vs. Detection Delta={delta}')
    #     plt.xlabel('Detection Delta ($d$)')
    #     plt.ylabel('Message Accuracy')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(f"./outputs/qim/{date}/message_accuracy_delta_{delta}.png")
    #     plt.close()

    #     print("\n--- Testing complete. Check generated plots in ./outputs/qim/ ---")

if __name__ == "__main__":
    # import numpy as np
    # import matplotlib.pyplot as plt

    # rng = np.random.default_rng(123)

    # # ---------- Basic utilities ----------
    # def Q_delta(u, Delta):
    #     # Nearest multiple of Delta
    #     return np.round(u / Delta) * Delta

    # def frac(u, Delta):
    #     # fractional part wrt Delta in [0, Delta)
    #     return (u % Delta)

    # # ---------- Embedding (beta-scaled R-QIM, binary m) ----------
    # def embed_beta_rqim(x, m, Delta=1.0, alpha=0.6, beta=4.0, k=0.0):
    #     """
    #     x: host vector
    #     m: {0,1} watermark bits (same shape as x)
    #     Delta: step
    #     alpha: mix weight (0<alpha<1)
    #     beta: scale used inside quantization domain
    #     k: shift (scalar or per-coordinate vector)
    #     """
    #     dm = (Delta/2.0) * m
    #     u  = beta * (x - (dm + k))
    #     q  = Q_delta(u, Delta)
    #     qp = (q / beta) + (dm + k)
    #     y  = (1.0 - alpha) * x + alpha * qp
    #     return y, qp  # return qp for diagnostics if you want

    # # ---------- One-step demix & bit decision for a candidate (alpha0, beta0, k0) ----------
    # def demix_and_bits(y, Delta=1.0, alpha0=0.6, beta0=4.0, k0=0.0):
    #     """
    #     Build candidate lattice from y, then demix to estimate x, and decide bits.
    #     """
    #     uy   = beta0 * (y - k0)
    #     qhat = Q_delta(uy, Delta)
    #     qph  = (qhat / beta0) + k0

    #     # demix (noiseless R-QIM inverse if alpha0 matches the embed alpha)
    #     xhat = (y - alpha0 * qph) / (1.0 - alpha0)

    #     # bit decision from coset of qph: near k0 => 0, near k0+Delta/2 => 1
    #     # turn coset into {0,1}
    #     coset = frac(qph - k0, Delta)  # in [0, Delta)
    #     mhat = np.round(2.0 * coset / Delta).astype(int)  # near 0 -> 0, near Delta/2 -> 1
    #     mhat = np.clip(mhat, 0, 1)
    #     return xhat, mhat, qph

    # # ---------- Detection score to sweep alpha0 ----------
    # def lattice_distance_score(y, Delta=1.0, alpha0=0.6, beta0=4.0, k0=0.0):
    #     """
    #     A self-consistency score: distance to lattice after one-step demix.
    #     Smaller is better; pick argmin over alpha0.
    #     """
    #     xhat, mhat, qph = demix_and_bits(y, Delta, alpha0, beta0, k0)

    #     # Recompute the "should-be" lattice point from xhat & decided bits (one refinement)
    #     dmhat = (Delta/2.0) * mhat
    #     u2 = beta0 * (xhat - (dmhat + k0))
    #     q2 = Q_delta(u2, Delta)

    #     # Mean squared distance to the lattice
    #     return np.mean((u2 - q2)**2)

    # # ---------- Demo ----------
    # n = 10000
    # Delta = 1.0
    # alpha_true = 0.62
    # beta_true  = 5.0

    # # Host and message
    # x = rng.normal(0.0, 1.0, size=n)
    # m = rng.integers(0, 2, size=n)  # 0/1 uniformly

    # # ----- Case A: public k (scalar 0.0) -> α-sweep reveals alpha -----
    # k_public = 0.0
    # y, _ = embed_beta_rqim(x, m, Delta=Delta, alpha=alpha_true, beta=beta_true, k=k_public)

    # alphas = np.linspace(0.05, 0.95, 181)  # sweep grid
    # scores_public = []
    # rec_err_public = []

    # for a0 in alphas:
    #     xhat, mhat, _ = demix_and_bits(y, Delta=Delta, alpha0=a0, beta0=beta_true, k0=k_public)
    #     scores_public.append(lattice_distance_score(y, Delta=Delta, alpha0=a0, beta0=beta_true, k0=k_public))
    #     rec_err_public.append(np.mean((xhat - x)**2))

    # alpha_hat_public = alphas[np.argmin(scores_public)]

    # print(f"[Public-k] argmin score alpha0 = {alpha_hat_public:.4f} (true alpha = {alpha_true})")
    # print(f"Public-k: min score = {np.min(scores_public):.6e}, MSE at min = {rec_err_public[np.argmin(scores_public)]:.6e}")

    # # ----- Case B: secret per-coordinate k_i -> α-sweep fails without the key -----
    # # Secret, per-coordinate k_i sampled from a PRNG with a (hidden) seed, limited to [0, Delta)
    # k_secret = rng.uniform(0.0, Delta, size=n)   # this is the "keyed" k_i
    # y_sec, _ = embed_beta_rqim(x, m, Delta=Delta, alpha=alpha_true, beta=beta_true, k=k_secret)

    # # Attacker wrongly assumes k0=0 (no key)
    # scores_secret_wrong = []
    # for a0 in alphas:
    #     scores_secret_wrong.append(lattice_distance_score(y_sec, Delta=Delta, alpha0=a0, beta0=beta_true, k0=0.0))

    # # (For reference) Legit receiver who knows the key recovers the sharp minimum
    # scores_secret_right = []
    # for a0 in alphas:
    #     scores_secret_right.append(lattice_distance_score(y_sec, Delta=Delta, alpha0=a0, beta0=beta_true, k0=k_secret))

    # alpha_hat_secret_right = alphas[np.argmin(scores_secret_right)]

    # print(f"[Secret-k, correct key] argmin score alpha0 = {alpha_hat_secret_right:.4f} (true alpha = {alpha_true})")
    # print(f"Secret-k: min score (correct-key) = {np.min(scores_secret_right):.6e}")

    # # ---------- Plots ----------
    # plt.figure(figsize=(7,4.5))
    # plt.plot(alphas, scores_public, label="Score (public k=0)")
    # plt.axvline(alpha_true, linestyle="--", label=f"true α = {alpha_true}")
    # plt.title("Detection score vs α₀ (public k)")
    # plt.xlabel("α₀ (sweep)")
    # plt.ylabel("lattice-distance score (lower=better)")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(7,4.5))
    # plt.plot(alphas, scores_secret_wrong, label="Score (secret kᵢ, attacker assumes k₀=0)")
    # plt.plot(alphas, scores_secret_right, label="Score (secret kᵢ, correct key)")
    # plt.axvline(alpha_true, linestyle="--", label=f"true α = {alpha_true}")
    # plt.title("Detection score vs α₀ with secret per-coordinate kᵢ")
    # plt.xlabel("α₀ (sweep)")
    # plt.ylabel("lattice-distance score (lower=better)")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(7,4.5))
    # plt.plot(alphas, rec_err_public)
    # plt.axvline(alpha_true, linestyle="--", label=f"true α = {alpha_true}")
    # plt.title("Recovery MSE vs α₀ (public k)")
    # plt.xlabel("α₀ (sweep)")
    # plt.ylabel("MSE( x̂(α₀) , x )")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

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
    sys.exit(test_qim_1(delta=args.d,embedding_alpha=args.a,k=args.k,plot=True,test=True))