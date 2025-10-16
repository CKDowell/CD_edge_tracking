# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 15:18:16 2025

@author: dowel
"""

import numpy as np
from scipy.stats import vonmises
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ------------------------
# Utility functions
# ------------------------
def wrap_ang(a):
    """Wrap angles to [-pi, pi)."""
    return (a + np.pi) % (2*np.pi) - np.pi

def logsumexp(a, axis=None):
    """Stable log-sum-exp."""
    a_max = np.max(a, axis=axis, keepdims=True)
    s = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if axis is None:
        return s.ravel()[0]
    return s

def estimate_kappa_from_R(Rbar):
    """
    Approximate kappa for von Mises from resultant length Rbar (0..1).
    Uses standard approximations (best for moderate/high Rbar).
    """
    if Rbar < 1e-6:
        return 1e-6
    if Rbar < 0.53:
        return 2*Rbar + Rbar**3 + 5*Rbar**5/6
    elif Rbar < 0.85:
        return -0.4 + 1.39*Rbar + 0.43/(1-Rbar)
    else:
        return 1.0/(Rbar**3 - 4*Rbar**2 + 3*Rbar)

# ------------------------
# HMM + Baum-Welch
# ------------------------
class VonMisesHMM:
    def __init__(self, n_states=4):
        assert n_states == 4, "This implementation uses 4 states (x1/x2 × flip/no-flip)."
        self.N = n_states
        self.trans = None       # transition matrix (NxN)
        self.pi0 = None         # initial state probabilities (N,)
        self.x1 = None          # base angle for component 1
        self.x2 = None          # base angle for component 2
        
        self.kappa = None       # concentration param (shared)
    
    def _init_params(self, y):
        T = len(y)
        # init pi0 uniform
        self.pi0 = np.full(self.N, 1.0/self.N)
        # init trans as slightly noisy identity (tendency to stay)
        p_stay = 0.95
        self.trans = np.full((self.N, self.N), (1-p_stay)/(self.N-1))
        np.fill_diagonal(self.trans, p_stay)
        # initialize base angles x1,x2 by a simple 2-means on unit vectors
        # cluster on (cos, sin)
        X = np.column_stack([np.cos(y), np.sin(y)])
        # simple Lloyd's k-means for two clusters
        rng = np.random.default_rng(0)
        centers = X[rng.choice(len(y), size=2, replace=False)]
        for _ in range(20):
            d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(d, axis=1)
            new_centers = np.array([X[labels==k].mean(axis=0) if np.any(labels==k) else centers[k] for k in (0,1)])
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        # convert centers back to angles
        angs = np.arctan2(centers[:,1], centers[:,0])
        self.x1, self.x2 = angs[0], angs[1]
        # init kappa by moment from overall resultant length
        R = np.abs(np.mean(np.exp(1j*y)))
        self.kappa = max(1.0, estimate_kappa_from_R(R))  # small floor
    
    def _compute_log_emission(self, y):
        """
        Return array (T, N) of log p(y_t | state).
        state -> mean as:
         state 0: x1
         state 1: x1 + pi
         state 2: x2
         state 3: x2 + pi
        """
        T = len(y)
        mus = np.array([self.x1, self.x1 + np.pi, self.x2, self.x2 + np.pi])
        mus = wrap_ang(mus)
        # scipy vonmises.logpdf supports vectorized loc but not vectorized kappa/state per element easily;
        # we'll loop over states (small N=4)
        log_em = np.zeros((T, self.N))
        for s in range(self.N):
            log_em[:, s] = vonmises.logpdf(y, self.kappa, loc=mus[s])
        return log_em
    
    def _forward_backward(self, log_em):
        """
        Compute forward-backward in log-space.
        returns:
          log_alpha (T,N), log_beta (T,N), log_likelihood (scalar)
        """
        T = log_em.shape[0]
        logA = np.log(self.trans + 1e-300)  # (N,N)
        logpi0 = np.log(self.pi0 + 1e-300)
        log_alpha = np.zeros((T, self.N))
        log_beta = np.zeros((T, self.N))
        # forward
        log_alpha[0] = logpi0 + log_em[0]
        for t in range(1, T):
            # log_alpha[t,j] = log_em[t,j] + logsum_i( log_alpha[t-1,i] + logA[i,j] )
            prev = log_alpha[t-1][:, None] + logA  # (N,N) where col j is i -> j
            log_alpha[t] = log_em[t] + logsumexp(prev, axis=0)
        # log-likelihood
        log_lik = logsumexp(log_alpha[-1])
        # backward
        log_beta[-1] = 0.0
        for t in range(T-2, -1, -1):
            # log_beta[t,i] = logsum_j( logA[i,j] + log_em[t+1,j] + log_beta[t+1,j] )
            right = logA + (log_em[t+1] + log_beta[t+1])[None, :]
            log_beta[t] = np.squeeze(logsumexp(right, axis=1))

        return log_alpha, log_beta, log_lik
    
    def fit(self, y, max_iters=100, tol=1e-6, verbose=False):
        """
        Fit HMM via Baum-Welch (EM).
        y: array of angles in radians ([-pi,pi) or any)
        """
        y = wrap_ang(np.asarray(y))
        T = len(y)
        self._init_params(y)
        
        prev_ll = -np.inf
        for it in range(max_iters):
            # E-step: compute posteriors
            log_em = self._compute_log_emission(y)  # (T,N)
            log_alpha, log_beta, log_lik = self._forward_backward(log_em)
            # posterior gamma_t(s)
            log_gamma = log_alpha + log_beta
            log_gamma = log_gamma - np.squeeze(logsumexp(log_gamma, axis=1))[:, None]
            gamma = np.exp(log_gamma)  # (T,N)
            # expected transitions xi_t(i,j) proportional to alpha[t,i] * A[i,j] * emission[t+1,j] * beta[t+1,j]
            logA = np.log(self.trans + 1e-300)
            xi_sum = np.zeros((self.N, self.N))
            for t in range(T-1):
                # compute matrix of shape (N,N): log_alpha[t,i] + logA[i,j] + log_em[t+1,j] + log_beta[t+1,j]
                M = log_alpha[t][:, None] + logA + log_em[t+1][None, :] + log_beta[t+1][None, :]
                M = M - logsumexp(M)  # normalize to log-probabilities
                xi_sum += np.exp(M)
            # M-step:
            # 1) Update pi0
            self.pi0 = gamma[0] / np.sum(gamma[0])
            # 2) Update transition matrix
            trans_new = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-12)
            # Small regularization to avoid zeros
            trans_new = (trans_new + 1e-8)
            trans_new = trans_new / trans_new.sum(axis=1, keepdims=True)
            self.trans = trans_new
            # 3) Update base angles x1 and x2:
            # For x1: combine responsibilities from state 0 (no flip) and state 1 (flip),
            # but rotate flip observations by -pi so they align with base x1.
            # Weighted complex sum gives circular mean.
            w_x1 = gamma[:, 0]
            w_x1_flip = gamma[:, 1]
            z1 = np.sum(w_x1 * np.exp(1j * y) + w_x1_flip * np.exp(1j * (y - np.pi)))
            if np.abs(z1) < 1e-12:
                # avoid zero vector, keep prior
                x1_new = self.x1
            else:
                x1_new = np.angle(z1)
            # For x2 similarly
            w_x2 = gamma[:, 2]
            w_x2_flip = gamma[:, 3]
            z2 = np.sum(w_x2 * np.exp(1j * y) + w_x2_flip * np.exp(1j * (y - np.pi)))
            if np.abs(z2) < 1e-12:
                x2_new = self.x2
            else:
                x2_new = np.angle(z2)
            self.x1, self.x2 = wrap_ang(x1_new), wrap_ang(x2_new)
            # 4) Update kappa (shared) by resultant length weighted over all states relative to their respective means.
            # For each state compute difference to its state's mean then compute weighted resultant.
            mus = np.array([self.x1, self.x1 + np.pi, self.x2, self.x2 + np.pi])
            mus = wrap_ang(mus)
            # weighted resultant length:
            sumx = 0.0
            sumw = 0.0
            for s in range(self.N):
                w = gamma[:, s]
                dx = wrap_ang(y - mus[s])
                sumx += np.sum(w * np.cos(dx))
                sumw += np.sum(w)
            Rbar = (sumx / sumw) if sumw > 0 else 0.0
            Rbar = np.clip(Rbar, 1e-8, 0.999999)
            kappa_new = estimate_kappa_from_R(Rbar)
            # keep kappa not too small
            self.kappa = max(1e-3, kappa_new)
            
            if verbose:
                print(f"Iter {it:3d}: loglik = {log_lik:.4f}, x1={np.degrees(self.x1):.1f}°, x2={np.degrees(self.x2):.1f}°, kappa={self.kappa:.3f}")
            # check convergence in log-lik
            if np.abs(log_lik - prev_ll) < tol:
                break
            prev_ll = log_lik
        
        # final posterior gamma and loglik
        self._last_log_em = log_em
        self._last_log_alpha = log_alpha
        self._last_log_beta = log_beta
        self._last_gamma = gamma
        self._last_loglik = prev_ll
        return self
    
    def posterior_states(self):
        """Return posterior state probabilities gamma (T,N) from last fit."""
        return getattr(self, "_last_gamma", None)
    
    def viterbi(self, y):
        """Compute Viterbi path (most likely state sequence) given data y using current params."""
        y = wrap_ang(np.asarray(y))
        T = len(y)
        log_em = self._compute_log_emission(y)
        logA = np.log(self.trans + 1e-300)
        logpi0 = np.log(self.pi0 + 1e-300)
        delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)
        delta[0] = logpi0 + log_em[0]
        for t in range(1, T):
            temp = delta[t-1][:, None] + logA  # (N,N)
            psi[t] = np.argmax(temp, axis=0)
            delta[t] = np.max(temp, axis=0) + log_em[t]
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states

# ------------------------
# Example / test with simulated data
# ------------------------

# __name__ =="Charlie test"
# if __name__=="Charlie test":
#         np.random.seed(1)
#         T = 1000
#         #Biased random walk
        
#         x1_true = np.cumsum(np.pi*np.random.normal(0.05,0.1,size=T))
#         x1_true = wrap_ang(x1_true)
#         plt.plot(x1_true)
#         x1_true180 = wrap_ang(x1_true-np.pi)
#         plt.plot(x1_true180)
#         x2_mu = np.deg2rad(0)
#         x3_mu = np.deg2rad(90)
        
#         kappa_true = 12.0
#         x2_true = vonmises.rvs(kappa_true,loc=x2_mu,size=T)
#         plt.plot(x2_true)
#         x3_true = vonmises.rvs(kappa_true,loc=x3_mu,size=T)
#         plt.plot(x2_true)
        
#         base_seq = (np.arange(T) // 100) % 4
        
#         y = np.zeros(T)
#         y[base_seq==0] = x1_true[base_seq==0]
#         y[base_seq==1] = x1_true180[base_seq==1]
#         y[base_seq==2] = x2_true[base_seq==2]
#         y[base_seq==3] = x3_true[base_seq==3]
        
#         base_flip = (np.arange(T) // 100) % 8
#         y[np.mod(base_flip,2)==0] = wrap_ang(y[np.mod(base_flip,2)==0]-np.pi)
        
#         plt.figure()
#         plt.plot(y)
        
        
#         hhm = VonMisesHMM()
        
        
        

if __name__ == "__main__":
    np.random.seed(1)
    T = 500
    x1_true = np.deg2rad(20)
    x2_true = np.deg2rad(140)
    kappa_true = 12.0
    # true hidden states: alternate every 120 timesteps, with random flips occasionally
    base_seq = (np.arange(T) // 120) % 2  # 0 -> x1, 1 -> x2
    flips = (np.random.rand(T) < 0.06).astype(int)
    mu_true = np.where(base_seq==0, x1_true, x2_true)
    mu_true = wrap_ang(mu_true + flips * np.pi)
    # sample
    y = vonmises.rvs(kappa_true, loc=mu_true, size=T)
    y = wrap_ang(y)
    
    # Fit HMM
    hmm = VonMisesHMM()
    hmm.fit(y, max_iters=200, tol=1e-6, verbose=True)
    
    print("\nTrue x1=%.1f°, x2=%.1f°, kappa=%.2f" % (np.degrees(x1_true), np.degrees(x2_true), kappa_true))
    print("Estimated x1=%.1f°, x2=%.1f°, kappa=%.2f" % (np.degrees(hmm.x1), np.degrees(hmm.x2), hmm.kappa))
    
    # get posterior and Viterbi
    gamma = hmm.posterior_states()
    vpath = hmm.viterbi(y)
    # Map vpath -> base & flip
    base_est = (vpath >= 2).astype(int)
    flips_est = ((vpath % 2) == 1).astype(int)
    # crude accuracy check
    print("Base accuracy (fraction):", np.mean(base_est == base_seq))
    print("Flip detection rate (fraction true flips detected):", np.sum((flips==1) & (flips_est==1)) / max(1, flips.sum()))
    
    
    y_unwrap = np.unwrap(y)
    mu_true_unwrap = np.unwrap(mu_true)
    vpath = hmm.viterbi(y)
    
    # --- Compute estimated means for plotting ---
    x1_est, x2_est = hmm.x1, hmm.x2
    kappa_est = hmm.kappa
    mus_est = np.array([x1_est, x1_est + np.pi, x2_est, x2_est + np.pi])
    mus_est = (mus_est + np.pi) % (2*np.pi) - np.pi
    
    # map viterbi state to its mean
    mu_vit = mus_est[vpath]
    mu_vit_unwrap = np.unwrap(mu_vit)
    
    # --- Figure layout ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # 1️⃣ Raw data and means
    axes[0].plot(y_unwrap, 'k.', alpha=0.5, label='data (unwrap)')
    axes[0].plot(mu_true_unwrap, 'r--', lw=2, label='true mean')
    axes[0].plot(mu_vit_unwrap, 'b-', lw=2, label='fitted mean (Viterbi)')
    axes[0].set_ylabel("Angle (radians, unwrapped)")
    axes[0].legend()
    axes[0].set_title("Circular data, true vs. fitted means")
    
    # 2️⃣ Posterior probabilities per state
    gamma = hmm.posterior_states()
    axes[1].imshow(
        gamma.T,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        extent=[0, len(y), 0, 4]
    )
    axes[1].set_yticks(np.arange(4) + 0.5)
    axes[1].set_yticklabels(['x1', 'x1+π', 'x2', 'x2+π'])
    axes[1].set_ylabel("State")
    axes[1].set_title("Posterior state probabilities (γ_t)")
    
    # 3️⃣ Viterbi-decoded states
    axes[2].plot(vpath, 'k-', lw=2)
    axes[2].set_ylabel("Viterbi state")
    axes[2].set_xlabel("Time index")
    axes[2].set_title("Most likely (Viterbi) state sequence")
    axes[2].set_yticks([0,1,2,3])
    axes[2].set_yticklabels(['x1','x1+π','x2','x2+π'])
    
    plt.tight_layout()
    plt.show()
#%% More complex HMM that has flips on top of states

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises
from scipy.special import i0, logsumexp

def wrap_ang(a):
    return (a + np.pi) % (2*np.pi) - np.pi

# -----------------------------
# Factorial HMM class
# -----------------------------
class FactorialCircularHMM:
    def __init__(self, x1_known, n_base=3):
        """
        x1_known: array of length T for known trajectory
        n_base: number of base states (here 3: x1,x2,x3)
        """
        self.x1_known = x1_known
        self.T = len(x1_known)
        self.n_base = n_base
        self.n_flip = 2
        # Initialize parameters
        self.mu = np.array([x1_known[0], 0.0, np.pi/2])  # mu[0] for x1, mu[1] unknown x2, mu[2] unknown x3
        self.kappa = 5.0
        # Transition matrices
        self.A_base = np.ones((n_base, n_base))/n_base
        self.A_flip = np.array([[0.95,0.05],[0.05,0.95]])
        # Initial probabilities
        self.pi_base = np.ones(n_base)/n_base
        self.pi_flip = np.array([0.5,0.5])

    # Compute emission means for all joint states
    def emission_means(self):
        # joint state index: (base, flip) -> 0..5
        means = np.zeros((self.T, self.n_base, self.n_flip))
        means[:,0,0] = self.x1_known
        means[:,0,1] = wrap_ang(self.x1_known + np.pi)
        means[:,1,0] = self.mu[1]
        means[:,1,1] = wrap_ang(self.mu[1] + np.pi)
        means[:,2,0] = self.mu[2]
        means[:,2,1] = wrap_ang(self.mu[2] + np.pi)
        return means

    # Compute joint emission probability: shape (T, n_base, n_flip)
    def emission_prob(self, y):
        means = self.emission_means()
        # p(y_t | s,f)
        B = np.exp(self.kappa * np.cos(y[:,None,None] - means)) / (2*np.pi*i0(self.kappa))
        return B

    # Forward-backward on joint space
    def forward_backward(self, y):
        T = self.T
        B = self.emission_prob(y)
        alpha = np.zeros_like(B)
        beta = np.zeros_like(B)
        # joint state initial: pi_base * pi_flip
        alpha[0] = np.outer(self.pi_base, self.pi_flip) * B[0]
        alpha[0] /= alpha[0].sum()
        for t in range(1,T):
            alpha[t] = np.zeros_like(alpha[0])
            for b2 in range(self.n_base):
                for f2 in range(self.n_flip):
                    # sum over previous joint states
                    sum_prev = 0.0
                    for b1 in range(self.n_base):
                        for f1 in range(self.n_flip):
                            sum_prev += alpha[t-1,b1,f1] * self.A_base[b1,b2] * self.A_flip[f1,f2]
                    alpha[t,b2,f2] = B[t,b2,f2] * sum_prev
            alpha[t] /= alpha[t].sum()
        # backward
        beta[-1] = 1.0
        for t in range(T-2,-1,-1):
            for b1 in range(self.n_base):
                for f1 in range(self.n_flip):
                    sum_next = 0.0
                    for b2 in range(self.n_base):
                        for f2 in range(self.n_flip):
                            sum_next += self.A_base[b1,b2] * self.A_flip[f1,f2] * B[t+1,b2,f2] * beta[t+1,b2,f2]
                    beta[t,b1,f1] = sum_next
            beta[t] /= beta[t].sum()
        # posterior gamma_t(b,f)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=(1,2), keepdims=True)
        return gamma

    # Fit via EM
    def fit(self, y, max_iters=50, verbose=True):
        y = wrap_ang(np.asarray(y))
        T = len(y)
        for it in range(max_iters):
            gamma = self.forward_backward(y)
            # update base transition
            xi_base = np.zeros_like(self.A_base)
            for b1 in range(self.n_base):
                for b2 in range(self.n_base):
                    xi_base[b1,b2] = np.sum(gamma[:-1,b1,:]*gamma[1:,b2,:])
            xi_base /= xi_base.sum(axis=1, keepdims=True)
            self.A_base = xi_base
            # update flip transition
            xi_flip = np.zeros_like(self.A_flip)
            for f1 in range(self.n_flip):
                for f2 in range(self.n_flip):
                    xi_flip[f1,f2] = np.sum(gamma[:-1,:,f1]*gamma[1:,:,f2])
            xi_flip /= xi_flip.sum(axis=1, keepdims=True)
            self.A_flip = xi_flip
            # update mu[1] and mu[2] (x2 and x3) via circular mean weighted by gamma
            for b_idx, mu_idx in zip([1,2],[1,2]):
                w = gamma[:,b_idx,:].sum(axis=1)  # sum over flip
                y_adj = np.concatenate([y, y-np.pi])
                w_adj = np.concatenate([w, w])
                z = np.sum(w[:,None] * np.exp(1j * np.stack([y, y-np.pi], axis=1)), axis=(0,1))
                self.mu[mu_idx] = np.angle(z)
            # update kappa using approximate method
            means = self.emission_means()
            R = np.abs(np.sum(gamma * np.exp(1j*(y[:,None,None]-means)))/T)
            self.kappa = (R*(2-R**2))/(1-R**2 + 1e-6)
            if verbose:
                print(f"Iter {it}: mu2={np.degrees(self.mu[1]):.1f}, mu3={np.degrees(self.mu[2]):.1f}, kappa={self.kappa:.2f}")

    # Viterbi decoding on joint states
    def viterbi(self, y):
        T = self.T
        B = self.emission_prob(y)
        delta = np.zeros_like(B)
        psi = np.zeros_like(B, dtype=int)
        delta[0] = np.log(np.outer(self.pi_base,self.pi_flip)+1e-12) + np.log(B[0]+1e-12)
        for t in range(1,T):
            for b2 in range(self.n_base):
                for f2 in range(self.n_flip):
                    temp = delta[t-1] + np.log(self.A_base[:,b2][:,None] * self.A_flip[:,f2][None,:]+1e-12)
                    psi[t,b2,f2] = np.unravel_index(np.argmax(temp), temp.shape)[0]
                    delta[t,b2,f2] = np.max(temp) + np.log(B[t,b2,f2]+1e-12)
        # backtrack
        states = np.zeros(T, dtype=int)
        states_flip = np.zeros(T, dtype=int)
        b,f = np.unravel_index(np.argmax(delta[-1]), delta[-1].shape)
        states[-1] = b
        states_flip[-1] = f
        for t in range(T-2,-1,-1):
            b = psi[t+1,b,f]
            f = np.argmax(delta[t,b,:])
            states[t] = b
            states_flip[t] = f
        return states, states_flip




# Simulate y using your example with x1_known
T = 600
np.random.seed(1)
x1_true = np.cumsum(np.pi*np.random.normal(0.05,0.1,size=T))
x1_true = wrap_ang(x1_true)
x2_mu = np.deg2rad(0)
x3_mu = np.deg2rad(90)
kappa_true = 12.0
x2_true = vonmises.rvs(kappa_true,loc=x2_mu,size=T)
x3_true = vonmises.rvs(kappa_true,loc=x3_mu,size=T)

base_seq = (np.arange(T)//100) % 4
y = np.zeros(T)
y[base_seq==0] = x1_true[base_seq==0]
y[base_seq==1] = x1_true[base_seq==1]
y[base_seq==2] = x2_true[base_seq==2]
y[base_seq==3] = x3_true[base_seq==3]
base_flip = (np.arange(T)//100) % 13
y[np.mod(base_flip,2)==0] = wrap_ang(y[np.mod(base_flip,2)==0]-np.pi)

# Fit factorial HMM
fhm = FactorialCircularHMM(x1_true)
fhm.fit(y, max_iters=20, verbose=True)
states_base, states_flip = fhm.viterbi(y)

# Visualization
plt.figure(figsize=(10,4))
plt.plot(np.unwrap(y), 'k.', alpha=0.5)
plt.plot(np.unwrap(x1_true), 'g--', label='x1 known')
plt.axhline(fhm.mu[1], color='r', lw=2, label='x2 est')
plt.axhline(fhm.mu[2], color='b', lw=2, label='x3 est')
plt.title("Factorial HMM fit")
plt.legend()
plt.show()
#%% 
plt.plot(y,color='k')
ypred = np.zeros(len(y))
ypred[states_base==0] = x1_true[states_base==0]
ypred[states_base==1] = x2_mu
ypred[states_base==2] = x3_mu
#plt.plot(ypred,color='b')
ypred[states_flip==1] = wrap_ang(ypred[states_flip==1]-np.pi)
plt.plot(ypred,color='r')

plt.plot(states_flip)
#%% H delta fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises
from scipy.special import i0, logsumexp

def wrap_ang(a):
    return (a + np.pi) % (2*np.pi) - np.pi

# -----------------------------
# Factorial HMM class
# -----------------------------
class hDelta_CircularHMM:
    def __init__(self, x1_known):
        """
        x1_known: array of size T x num bases
        
        """
        self.x1_known = x1_known
        self.T = len(x1_known)
        self.n_base = x1_known.shape[1]
        self.n_flip = 2
        # Initialize parameters
        self.mu = x1_known[0,:]  # mu[0] for x1, mu[1] unknown x2, mu[2] unknown x3
        self.kappa = 5.0
        # Transition matrices
        self.A_base = np.ones((self.n_base, self.n_base))/self.n_base
        self.A_flip = np.array([[0.95,0.05],[0.05,0.95]])
        # Initial probabilities
        self.pi_base = np.ones(self.n_base)/self.n_base
        self.pi_flip = np.array([0.5,0.5])

    # Compute emission means for all joint states
    def emission_means(self):
        # joint state index: (base, flip) -> 0..5
        means = np.zeros((self.T, self.n_base, self.n_flip))
        means[:,:,0] = self.x1_known
        means[:,:,1] = wrap_ang(self.x1_known + np.pi)
        # means[:,1,0] = self.mu[1]
        # means[:,1,1] = wrap_ang(self.mu[1] + np.pi)
        # means[:,2,0] = self.mu[2]
        # means[:,2,1] = wrap_ang(self.mu[2] + np.pi)
        return means

    # Compute joint emission probability: shape (T, n_base, n_flip)
    def emission_prob(self, y):
        means = self.emission_means()
        # p(y_t | s,f)
        B = np.exp(self.kappa * np.cos(y[:,None,None] - means)) / (2*np.pi*i0(self.kappa))
        return B

    # Forward-backward on joint space
    def forward_backward(self, y):
        T = self.T
        B = self.emission_prob(y)
        alpha = np.zeros_like(B)
        beta = np.zeros_like(B)
        # joint state initial: pi_base * pi_flip
        alpha[0] = np.outer(self.pi_base, self.pi_flip) * B[0]
        alpha[0] /= alpha[0].sum()
        for t in range(1,T):
            alpha[t] = np.zeros_like(alpha[0])
            for b2 in range(self.n_base):
                for f2 in range(self.n_flip):
                    # sum over previous joint states
                    sum_prev = 0.0
                    for b1 in range(self.n_base):
                        for f1 in range(self.n_flip):
                            sum_prev += alpha[t-1,b1,f1] * self.A_base[b1,b2] * self.A_flip[f1,f2]
                    alpha[t,b2,f2] = B[t,b2,f2] * sum_prev
            alpha[t] /= alpha[t].sum()
        # backward
        beta[-1] = 1.0
        for t in range(T-2,-1,-1):
            for b1 in range(self.n_base):
                for f1 in range(self.n_flip):
                    sum_next = 0.0
                    for b2 in range(self.n_base):
                        for f2 in range(self.n_flip):
                            sum_next += self.A_base[b1,b2] * self.A_flip[f1,f2] * B[t+1,b2,f2] * beta[t+1,b2,f2]
                    beta[t,b1,f1] = sum_next
            beta[t] /= beta[t].sum()
        # posterior gamma_t(b,f)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=(1,2), keepdims=True)
        return gamma

    # Fit via EM
    def fit(self, y, max_iters=50, verbose=True):
        y = wrap_ang(np.asarray(y))
        T = len(y)
        for it in range(max_iters):
            gamma = self.forward_backward(y)
            # update base transition
            xi_base = np.zeros_like(self.A_base)
            for b1 in range(self.n_base):
                for b2 in range(self.n_base):
                    xi_base[b1,b2] = np.sum(gamma[:-1,b1,:]*gamma[1:,b2,:])
            xi_base /= xi_base.sum(axis=1, keepdims=True)
            self.A_base = xi_base
            # update flip transition
            xi_flip = np.zeros_like(self.A_flip)
            for f1 in range(self.n_flip):
                for f2 in range(self.n_flip):
                    xi_flip[f1,f2] = np.sum(gamma[:-1,:,f1]*gamma[1:,:,f2])
            xi_flip /= xi_flip.sum(axis=1, keepdims=True)
            self.A_flip = xi_flip
            # # update mu[1] and mu[2] (x2 and x3) via circular mean weighted by gamma
            # print(gamma.shape)
            # for b_idx, mu_idx in zip([1,2],[1,2]):
            #     w = gamma[:,b_idx,:].sum(axis=1)  # sum over flip
            #     y_adj = np.concatenate([y, y-np.pi])
            #     w_adj = np.concatenate([w, w])
            #     z = np.sum(w[:,None] * np.exp(1j * np.stack([y, y-np.pi], axis=1)), axis=(0,1))
            #     self.mu[mu_idx] = np.angle(z)
            # update kappa using approximate method
            means = self.emission_means()
            R = np.abs(np.sum(gamma * np.exp(1j*(y[:,None,None]-means)))/T)
            self.kappa = (R*(2-R**2))/(1-R**2 + 1e-6)
            if verbose:
                print(f"Iter {it}: ")
                #print(f"Iter {it}: mu2={np.degrees(self.mu[1]):.1f}, mu3={np.degrees(self.mu[2]):.1f}, kappa={self.kappa:.2f}")

    # Viterbi decoding on joint states
    def viterbi(self, y):
        T = self.T
        B = self.emission_prob(y)
        delta = np.zeros_like(B)
        psi = np.zeros_like(B, dtype=int)
        delta[0] = np.log(np.outer(self.pi_base,self.pi_flip)+1e-12) + np.log(B[0]+1e-12)
        for t in range(1,T):
            for b2 in range(self.n_base):
                for f2 in range(self.n_flip):
                    temp = delta[t-1] + np.log(self.A_base[:,b2][:,None] * self.A_flip[:,f2][None,:]+1e-12)
                    psi[t,b2,f2] = np.unravel_index(np.argmax(temp), temp.shape)[0]
                    delta[t,b2,f2] = np.max(temp) + np.log(B[t,b2,f2]+1e-12)
        # backtrack
        states = np.zeros(T, dtype=int)
        states_flip = np.zeros(T, dtype=int)
        b,f = np.unravel_index(np.argmax(delta[-1]), delta[-1].shape)
        states[-1] = b
        states_flip[-1] = f
        for t in range(T-2,-1,-1):
            b = psi[t+1,b,f]
            f = np.argmax(delta[t,b,:])
            states[t] = b
            states_flip[t] = f
        return states, states_flip




#%%
from analysis_funs.regression import fci_regmodel

import numpy as np
import pandas as pd
import src.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt 
from src.utilities import imaging as im
from skimage import io, data, registration, filters, measure
from scipy import signal as sg
from analysis_funs.CX_imaging import CX
from analysis_funs.CX_analysis_col import CX_a
from src.utilities import funcs as fn
from scipy import stats
from Utilities.utils_general import utils_general as ug
#%% Try with hDeltaJ
datadir = r"Y:\\Data\\FCI\\Hedwig\\hDeltaJ\\240529\\f1\\Trial3"
cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
eb_phase = cxa.pdat['phase_eb'][:,np.newaxis]
heading = cxa.ft2['ft_heading'].to_numpy()
e_e = cxa.get_entries_exits_like_jumps()
infgoal = np.zeros((len(eb_phase),1))

for i,e in enumerate(e_e):
    dx = np.arange(e[0],e[2])
    infgoal[dx] = stats.circmean(heading[e[0]-10:e[0]],low=-np.pi,high=np.pi)
    
inputs = np.append(eb_phase,infgoal,axis=1)
fhm = hDelta_CircularHMM(eb_phase)
y = cxa.pdat['phase_fsb_upper']
fhm.fit(y)
states_base, states_flip = fhm.viterbi(y)
plt.plot(states_base)
plt.plot(states_flip)

yfit = np.zeros(len(y))
for i in range(inputs.shape[1]):
    yfit[np.logical_and(states_base==i,states_flip==0)] = inputs[np.logical_and(states_base==i,states_flip==0),i]
    yfit[np.logical_and(states_base==i,states_flip==1)] = ug.circ_subtract(inputs[np.logical_and(states_base==i,states_flip==1),i],np.pi)
#%% 
x = np.arange(0,len(y))
plt.plot(x,y,color='k')
plt.scatter(x[states_flip==0],yfit[states_flip==0],color='r',s=5,zorder=5)
plt.scatter(x[states_flip==1],yfit[states_flip==1],color=[1,0.5,0.5],s=5,zorder=5)
ins = cxa.ft2['instrip'].to_numpy()
plt.plot(x,ins*np.pi*2-np.pi,color=[0.5,0.5,0.5])
#%% Try with hDeltaC
datadir = r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial2"
cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
eb_phase = cxa.pdat['offset_eb_phase'].to_numpy()[:,np.newaxis]
heading = cxa.ft2['ft_heading'].to_numpy()
e_e = cxa.get_entries_exits_like_jumps()
infgoal = np.zeros((len(eb_phase),1))
x = cxa.ft2['ft_posx'].to_numpy()
y = cxa.ft2['ft_posy'].to_numpy()
x,y = cxa.fictrac_repair(x,y)
for i,e in enumerate(e_e):
    idx = np.arange(e[1],e[2])
    tx = x[idx]
    ty = y[idx]
    tx = tx-tx[0]
    ty = ty-ty[0]
    xm = np.argmax(np.abs(tx))
    dx = tx[xm]
    dy = ty[xm]
    infgoal[idx[0]:] = np.arctan(dx/dy) 
    #infgoal[dx] = stats.circmean(heading[e[0]-10:e[0]],low=-np.pi,high=np.pi)
    
inputs = np.append(eb_phase,infgoal,axis=1)
fhm = hDelta_CircularHMM(inputs)
y = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
fhm.fit(y)
states_base, states_flip = fhm.viterbi(y)
plt.plot(states_base)
plt.plot(states_flip)

yfit = np.zeros(len(y))
for i in range(inputs.shape[1]):
    yfit[np.logical_and(states_base==i,states_flip==0)] = inputs[np.logical_and(states_base==i,states_flip==0),i]
    yfit[np.logical_and(states_base==i,states_flip==1)] = ug.circ_subtract(inputs[np.logical_and(states_base==i,states_flip==1),i],np.pi)
#%% 
x = np.arange(0,len(y))
plt.plot(x,y,color='k')
plt.scatter(x[states_flip==0],yfit[states_flip==0],color='r',s=5,zorder=5)
plt.scatter(x[states_flip==1],yfit[states_flip==1],color=[1,0.5,0.5],s=5,zorder=5)
ins = cxa.ft2['instrip'].to_numpy()
plt.plot(x,ins*np.pi*2-np.pi,color=[0.5,0.5,0.5])

#%% Try wit




























