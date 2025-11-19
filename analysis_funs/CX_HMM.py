# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 17:00:19 2025

@author: dowel

Script will model phase data as a HHM
The states can then be used to run other models

"""

import numpy as np

from scipy.stats import vonmises

#%% 
class CX_HMM:
    def __init__(self,n_states): 
        #self.cxa = cxa
        self.N = n_states
        self.trans = None       # transition matrix (NxN)
        self.pi0 = None         # initial state probabilities (N,)
        self.angles = None
        
        self.kappa = None       
    def _init_params(self, y):
        T = len(y)
        # init pi0 uniform
        self.pi0 = np.full(self.N, 1.0/self.N)
        # init trans as slightly noisy identity (tendency to stay)
        p_stay = 0.95
        self.trans = np.full((self.N, self.N), (1-p_stay)/(self.N-1))
        np.fill_diagonal(self.trans, p_stay)
       
       
        # initialize means using N-means clustering on (cos,sin)
        X = np.column_stack([np.cos(y), np.sin(y)])
        rng = np.random.default_rng(0)
        centers = X[rng.choice(len(y), size=self.N, replace=False)]
        for _ in range(30):
            d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(d, axis=1)
            new_centers = np.array(
                [X[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
                 for k in range(self.N)]
            )
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        self.angles = np.arctan2(centers[:, 1], centers[:, 0])

        # global kappa from overall resultant length
        R = np.abs(np.mean(np.exp(1j * y)))
        self.kappa = np.full(self.N, max(1.0, self.estimate_kappa_from_R(R)))
   
    def _compute_log_emission(self, y,x):
         """Compute log p(y_t | state s) for each t,s."""
         T = len(y)
         log_em = np.zeros((T, self.N))
         for s in range(self.N):
             mean_s = self.wrap_ang(x + self.angles[s])
             log_em[:, s] = vonmises.logpdf(y, self.kappa[s], loc=mean_s)
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
            log_alpha[t] = log_em[t] + self.logsumexp(prev, axis=0)
        # log-likelihood
        log_lik = self.logsumexp(log_alpha[-1])
        # backward
        log_beta[-1] = 0.0
        for t in range(T-2, -1, -1):
            # log_beta[t,i] = logsum_j( logA[i,j] + log_em[t+1,j] + log_beta[t+1,j] )
            right = logA + (log_em[t+1] + log_beta[t+1])[None, :]
            log_beta[t] = np.squeeze(self.logsumexp(right, axis=1))
    
        return log_alpha, log_beta, log_lik
   
    def fit(self, y,x, max_iters=100, tol=1e-6, verbose=False):
        """
        Fit HMM via Baum-Welch (EM).
        y: array of angles in radians ([-pi,pi) or any)
        """
        y = self.wrap_ang(np.asarray(y))
        T = len(y)
        self._init_params(y)
        
        prev_ll = -np.inf
        for it in range(max_iters):
            # E-step: compute posteriors
            log_em = self._compute_log_emission(y,x)  # (T,N)
            log_alpha, log_beta, log_lik = self._forward_backward(log_em)
            # posterior gamma_t(s)
            log_gamma = log_alpha + log_beta
            log_gamma = log_gamma - np.squeeze(self.logsumexp(log_gamma, axis=1))[:, None]
            gamma = np.exp(log_gamma)  # (T,N)
            # expected transitions xi_t(i,j) proportional to alpha[t,i] * A[i,j] * emission[t+1,j] * beta[t+1,j]
            logA = np.log(self.trans + 1e-300)
            xi_sum = np.zeros((self.N, self.N))
            for t in range(T-1):
                # compute matrix of shape (N,N): log_alpha[t,i] + logA[i,j] + log_em[t+1,j] + log_beta[t+1,j]
                M = log_alpha[t][:, None] + logA + log_em[t+1][None, :] + log_beta[t+1][None, :]
                M = M - self.logsumexp(M)  # normalize to log-probabilities
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
            
            z = np.sum(gamma * np.exp(1j * (y-x)[:,None]), axis=0)
            R_s = np.abs(z) / np.maximum(np.sum(gamma, axis=0), 1e-12)
            mu_s = np.angle(z)
            kappa_s = np.array([self.estimate_kappa_from_R(R) for R in R_s])
            self.angles = self.wrap_ang(mu_s)
            self.kappa = kappa_s
       
            if verbose:
                print(f"Iter {it:3d}  logL={log_lik:.4f}")
            if np.abs(log_lik - prev_ll) < tol:
                if verbose:
                    print("Converged.")
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
    
    def viterbi(self, y,x):
        """Compute Viterbi path (most likely state sequence) given data y using current params."""
        y = self.wrap_ang(np.asarray(y))
        x = self.wrap_ang(np.asarray(x))
        T = len(y)
        log_em = self._compute_log_emission(y,x)
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
    # Utility functions
    # ------------------------
    def wrap_ang(self,a):
        """Wrap angles to [-pi, pi)."""
        return (a + np.pi) % (2*np.pi) - np.pi
   
    def logsumexp(self,a, axis=None):
        """Stable log-sum-exp."""
        a_max = np.max(a, axis=axis, keepdims=True)
        s = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
        if axis is None:
            return s.ravel()[0]
        return s
    def estimate_kappa_from_R(self,Rbar):
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
   
     