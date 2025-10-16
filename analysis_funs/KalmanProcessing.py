# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 14:19:44 2025

@author: dowel
"""

# Python code to implement a simple Kalman filter for circular variables (angles)
# The approach: represent angle as a 2D unit vector [cos(theta), sin(theta)].
# Use a linear Kalman filter on that 2D vector. After each update, renormalize
# the state to unit length and adjust the covariance appropriately.
#
# The script includes:
# - CircularKalmanFilter class with predict and update methods
# - A simulation demonstrating performance: true angle evolves with constant angular velocity + process noise.
# - No external data required. Plots show true angle, noisy angle measurements, and estimated angle.
#
# This cell will run and produce plots that you can inspect.

import numpy as np
import matplotlib.pyplot as plt

class CircularKalmanFilter:
    def __init__(self, Q_scale=1e-3, R_scale=1e-2):
        """
        Initialize a simple 2D Kalman filter for unit vector representation of an angle.
        State x is 2x1: [cos(theta), sin(theta)].
        P is 2x2 covariance.
        Q_scale: process noise scale (applied to identity)
        R_scale: measurement noise scale (applied to identity)
        """
        self.x = np.array([1.0, 0.0])  # start at angle 0
        self.P = np.eye(2) * 1e-3
        self.Q = np.eye(2) * Q_scale
        self.R = np.eye(2) * R_scale
        # measurement matrix (we measure the 2D unit vector derived from angle)
        self.H = np.eye(2)

    def predict(self):
        # simple random-walk model on the vector components: x_{k+1} = x_k + w, w~N(0,Q)
        # For problems with known angular velocity, incorporate a rotation here.
        # Predicted state and covariance
        self.x = self.x  # identity dynamics
        self.P = self.P + self.Q

    def update(self, measured_angle):
        # Convert angle measurement to unit vector
        z = np.array([np.cos(measured_angle), np.sin(measured_angle)])

        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Innovation (difference in vector space)
        y = z - (self.H @ self.x)

        # Update
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

        # Renormalize state to unit length (project back to circle)
        norm = np.linalg.norm(self.x)
        if norm == 0:
            # avoid division by zero
            self.x = np.array([1.0, 0.0])
            self.P = np.eye(2) * 1e-3
            return
        self.x = self.x.ravel()
        # Jacobian of normalization g(x) = x / ||x|| evaluated at current x
        J = (1.0 / norm) * (np.eye(2) - np.outer(self.x, self.x) / (norm**2))

        # Normalize
        self.x = self.x / norm

        # Adjust covariance using linearization through normalization
        self.P = J @ self.P @ J.T

    def angle(self):
        # return angle estimate in [-pi, pi)
        return np.arctan2(self.x[1], self.x[0])



ckf = CircularKalmanFilter(Q_scale=1e-4, R_scale=0.001)
tangle = cxa.pdat['phase_fsb_upper']
est_angles = np.zeros(len(tangle))

# Increase time increment between predict and update. Also include out of phase data
for k in range(len(est_angles)):
    ckf.predict()
    ckf.update(tangle[k])
    est_angles[k] = ckf.angle()

plt.plot(tangle,color='k')
plt.plot(ug.circ_subtract(tangle,np.pi),color=[0.5,0.5,0.5])
plt.plot(est_angles,color='r')
#%% Two inputs
class CircularKalmanFilter:
    def __init__(self, Q_scale=1e-3, R_scale=1e-2):
        """
        Initialize a simple 2D Kalman filter for unit vector representation of an angle.
        State x is 2x1: [cos(theta), sin(theta)].
        P is 2x2 covariance.
        Q_scale: process noise scale (applied to identity)
        R_scale: measurement noise scale (applied to identity)
        """
        self.x = np.array([1.0, 0.0,1.0,0.0])  # start at angle 0
        self.P = np.eye(4) * 1e-3
        self.Q = np.eye(4) * Q_scale
        self.R = np.eye(4) * R_scale
        # measurement matrix (we measure the 2D unit vector derived from angle)
        self.H = np.eye(4)

    def predict(self):
        # simple random-walk model on the vector components: x_{k+1} = x_k + w, w~N(0,Q)
        # For problems with known angular velocity, incorporate a rotation here.
        # Predicted state and covariance
        self.x = self.x  # identity dynamics
        self.P = self.P + self.Q

    def update(self, measured_theta,measured_phi):
        # Convert angle measurement to unit vector
        z = np.array([
            np.cos(measured_theta), np.sin(measured_theta),
            np.cos(measured_phi),   np.sin(measured_phi)
        ])

        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Innovation (difference in vector space)
        y = z - (self.H @ self.x)

        # Update
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        # Renormalize state to unit length (project back to circle)
        norm = np.linalg.norm(self.x)
        if norm == 0:
            # avoid division by zero
            self.x = np.array([1.0, 0.0,1.0,0.0])
            self.P = np.eye(4) * 1e-3
            return
        self.x = self.x.ravel()
        # Jacobian of normalization g(x) = x / ||x|| evaluated at current x
        J = (1.0 / norm) * (np.eye(4) - np.outer(self.x, self.x) / (norm**2))

        # Normalize
        for i in [0, 2]:
            v = self.x[i:i+2]
            norm = np.linalg.norm(v)
            if norm > 1e-9:
                self.x[i:i+2] = v / norm
            else:
                self.x[i:i+2] = np.array([1.0, 0.0])

        # Adjust covariance using linearization through normalization
        self.P = J @ self.P @ J.T

    def angle(self):
        theta = np.arctan2(self.x[1], self.x[0])
        phi   = np.arctan2(self.x[3], self.x[2])
        return theta, phi



ckf = CircularKalmanFilter(Q_scale=1e-4, R_scale=0.001)
tangle = cxa.pdat['phase_fsb_upper']
#tangle2 = ug.circ_subtract(cxa.pdat['phase_fsb_upper'],np.pi)
tangle2 = cxa.pdat['phase_eb']

# Increase time increment between predict and update. Also include out of phase data
xrange = np.arange(0,len(tangle),5,dtype='int')
est_angles = np.zeros((len(xrange),2))
for ki,k in enumerate(xrange):
    ckf.predict()
    ckf.update(tangle[k],tangle2[k])
    est_angles[ki,:] = ckf.angle()
x = np.arange(0,len(tangle))
plt.scatter(x,tangle,color='k',s=5)
#plt.scatter(x,ug.circ_subtract(tangle,np.pi),color=[0.5,0.5,0.5],s=5)
plt.plot(xrange,est_angles[:,0],color='r')
plt.plot(xrange,ug.circ_subtract(est_angles[:,0],np.pi),color=[1,0.5,0.5])




#%%
# Simulation to demonstrate the filter
np.random.seed(42)

T = 200
dt = 1.0
true_theta = np.zeros(T)
omega = 0.05  # constant angular velocity (radians per step)
process_theta_noise_std = 0.02  # small process noise on angle
measurement_angle_noise_std = 0.3  # measurement noise (radians)

# Generate true angle time series
for k in range(1, T):
    true_theta[k] = true_theta[k-1] + omega * dt + np.random.randn() * process_theta_noise_std

# Generate noisy angle measurements (wrapped to [-pi, pi])
measurements = true_theta + np.random.randn(T) * measurement_angle_noise_std
measurements = (measurements + np.pi) % (2*np.pi) - np.pi

# Run the circular Kalman filter
ckf = CircularKalmanFilter(Q_scale=1e-4, R_scale=0.05)
est_angles = np.zeros(T)












for k in range(T):
    ckf.predict()
    ckf.update(measurements[k])
    est_angles[k] = ckf.angle()

# Unwrap angles for plotting so curves are smooth
true_unwrap = np.unwrap(true_theta)
meas_unwrap = np.unwrap(measurements)
est_unwrap = np.unwrap(est_angles)

# Plot results
plt.figure(figsize=(10, 4))
plt.plot(true_unwrap, label='True angle')
plt.plot(meas_unwrap, label='Measured angle (noisy)', linestyle='dotted')
plt.plot(est_unwrap, label='Estimated angle (CKF)')
plt.legend()
plt.title('Circular Kalman Filter — angle estimate')
plt.xlabel('Time step')
plt.ylabel('Angle (radians)')
plt.grid(True)
plt.show()

# Plot estimation error (wrapped residual)
residual = np.angle(np.exp(1j*(est_angles - true_theta)))  # principal angle difference
plt.figure(figsize=(10, 3))
plt.plot(residual)
plt.title('Estimation error (wrapped)')
plt.xlabel('Time step')
plt.ylabel('Error (radians)')
plt.grid(True)
plt.show()

# Print final RMSE (in radians)
rmse = np.sqrt(np.mean((np.angle(np.exp(1j*(est_angles - true_theta))))**2))
print(f'Final RMSE (radians): {rmse:.4f}')
print(f'Final RMSE (degrees): {rmse * 180/np.pi:.3f}')
