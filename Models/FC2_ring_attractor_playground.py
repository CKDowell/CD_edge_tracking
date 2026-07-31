# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 12:15:30 2026

@author: dowel
"""

#%% test on toy
# 

#Sweep through different input offsets
net = ring_attractor(kernel='connectome',w_inh=8,kappa_exc=10,gain=.2,hDelta_exc=.15)

thetas = np.linspace(-np.pi,np.pi,50)
r0 = None
outplot = np.zeros((len(thetas),16))
angles = np.zeros(len(thetas))
for it,t in enumerate(thetas):
   # inputs = net.inputs([t,0],[.25,1],[2,2])
    inputs = net.input_cosine([t,0],[.3,1])
    r = net.run(inputs,T=2)
    outplot[it,:] = r
    ang,L = net.decode(r)
    angles[it] = ang
plt.imshow(outplot,aspect='auto')
angplt = 8*(angles+np.pi)/np.pi
y = np.arange(0,50)
plt.scatter(angplt,y,color='r')

# Same offset different inhib strength inputs differ by half

#%%

inhib = 10**np.linspace(0,1.2,50)
r0 = None
outplot = np.zeros((len(thetas),16))
for it,i in enumerate(inhib):
    net = ring_attractor(kernel='connectome',w_inh=i,kappa_exc=10,gain=.2,hDelta_exc=.15,J0=6)
    inputs = net.inputs([-np.pi/2,0],[.75,1],[5,2])
    r = net.run(inputs,T=5)
    outplot[it,:] = r
    
plt.figure()
plt.imshow(outplot,aspect='auto')

#%% Import data and see how it does
from EdgeTrackingOriginal.ETpap_plots.ET_paper import ET_paper
from Utilities.utils_general import utils_general as ug
plt.close('all')
datadir = r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260429\f2\Trial1'
etp = ET_paper(datadir,regions=['eb_ch1','fsb1_ch1','fsb2_ch2'])

#%% Fit params to data
if __name__ == '__main__': 
    datadir = r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260429\f2\Trial1'

    etp = ET_paper(datadir,regions=['eb_ch1','fsb1_ch1','fsb2_ch2'])
    U = np.asarray(etp.cxa.pdat['wedges_fsb2_ch2'], float)   # hDeltaC input
    Y = np.asarray(etp.cxa.pdat['wedges_fsb1_ch1'], float)   # FC2 target
    T = min(len(U), len(Y)); U, Y = U[:T], Y[:T]

    dt = float(np.median(np.diff(etp.cxa.pv2['relative_time'])))   # real frame interval
    mask = tracking_mask(Y, U, thresh_deg=45.0)                    # epochs FC2 follows hDeltaC

    res = fit_ring(U, Y, mask=mask,
                   free=('J0','w_inh','gain','tau'),
                   fixed=dict(kappa_exc=10.0, I_scale=1.0, dt=dt),  # pin the sloppy ones
                   maxiter=60, popsize=15, seed=0, workers=-1)      # workers=-1: all cores
    print(res['params'], 'loss', res['loss'])
    print('effective a =', a_from_tau(res['params']['tau'], dt))

    # check identifiability of anything you report:
    import numpy as np
    gridw = np.linspace(0, 8, 11)
    Lw = profile_param('w_inh', gridw, U, Y, mask=mask,
                       fixed=dict(kappa_exc=10.0, I_scale=1.0, dt=dt))
    plt.figure(); plt.plot(gridw, Lw/Lw.min(), 'o-'); plt.xlabel('w_inh'); plt.ylabel('loss/min')
    # flat curve => not constrained by your data; clear minimum => constrained


#%% Fit core params to hDeltaC -> FC2 on tracking epochs
U = np.asarray(etp.cxa.pdat['wedges_fsb2_ch2'], float)   # hDeltaC input
Y = np.asarray(etp.cxa.pdat['wedges_fsb1_ch1'], float)   # FC2 target
T = min(len(U), len(Y)); U, Y = U[:T], Y[:T]

dt = float(np.median(np.diff(etp.cxa.pv2['relative_time'])))   # real frame interval
mask = tracking_mask(Y, U, thresh_deg=45.0)                    # epochs FC2 follows hDeltaC

res = fit_ring(U, Y, mask=mask,
               free=('J0','w_inh','gain','tau'),
               fixed=dict(kappa_exc=10.0, I_scale=1.0, dt=dt),  # pin the sloppy ones
               maxiter=60, popsize=15, seed=0, workers=1)      # workers=-1: all cores
print(res['params'], 'loss', res['loss'])
print('effective a =', a_from_tau(res['params']['tau'], dt))

# check identifiability of anything you report:
import numpy as np
gridw = np.linspace(0, 8, 11)
Lw = profile_param('w_inh', gridw, U, Y, mask=mask,
                   fixed=dict(kappa_exc=10.0, I_scale=1.0, dt=dt))
plt.figure(); plt.plot(gridw, Lw/Lw.min(), 'o-'); plt.xlabel('w_inh'); plt.ylabel('loss/min')
# flat curve => not constrained by your data; clear minimum => constrained



#%%
ins = etp.cxa.ft2['instrip'].to_numpy().astype(float)[:,np.newaxis]
u = ug()
_,_,dd = u.get_velocity(etp.cxa.ft2['ft_posx'].to_numpy(),etp.cxa.ft2['ft_posy'].to_numpy(),etp.cxa.pv2['relative_time'].to_numpy())

net = ring_attractor(kernel='connectome',w_inh=2,kappa_exc=10,gain=.2,hDelta_exc=.15,dt=.1,tau=.05,J0=5.5)

# Model with hdeltaC input only
r,rt = net.run_cont_inputs(etp.cxa.pdat['wedges_fsb2_ch2'],T=1500)
rlen = len(rt)
at_phase = ug.phase_from_wed(rt)
atphase_plot = 32+8*(at_phase+np.pi)/np.pi
fc2phase_plot = 32+8*(etp.cxa.pdat['phase_fsb1_ch1'][:rlen]+np.pi)/np.pi
hdcphase_plot = 32+8*(etp.cxa.pdat['phase_fsb2_ch2'][:rlen]+np.pi)/np.pi
ebphase_plot = 32+8*(etp.cxa.pdat['phase_eb_ch1'][:rlen]+np.pi)/np.pi
pltmat = np.append(rt,etp.cxa.pdat['wedges_fsb2_ch2'][:rlen,:],axis=1)
pltmat = np.append(pltmat,etp.cxa.pdat['wedges_fsb1_ch1'][:rlen,:],axis=1)

plt.imshow(pltmat,aspect='auto',interpolation='none',vmin=0,vmax=1,cmap='Blues')
y = np.arange(0,rlen)
plt.scatter(atphase_plot,y,color='r',s=5)
plt.scatter(fc2phase_plot,y,color='m',s=5)
plt.scatter(hdcphase_plot,y,color='g',s=5)
plt.scatter(ebphase_plot,y,color='k',s=5)
plt.plot(ins[:rlen]*10+50,y,color='r')
plt.plot(dd[:rlen]+60,y,color='k')

# Model with hdeltaC + heading
balance = 0.75
t_input = etp.cxa.pdat['wedges_fsb2_ch2']*balance+etp.cxa.pdat['wedges_eb_ch1']*(1-balance)
t_input = etp.cxa.pdat['wedges_fsb2_ch2']*(ins-1)*-1+etp.cxa.pdat['wedges_eb_ch1']*ins


r,rt = net.run_cont_inputs(t_input,T=1500)
rlen = len(rt)
at_phase = ug.phase_from_wed(rt)
atphase_plot = 32+8*(at_phase+np.pi)/np.pi
fc2phase_plot = 32+8*(etp.cxa.pdat['phase_fsb1_ch1'][:rlen]+np.pi)/np.pi
hdcphase_plot = 32+8*(etp.cxa.pdat['phase_fsb2_ch2'][:rlen]+np.pi)/np.pi
pltmat = np.append(rt,etp.cxa.pdat['wedges_fsb2_ch2'][:rlen,:],axis=1)
pltmat = np.append(pltmat,etp.cxa.pdat['wedges_fsb1_ch1'][:rlen,:],axis=1)
plt.figure()
plt.imshow(pltmat,aspect='auto',interpolation='none',vmin=0,vmax=1,cmap='Blues')
y = np.arange(0,rlen)
plt.scatter(atphase_plot,y,color='r',s=5)
plt.scatter(fc2phase_plot,y,color='m',s=5)
plt.scatter(hdcphase_plot,y,color='g',s=5)
plt.scatter(ebphase_plot,y,color='k',s=5)

plt.plot(ins[:rlen]*10+50,y,color='r')

# Add in hDeltaJ kernel - think about what that could becell_type == FB5A
#%% Model balance between FC2 signals
# from modelling.fit_balance import fit_balance, jax_ring_step

# J = net.J
# w_inh = net.w_inh
# tau = net.tau
# dt = net.dt
# r_max = net.r_max
# gain = net.gain
# step = jax_ring_step(J, w_inh, tau, dt, r_max, gain)
# Y = etp.cxa.pdat['wedges_fsb1_ch1']
# inpA = etp.cxa.pdat['wedges_fsb2_ch2']
# inpB = etp.cxa.pdat['wedges_eb_ch1']
# I_amp = 5
# dx = np.arange(0,8000)

# res = fit_balance(
#     step, Y[dx], inpA[dx,:], inpB[dx,:], 1,
#     spf=5,        # sim steps per imaging frame (match your dt-per-frame)
#     window=50,      # multiple-shooting horizon (see below)
#     drift_std=0.15, # how fast b may change per frame (smoothness prior)
# )

# ins = etp.cxa.ft2['instrip'].to_numpy()
# b = res['b']
# fc2 = etp.cxa.pdat['phase_fsb1_ch1']/np.pi
# hdc = etp.cxa.pdat['phase_fsb2_ch2']/np.pi
# eb = etp.cxa.pdat['phase_eb_ch1']/np.pi
# t = np.arange(0,len(fc2))/10 
# plt.plot(t[dx],ins[dx]-2,color='r')
# plt.plot(t[dx[:-1]],res['b']-2,color='k')
# plt.scatter(t[dx],fc2[dx],color='b',s=3)
# plt.scatter(t[dx],hdc[dx],color='g',s=3)
# plt.scatter(t[dx],eb[dx],color='k',s=3)


