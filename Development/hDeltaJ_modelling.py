# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 12:16:28 2025

@author: dowel
"""

from analysis_funs.regression import fci_regmodel

import numpy as np
import pandas as pd
import src.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt 
from analysis_funs.utilities import imaging as im
from scipy import signal as sg
from analysis_funs.CX_imaging import CX
from analysis_funs.CX_analysis_col import CX_a
from src.utilities import funcs as fn
from scipy import stats
from Utilities.utils_general import utils_general as ug
from scipy.optimize import minimize

plt.rcParams['pdf.fonttype'] = 42 
#%% Load example data
datadir =r"Y:\Data\FCI\Hedwig\hDeltaJ\251011\f1\Trial1"
#datadir =r"Y:\Data\FCI\Hedwig\hDeltaJ\251028\f1\Trial2"
#datadir= r"Y:\Data\FCI\Hedwig\hDeltaK_SS63089\250807\f2\Trial4"
datadir = r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f2\Trial2"
cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
plt.close('all')
cxa.simple_raw_plot(plotphase=True,regions = ['eb','fsb_upper'])
cxa.plot_traj_arrow(cxa.pdat['offset_fsb_upper_phase'],np.mean(cxa.pdat['wedges_fsb_upper']/4,axis=1),a_sep= 4)

#%% Model 1 - fit offset
class model1:
    def __init__(self):
        self.version=1
    def predict_phase(self,phase,phase_eb,offset):
        phase_pred = ug.circ_subtract(phase_eb,offset)
        return phase_pred
    
    def fit_model(self,phase,phase_eb):
        offset = 0
        bounds = [(-np.pi,np.pi)]
        def objective(offset):
            phase_pred = self.predict_phase(phase,phase_eb,offset)
            return ug.circ_mse(phase,phase_pred)
        self.res = minimize(objective,offset,bounds=bounds,method='L-BFGS-B')
phase = cxa.pdat['phase_fsb_upper']
phase_eb = cxa.pdat['phase_eb']    
mdl1 = model1()
mdl1.fit_model(phase,phase_eb)
x = np.arange(0,len(phase))
phase_pred = mdl1.predict_phase(phase,phase_eb,mdl1.res.x)

plt.scatter(x,phase,color='k',s=2)


plt.scatter(x,phase_pred,color='r',s=2)

print(180*(mdl1.res.x/np.pi)) 
mse = ug.circ_mse(phase,phase_pred)
print('MSE: ',mse)   


#%% Model 2 - offset and delay
class model2:
    def __init__(self):
        self.version=1
    def predict_phase(self,phase,phase_eb,offset):
        phase_pred = ug.circ_subtract(phase_eb,offset)
        return phase_pred
    
    def fit_model(self,phase,phase_eb,delay_range):
        offset = 0
        bounds = [(-np.pi,np.pi)]
        mses = np.zeros(delay_range-1)
        models = {}
        for i in np.arange(1,delay_range,dtype=int):
            
            def objective(offset):
                phase_pred = self.predict_phase(phase[i:],phase_eb[:-i],offset)
                return ug.circ_mse(phase[i:],phase_pred)
            models.update({str(i): minimize(objective,offset,bounds=bounds,method='L-BFGS-B')})
            mses[i-1] = models[str(i)].fun
            
        best = np.argmin(mses)+1
        self.mses = mses
        self.res = models[str(best)]
        self.delay = best
phase = cxa.pdat['phase_fsb_upper']
phase_eb = cxa.pdat['phase_eb']    
mdl2 = model2()
mdl2.fit_model(phase,phase_eb,50)
x = np.arange(0,len(phase))
print(mdl2.delay)
phase_pred = mdl2.predict_phase(phase[mdl2.delay:],phase_eb[:-mdl2.delay],mdl2.res.x)

plt.scatter(x,phase,color='k',s=2)


plt.scatter(x[mdl2.delay:],phase_pred,color='r',s=2)

print(180*(mdl2.res.x/np.pi)) 
mse = ug.circ_mse(phase[mdl2.delay:],phase_pred)
print('MSE: ',mse)   
#%% Model 3 - offset and delay in versus out strip
class model3:
    def __init__(self):
        self.version=1
    def predict_phase(self,phase,phase_eb,offset_in,offset_out,ins):
        phase_pred = np.zeros_like(phase)
        phase_pred[ins>0] = ug.circ_subtract(phase_eb[ins>0],offset_in)
        phase_pred[ins<1] = ug.circ_subtract(phase_eb[ins<1],offset_out)

        return phase_pred
    
    def fit_model(self,phase,phase_eb,ins,delay_range):
        offsets = [0,0]
        bounds = [(-np.pi,np.pi),(-np.pi,np.pi)]
        mses = np.zeros(delay_range-1)
        models = {}
        for i in np.arange(1,delay_range,dtype=int):
            
            def objective(offsets):
                phase_pred = self.predict_phase(phase[i:],phase_eb[:-i],offsets[0],offsets[1],ins[i:])
                return ug.circ_mse(phase[i:],phase_pred)
            models.update({str(i): minimize(objective,offsets,bounds=bounds,method='L-BFGS-B')})
            mses[i-1] = models[str(i)].fun
            
        best = np.argmin(mses)+1
        self.mses = mses
        self.res = models[str(best)]
        self.delay = best
        
phase = cxa.pdat['phase_fsb_upper']
phase_eb = cxa.pdat['phase_eb']    
ins = cxa.ft2['instrip'].to_numpy()
u = ug()
vx,vy,vd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'])
vd = vd/np.std(vd)
mdl3 = model3()
mdl3.fit_model(phase,phase_eb,ins,30)
x = np.arange(0,len(phase))
print(mdl3.delay)
phase_pred = mdl3.predict_phase(phase[mdl3.delay:],phase_eb[:-mdl3.delay],mdl3.res.x[0],mdl3.res.x[1],ins[mdl3.delay:])
phase_pred3 = phase_pred.copy()

plt.scatter(x,phase,color='k',s=2)


plt.scatter(x[mdl3.delay:],phase_pred,color='r',s=2)

print(180*(mdl3.res.x/np.pi)) 
mse = ug.circ_mse(phase[mdl3.delay:],phase_pred)
print('MSE: ',mse)   

plt.plot(x,ins,color='b')
plt.plot(x[1:],vd/2-4,color=[0.5,0.5,0.5])
#%% Model 3.5 - pre air offset also
class model3_2:
    def __init__(self):
        self.version=1
    def predict_phase(self,phase,phase_eb,offset_pre,offset_in,offset_out,ins):
        phase_pred = np.zeros_like(phase)
        phase_pred[ins>0] = ug.circ_subtract(phase_eb[ins>0],offset_in)
        phase_pred[ins<1] = ug.circ_subtract(phase_eb[ins<1],offset_out*np.sign(phase_eb[ins<1]))
        first = np.where(ins>0)[0][0]
        phase_pred[:first] = ug.circ_subtract(phase_eb[:first],offset_pre)
        return phase_pred
    
    def fit_model(self,phase,phase_eb,ins,delay_range):
        offsets = [0,0,0]
        bounds = [(-np.pi,np.pi),(-np.pi,np.pi),(-np.pi,np.pi)]
        mses = np.zeros(delay_range-1)
        models = {}
        for i in np.arange(1,delay_range,dtype=int):
            print('Delay: ',i)
            def objective(offsets):
                phase_pred = self.predict_phase(phase[i:],phase_eb[:-i],offsets[0],offsets[1],offsets[2],ins[i:])
                return ug.circ_mse(phase[i:],phase_pred)
            models.update({str(i): minimize(objective,offsets,bounds=bounds,method='L-BFGS-B')})
            mses[i-1] = models[str(i)].fun
            
        best = np.argmin(mses)+1
        self.mses = mses
        self.res = models[str(best)]
        self.delay = best
        
phase = cxa.pdat['phase_fsb_upper']
phase_eb = cxa.pdat['phase_eb']    
ins = cxa.ft2['instrip'].to_numpy()
u = ug()
vx,vy,vd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'])
vd = vd/np.std(vd)
mdl3_2 = model3_2()
mdl3_2.fit_model(phase,phase_eb,ins,30)
x = np.arange(0,len(phase))
print(mdl3_2.delay)
phase_pred = mdl3_2.predict_phase(phase[mdl3_2.delay:],phase_eb[:-mdl3_2.delay],mdl3_2.res.x[0],mdl3_2.res.x[1],mdl3_2.res.x[2],ins[mdl3_2.delay:])
phase_pred3 = phase_pred.copy()

plt.scatter(x,phase,color='k',s=2)


plt.scatter(x[mdl3_2.delay:],phase_pred,color='r',s=2)

print(180*(mdl3_2.res.x/np.pi)) 
mse = ug.circ_mse(phase[mdl3_2.delay:],phase_pred)
print('MSE: ',mse)   

plt.plot(x,ins,color='b')
plt.plot(x[1:],vd/2-4,color=[0.5,0.5,0.5])

#%% Model 4 - Update model
class model4:
    def __init__(self):
        self.version=1
    def predict_phase(self,phase,phase_eb,offset,w):
        phase_offset = ug.circ_subtract(phase_eb,offset)
        phase_pred = np.zeros_like(phase)
        phase_pred[0] = phase_offset[0]
        for i in range(len(phase_pred)-1):
            delta = ug.circ_subtract(phase_pred[i],phase_offset[i])
            phase_pred[i+1] = ug.circ_subtract(phase_pred[i],delta*w)
        

        return phase_pred
    
    def fit_model(self,phase,phase_eb,delay_range):
        params = [.1]
        bounds = [(-5,5)]
        mses = np.zeros(delay_range-1)
        models = {}
        for i in np.arange(1,delay_range,dtype=int):
            
            def objective(params):
                phase_pred = self.predict_phase(phase[i:],phase_eb[:-i],np.pi,params[0])
                return ug.circ_mse(phase[i:],phase_pred)
            models.update({str(i): minimize(objective,params,bounds=bounds,method='L-BFGS-B')})
            mses[i-1] = models[str(i)].fun
            
        best = np.argmin(mses)+1
        self.mses = mses
        self.res = models[str(best)]
        self.delay = best

phase = cxa.pdat['phase_fsb_upper']
phase_eb = cxa.pdat['phase_eb']    
vx,vy,vd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'])
vperc = np.percentile(vd,[10,95])
vd[vd<.5] = 0
vd = vd/np.std(vd)

mdl4 = model4()
mdl4.fit_model(phase,phase_eb,15)


phase_pred = mdl4.predict_phase(phase[mdl4.delay:],phase_eb[:-mdl4.delay],np.pi
                                ,mdl4.res.x[0])
plt.scatter(x,phase,color='k',s=2)


plt.scatter(x[mdl4.delay:],phase_pred,color='r',s=2)

mse = ug.circ_mse(phase[mdl4.delay:],phase_pred)
print('MSE: ',mse)   

plt.plot(x,ins,color='b')
plt.plot(x[1:],vd/2-4,color=[0.5,0.5,0.5])

angvel = ug.get_ang_velocity(cxa.ft2['ft_heading'].to_numpy(),cxa.pv2['relative_time'])
angvel = angvel/np.std(angvel)
plt.plot(x[1:],angvel/10 -5,color='k')
plt.scatter(x[mdl3.delay:],phase_pred3,color='g',s=2)


#%% Model 5 - velocity update model - this works very well for hDeltaK
class model5:
    def __init__(self):
        self.version=1
    def predict_phase(self,phase,phase_eb,offset,w,velocity):
        phase_offset = ug.circ_subtract(phase_eb,offset)
        phase_pred = np.zeros_like(phase)
        phase_pred[0] = phase_offset[0]
        for i in range(len(phase_pred)-1):
            delta = ug.circ_subtract(phase_pred[i],phase_offset[i])
            phase_pred[i+1] = ug.circ_subtract(phase_pred[i],delta*w*velocity[i])
        

        return phase_pred
    
    def fit_model(self,phase,phase_eb,delay_range,velocity):
        params = [.1]
        bounds = [(-5,5)]
        mses = np.zeros(delay_range-1)
        models = {}
        for i in np.arange(1,delay_range,dtype=int):
            
            def objective(params):
                phase_pred = self.predict_phase(phase[i:],phase_eb[:-i],np.pi,params[0],velocity[:-i])
                return ug.circ_mse(phase[i:],phase_pred)
            models.update({str(i): minimize(objective,params,bounds=bounds,method='L-BFGS-B')})
            mses[i-1] = models[str(i)].fun
            
        best = np.argmin(mses)+1
        self.mses = mses
        self.res = models[str(best)]
        self.delay = best
        self.models =models
phase = cxa.pdat['phase_fsb_upper']
phase_eb = cxa.pdat['phase_eb']    
vx,vy,vd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'])
vperc = np.percentile(vd,[10,95])
vd[vd<.5] = 0
vd = vd/np.std(vd)

mdl5 = model5()
mdl5.fit_model(phase,phase_eb,15,vd)


phase_pred = mdl5.predict_phase(phase[mdl5.delay:],phase_eb[:-mdl5.delay],np.pi
                                ,mdl5.res.x[0],vd[mdl5.delay:])
plt.scatter(x,phase,color='k',s=2)


plt.scatter(x[mdl5.delay:],phase_pred,color='r',s=2)

mse = ug.circ_mse(phase[mdl5.delay:],phase_pred)
print('MSE: ',mse)   

plt.plot(x,ins,color='b')
plt.plot(x[1:],vd/2-4,color=[0.5,0.5,0.5])

angvel = ug.get_ang_velocity(cxa.ft2['ft_heading'].to_numpy(),cxa.pv2['relative_time'])
angvel = angvel/np.std(angvel)
plt.plot(x[1:],angvel/10 -5,color='k')
plt.scatter(x[mdl3.delay:],phase_pred3,color='g',s=2)
#%% Model 5 plus
class model5_2:
    def __init__(self):
        self.version=1
        
    def predict_phase(self,phase,phase_eb,offset_in,offset_out,w_in,w_out,velocity):
        phase_offset_in = ug.circ_subtract(phase_eb,offset_in)
        phase_offset_out = ug.circ_subtract(phase_eb,offset_out)
        phase_pred = np.zeros_like(phase)
        phase_pred[0] = phase_offset_out[0]
        for i in range(len(phase_pred)-1):
            delta_in = ug.circ_subtract(phase_pred[i],phase_offset_in[i])
            delta_out = ug.circ_subtract(phase_pred[i],phase_offset_out[i])
            delta_weighted = ins[i]*delta_in*w_in + (1-ins[i])*delta_out*w_out
            
            
            phase_pred[i+1] = ug.circ_subtract(phase_pred[i],delta_weighted)
        

        return phase_pred
    
    def fit_model(self,phase,phase_eb,delay_range,velocity):
        params = [0,.1,.05]
        bounds = [(-np.pi,np.pi),(-.5,.5),(-.5,.5)]
        mses = np.zeros(delay_range-1)
        models = {}
        for i in np.arange(1,delay_range,dtype=int):
            print('Delay: ',i)
            def objective(params):
                phase_pred = self.predict_phase(phase[i:],phase_eb[:-i],params[0],2.79556692,params[1],params[2],velocity[:-i])
                return ug.circ_mse(phase[i:],phase_pred)
            models.update({str(i): minimize(objective,params,bounds=bounds,method='L-BFGS-B')})
            mses[i-1] = models[str(i)].fun
            
        best = np.argmin(mses)+1
        self.mses = mses
        self.res = models[str(best)]
        self.delay = best
        self.models =models
        
    def fit_downsample(self,phase,phase_eb,delay_range,velocity,dfactor=2):
        dx = np.arange(0,len(phase),step=dfactor,dtype='int') # downsample without interpolation, fine for most intervals
        phase = phase[dx]
        phase_eb = phase_eb[dx]
        delay_range = int(delay_range/dfactor)
        velocity = velocity[dx]
        
        self.fit_model(phase,phase_eb,delay_range,velocity)
        self.delay = int(self.delay*dfactor)
        
phase = cxa.pdat['phase_fsb_upper']
phase_eb = cxa.pdat['phase_eb']    
vx,vy,vd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'])
vperc = np.percentile(vd,[10,95])
vd[vd<.5] = 0
vd = vd/np.std(vd)

mdl5_2 = model5_2()
mdl5_2.fit_model(phase,phase_eb,5,vd)


phase_pred = mdl5_2.predict_phase(phase[mdl5_2.delay:],phase_eb[:-mdl5_2.delay],mdl5_2.res.x[0],np.pi
                                ,mdl5_2.res.x[1],mdl5_2.res.x[2],vd[:-mdl5_2.delay])
plt.scatter(x,phase,color='k',s=2)


plt.scatter(x[mdl5_2.delay:],phase_pred,color='r',s=2)

mse = ug.circ_mse(phase[mdl5_2.delay:],phase_pred)
print('MSE: ',mse)   

plt.plot(x,ins,color='b')
plt.plot(x[1:],vd/2-4,color=[0.5,0.5,0.5])

angvel = ug.get_ang_velocity(cxa.ft2['ft_heading'].to_numpy(),cxa.pv2['relative_time'])
angvel = angvel/np.std(angvel)
plt.plot(x[1:],angvel/10 -5,color='k')
plt.scatter(x[mdl3.delay:],phase_pred3,color='g',s=2)

#%% Model 5 plus plus, has pre plume, in plume and out plume
class model5_3:
    def __init__(self):
        self.version=1
        
    def predict_phase(self,phase,phase_eb,offset_pre,offset_in,offset_out,w_pre,w_in,w_out,velocity,ins):
        phase_offset_in = ug.circ_subtract(phase_eb,offset_in)
        phase_offset_out = ug.circ_subtract(phase_eb,offset_out)
        phase_offset_pre = ug.circ_subtract(phase_eb,offset_pre)
        phase_pred = np.zeros_like(phase)
        phase_pred[0] = phase_offset_out[0]
        first  = np.where(ins>0)[0][0]
        for i in range(len(phase_pred)-1):
            if i<first:
                delta_pre = ug.circ_subtract(phase_pred[i],phase_offset_pre[i])
                delta_weighted = delta_pre*w_pre
            else:
                delta_in = ug.circ_subtract(phase_pred[i],phase_offset_in[i])
                delta_out = ug.circ_subtract(phase_pred[i],phase_offset_out[i])
                delta_weighted = ins[i]*delta_in*w_in + (1-ins[i])*delta_out*w_out
            
            
            phase_pred[i+1] = ug.circ_subtract(phase_pred[i],delta_weighted)
        
        return phase_pred
    
    def fit_model(self,phase,phase_eb,delay_range,velocity,ins):
        params = [np.pi,0,np.pi,.1,.05,.05]
        bounds = [(-np.pi,np.pi),(-np.pi,np.pi),(-np.pi,np.pi),(-.5,.5),(-.5,.5),(-.5,.5)]
        mses = np.zeros(delay_range-1)
        models = {}
        for i in np.arange(1,delay_range,dtype=int):
            print('Delay: ',i)
            def objective(params):
                phase_pred = self.predict_phase(phase[i:],phase_eb[:-i],params[0],params[1],
                                                params[2],params[3],params[4],params[5],velocity[:-i],ins[:-i])
                return ug.circ_mse(phase[i:],phase_pred)
            models.update({str(i): minimize(objective,params,bounds=bounds,method='L-BFGS-B')})
            mses[i-1] = models[str(i)].fun
            
        best = np.argmin(mses)+1
        self.mses = mses
        self.res = models[str(best)]
        self.delay = best
        self.models =models
        
    def fit_downsample(self,phase,phase_eb,delay_range,velocity,dfactor=2):
        dx = np.arange(0,len(phase),step=dfactor,dtype='int') # downsample without interpolation, fine for most intervals
        phase = phase[dx]
        phase_eb = phase_eb[dx]
        delay_range = int(delay_range/dfactor)
        velocity = velocity[dx]
        
        self.fit_model(phase,phase_eb,delay_range,velocity)
        self.delay = int(self.delay*dfactor)
        
phase = cxa.pdat['phase_fsb_upper']
phase_eb = cxa.pdat['phase_eb']    
vx,vy,vd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'])
vperc = np.percentile(vd,[10,95])
vd[vd<.5] = 0
vd = vd/np.std(vd)

mdl5_3 = model5_3()
mdl5_3.fit_model(phase,phase_eb,5,vd,ins)


phase_pred = mdl5_3.predict_phase(phase[mdl5_3.delay:],phase_eb[:-mdl5_3.delay],mdl5_3.res.x[0],mdl5_3.res.x[1]
                                ,mdl5_3.res.x[2],mdl5_3.res.x[3],mdl5_3.res.x[4],mdl5_3.res.x[5],vd[:-mdl5_3.delay],ins[:-mdl5_3.delay])
plt.scatter(x,phase,color='k',s=2)


plt.scatter(x[mdl5_3.delay:],phase_pred,color='r',s=2)

mse = ug.circ_mse(phase[mdl5_3.delay:],phase_pred)
print('MSE: ',mse)   

plt.plot(x,ins,color='b')
plt.plot(x[1:],vd/2-4,color=[0.5,0.5,0.5])

angvel = ug.get_ang_velocity(cxa.ft2['ft_heading'].to_numpy(),cxa.pv2['relative_time'])
angvel = angvel/np.std(angvel)
plt.plot(x[1:],angvel/10 -5,color='k')
plt.scatter(x[mdl3_2.delay:],phase_pred3,color='g',s=2)

#%%
#%% Model 5 plus plus parallel, has pre plume, in plume and out plume
from joblib import Parallel, delayed
class model5_3p:
    def __init__(self):
        self.version=1
        
    def predict_phase(self,phase,phase_eb,offset_pre,offset_in,offset_out,w_pre,w_in,w_out,gamma,velocity,ins):
        phase_offset_in = ug.circ_subtract(phase_eb,offset_in)
        phase_offset_out = ug.circ_subtract(phase_eb,offset_out)
        phase_offset_pre = ug.circ_subtract(phase_eb,offset_pre)
        phase_pred = np.zeros_like(phase)
        phase_pred[0] = phase_offset_out[0]
        first  = np.where(ins>0)[0][0]
        for i in range(len(phase_pred)-1):
            if i<first:
                delta_pre = ug.circ_subtract(phase_pred[i],phase_offset_pre[i])
                delta_weighted = delta_pre*w_pre*velocity[i]
            else:
                delta_in = ug.circ_subtract(phase_pred[i],phase_offset_in[i])
                delta_out = ug.circ_subtract(phase_pred[i],phase_offset_out[i])
                delta_weighted = (ins[i]*delta_in*w_in + (1-ins[i])*delta_out*w_out)*velocity[i]**gamma
            
            
            phase_pred[i+1] = ug.circ_subtract(phase_pred[i],delta_weighted)
        
        return phase_pred
    
    def fit_model(self,phase,phase_eb,delay_range,velocity,ins):
        params = [np.pi,0,np.pi,.1,.05,.05,1]
        bounds = [(-np.pi,np.pi),(-np.pi,np.pi),(-np.pi,np.pi),(-.5,.5),(-.5,.5),(-.5,.5),(-2,2)]
        mses = np.zeros(delay_range-1)
        models = {}
        
        def fit_delay(i):
            np.random.seed(i)
            # phase_t = phase[i:].copy()
            # phase_eb_t = phase[:-i].copy()
            # vel_t = velocity[:-i].copy()
            # ins_t = ins[:-i].copy()
            
            def objective(params):
                phase_pred = self.predict_phase(phase[i:],phase_eb[:-i],params[0],params[1],
                                                         params[2],params[3],params[4],params[5],params[6],velocity[:-i],ins[:-i])
                return ug.circ_mse(phase[i:],phase_pred)
        
        
            return i,minimize(objective,params,bounds=bounds,method='L-BFGS-B')
        
        results = Parallel(n_jobs=-1,backend='loky')(delayed(fit_delay)(i) for i in range(1,delay_range))
        
        self.models = {str(i): res for i, res in results}
        self.mses = np.array([res.fun for _, res in results])
        self.delay = np.argmin(self.mses) + 1
        self.res = self.models[str(self.delay)]
        
        
        
        # for i in np.arange(1,delay_range,dtype=int):
        #     print('Delay: ',i)
        #     def objective(params):
        #         phase_pred = self.predict_phase(phase[i:],phase_eb[:-i],params[0],params[1],
        #                                         2.79556692,params[2],params[3],params[4],velocity[:-i])
        #         return ug.circ_mse(phase[i:],phase_pred)
        #     models.update({str(i): minimize(objective,params,bounds=bounds,method='L-BFGS-B')})
        #     mses[i-1] = models[str(i)].fun
            
        # best = np.argmin(mses)+1
        # self.mses = mses
        # self.res = models[str(best)]
        # self.delay = best
        # self.models =models
        
    def fit_downsample(self,phase,phase_eb,delay_range,velocity,dfactor=2):
        dx = np.arange(0,len(phase),step=dfactor,dtype='int') # downsample without interpolation, fine for most intervals
        phase = phase[dx]
        phase_eb = phase_eb[dx]
        delay_range = int(delay_range/dfactor)
        velocity = velocity[dx]
        
        self.fit_model(phase,phase_eb,delay_range,velocity)
        self.delay = int(self.delay*dfactor)
        
phase = cxa.pdat['phase_fsb_upper']
ins = cxa.ft2['instrip'].to_numpy()
phase_eb = cxa.pdat['phase_eb']    
vx,vy,vd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'])
vperc = np.percentile(vd,[10,95])
vd[vd<.5] = 0
vd = vd/np.std(vd)
x= np.arange(0,len(phase))
mdl5_3p = model5_3p()
mdl5_3p.fit_model(phase,phase_eb,15,vd,ins)

phase_pred = mdl5_3p.predict_phase(phase[mdl5_3p.delay:],phase_eb[:-mdl5_3p.delay],mdl5_3p.res.x[0],mdl5_3p.res.x[1]
                                ,mdl5_3p.res.x[2],mdl5_3p.res.x[3],mdl5_3p.res.x[4],mdl5_3p.res.x[5],mdl5_3p.res.x[6],vd[:-mdl5_3p.delay],ins[:-mdl5_3p.delay])
plt.scatter(x,phase,color='k',s=2)


plt.scatter(x[mdl5_3p.delay:],phase_pred,color='r',s=2)

mse = ug.circ_mse(phase[mdl5_3p.delay:],phase_pred)
print('MSE: ',mse)   

plt.plot(x,ins,color='b')
plt.plot(x[1:],vd/2-4,color=[0.5,0.5,0.5])

angvel = ug.get_ang_velocity(cxa.ft2['ft_heading'].to_numpy(),cxa.pv2['relative_time'])
angvel = angvel/np.std(angvel)
plt.plot(x[1:],angvel/10 -5,color='k')
plt.scatter(x[mdl3_2.delay:],phase_pred3,color='g',s=2)
#%%
phase_pred = mdl5_3.predict_phase(phase[mdl5_3.delay:],phase_eb[:-mdl5_3.delay],mdl5_3.res.x[0],mdl5_3.res.x[1],2.79556692
                                ,mdl5_3.res.x[2],mdl5_3.res.x[3],mdl5_3.res.x[4],vd[:-mdl5_3.delay])
plt.scatter(x,phase,color='k',s=2)


plt.scatter(x[mdl5_3.delay:],phase_pred,color='r',s=2)

mse = ug.circ_mse(phase[mdl5_3.delay:],phase_pred)
print('MSE: ',mse)   

plt.plot(x,ins,color='b')
plt.plot(x[1:],vd/2-4,color=[0.5,0.5,0.5])

angvel = ug.get_ang_velocity(cxa.ft2['ft_heading'].to_numpy(),cxa.pv2['relative_time'])
angvel = angvel/np.std(angvel)
plt.plot(x[1:],angvel/10 -5,color='k')
plt.scatter(x[mdl3_2.delay:],phase_pred3,color='g',s=2)

#%% Model 6 - in plume and out plume update 

# Improve by adding delay for in odour

class model6:
    def __init__(self):
        self.version=1
    def predict_phase(self,phase,phase_eb,offset_in,offset_out,w_in,w_out,velocity,sup,ins):
        phase_offset_in = ug.circ_subtract(phase_eb,offset_in)
        phase_offset_out  = ug.circ_subtract(phase_eb,offset_out)
        

        phase_pred = np.zeros_like(phase)
        phase_pred[0] = phase_offset_out[0]
        velocity = velocity**sup
        p_in = 1/(1+np.exp(-10*ins))
        
        
        for i in range(len(phase_pred)-1):
            delta_in =ug.circ_subtract(phase_pred[i],phase_offset_in[i])
            delta_out = ug.circ_subtract(phase_pred[i],phase_offset_out[i])
            phase_pred[i+1] = ug.circ_subtract(phase_pred[i],velocity[i]*(p_in[i]*delta_in*w_in + (1-p_in[i])*delta_out*w_out))
            
            
            # if ins[i]>0:
            #     delta = ug.circ_subtract(phase_pred[i],phase_offset_in[i])
            #     phase_pred[i+1] = ug.circ_subtract(phase_pred[i],delta*w_in*velocity[i])
            # else:
            #     delta = ug.circ_subtract(phase_pred[i],phase_offset_out[i])
            #     phase_pred[i+1] = ug.circ_subtract(phase_pred[i],delta*w_out*velocity[i])

        return phase_pred
    
    def fit_model(self,phase,phase_eb,delay_range,velocity,ins):
        params = [.1,.1,1,0,np.pi]
        #params = [-9.616e-02,  1.468e-01,  1.008e+00,  2.521e-03,  3.137e+00]
        bounds = [(-1,1),(-1,1),(0.1,3),(-np.pi,np.pi),(-np.pi,np.pi)]
        mses = np.zeros(delay_range-1)
        models = {}
        for i in np.arange(1,delay_range,dtype=int):
            
            def objective(params):
                phase_pred = self.predict_phase(phase[i:],phase_eb[:-i],params[3],params[4],params[0],params[1],velocity[:-i],params[2],ins[:-i])
                return ug.circ_mse(phase[i:],phase_pred)
            models.update({str(i): minimize(objective,params,bounds=bounds,method='L-BFGS-B')})
            mses[i-1] = models[str(i)].fun
            
        best = np.argmin(mses)+1
        self.mses = mses
        self.res = models[str(best)]
        self.delay = best
        self.models =models
mdl6 = model6()
phase = cxa.pdat['phase_fsb_upper']
phase_eb = cxa.pdat['phase_eb']  
ins = cxa.ft2['instrip'].to_numpy().astype('float')
vx,vy,vd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'])

vd = vd/np.std(vd)
mdl6.fit_model(phase,phase_eb,7,vd,ins)
phase_pred = mdl6.predict_phase(phase[mdl6.delay:],phase_eb[:-mdl6.delay],mdl6.res.x[3],mdl6.res.x[4],
                                mdl6.res.x[0],mdl6.res.x[1],vd[:-mdl5.delay],mdl6.res.x[2],ins[:-mdl5.delay])

x = np.arange(0,len(phase))
plt.scatter(x,phase,color='k',s=2)

plt.scatter(x[mdl6.delay:],phase_pred,color='r',s=2)

mse = ug.circ_mse(phase[mdl6.delay:],phase_pred)
print('MSE: ',mse)   

plt.plot(x,ins,color='b')
plt.plot(x[1:],vd/2-4,color=[0.5,0.5,0.5])
plt.scatter(x[mdl3.delay:],phase_pred3,color='g',s=2)
#%% model 7 a kernel model
class model7:
    def __init__(self):
        self.version=1
    def predict_phase(self,phase_eb,offset,mu,sigma,w,winlen=100):
        
        phase_eb_o = ug.circ_subtract(phase_eb,offset)
        psin = np.sin(phase_eb_o)
        pcos= np.cos(phase_eb_o)
        kernel = stats.norm.pdf(np.arange(0,winlen),loc=mu,scale=sigma)*w
        psin_c = np.convolve(psin,kernel,mode='full')
        pcos_p = np.convolve(pcos,kernel,mode='full')
        phase_pred = np.arctan2(psin_c,pcos_p)
        phase_pred = phase_pred[winlen:-winlen+1]
        
        return phase_pred
    
    def fit_model(self,phase,phase_eb,winlen=100):
        params = [np.pi,1,5,.1]
        bounds = [(-np.pi,np.pi),(0,winlen),(.5,winlen),(-5,5)]
        
        
        def objective(params):
            phase_pred = self.predict_phase(phase_eb,params[0],params[1],params[2],params[3],winlen)
            return ug.circ_mse(phase[winlen:],phase_pred)
        self.res = minimize(objective,params,bounds=bounds,method='L-BFGS-B')
        self.winlen = winlen
            
            
phase = cxa.pdat['phase_fsb_upper']
phase_eb = cxa.pdat['phase_eb']  
mdl = model7()
mdl.fit_model(phase,phase_eb)

phase_pred = mdl.predict_phase(phase_eb,mdl.res.x[0],mdl.res.x[1],mdl.res.x[2],mdl.res.x[3])
x = np.arange(0,len(phase))
plt.scatter(x,phase,s=1)
plt.scatter(x[mdl.winlen:],phase_pred,s=1)
plt.figure()
x = np.arange(0,mdl.winlen)
kernel = stats.norm.pdf(x,loc=mdl.res.x[1],scale=mdl.res.x[2])*mdl.res.x[3]
plt.scatter(x,kernel,s=2,color='k')
#%% 
class model7_2:
    def __init__(self):
        self.version=1
    def predict_phase(self,phase_eb,velocity,offset,mu,sigma,w,gamma,winlen=100):
        vel_g = velocity**gamma
        phase_eb_o = ug.circ_subtract(phase_eb,offset)
        psin = np.sin(phase_eb_o)
        pcos= np.cos(phase_eb_o)
        kernel = stats.norm.pdf(np.arange(0,winlen),loc=mu,scale=sigma)*w
        psin_c = np.convolve(psin*vel_g,kernel,mode='full')
        pcos_p = np.convolve(pcos*vel_g,kernel,mode='full')
        phase_pred = np.arctan2(psin_c,pcos_p)
        phase_pred = phase_pred[winlen:-winlen+1]
        
        return phase_pred
    
    def fit_model(self,phase,phase_eb,velocity,winlen=100):
        if isinstance(winlen, (np.ndarray, list)):
            winlen = float(np.squeeze(winlen))
        params = [np.pi,10,5,.1,1]
        bounds = [(-np.pi,np.pi),(0,winlen),(.5,winlen),(-5,5),(-1,3)]
        #print(bounds)
        
        def objective(params):
            phase_pred = self.predict_phase(phase_eb,velocity,params[0],params[1],params[2],params[3],params[4],winlen)
            return ug.circ_mse(phase[winlen:],phase_pred)
        self.res = minimize(objective,params,bounds=bounds,method='L-BFGS-B')
        self.winlen = winlen
            
            
phase = cxa.pdat['phase_fsb_upper']
phase_eb = cxa.pdat['phase_eb']  
mdl = model7_2()

u = ug()
vx,vy,vd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'])

vd = np.append(0,vd)
vd = vd+ 1e-6

mdl.fit_model(phase,phase_eb,vd)
phase_pred = mdl.predict_phase(phase_eb,vd,mdl.res.x[0],mdl.res.x[1],mdl.res.x[2],mdl.res.x[3],mdl.res.x[4])
x = np.arange(0,len(phase))
plt.scatter(x,phase,s=1)
plt.scatter(x[mdl.winlen:],phase_pred,s=1)
plt.figure()
x = np.arange(0,mdl.winlen)
kernel = stats.norm.pdf(x,loc=mdl.res.x[1],scale=mdl.res.x[2])*mdl.res.x[3]
plt.scatter(x,kernel,s=2,color='k')

#%%
xO = np.arange(0,len(cxa.pdat['phase_eb']))

dx =np.arange(0,len(cxa.pdat['phase_eb']),500,dtype='int')
for i,d in enumerate(dx[:-1]):
    x = xO[d:dx[i+1]].copy()

    phase = cxa.pdat['phase_fsb_upper'][d:dx[i+1]]
    phase_eb = cxa.pdat['phase_eb'][d:dx[i+1]]
    mdl = model7_2()
    u = ug()
    vx,vy,vd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'])
    
    vd = np.append(0,vd)
    vd = vd+ 1e-6
    vd = vd[d:dx[i+1]]
    mdl.fit_model(phase,phase_eb,vd)
    phase_pred = mdl.predict_phase(phase_eb,vd,mdl.res.x[0],mdl.res.x[1],mdl.res.x[2],mdl.res.x[3],mdl.res.x[4])
    
    plt.figure(1)
    plt.scatter(x,phase,s=1,color='k')
    plt.scatter(x[mdl.winlen:],phase_pred,s=1,color='r')
    plt.figure(2)
    x = np.arange(0,mdl.winlen)
    kernel = stats.norm.pdf(x,loc=mdl.res.x[1],scale=mdl.res.x[2])*mdl.res.x[3]
    plt.scatter(x,kernel,s=2,color='k')
#%%
plt.close('')
phase = cxa.pdat['phase_fsb_upper']
phase_eb = cxa.pdat['phase_eb']
jumps = cxa.get_jumps()
vx,vy,vd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'])

vd = np.append(0,vd)
vd = vd+ 1e-6
winlen = 50
for j in jumps:
    dx = np.arange(j[0],j[2])
    mdl.fit_model(phase[dx],phase_eb[dx],vd[dx],winlen=winlen)
    phase_pred = mdl.predict_phase(phase_eb[dx],vd[dx],mdl.res.x[0],mdl.res.x[1],mdl.res.x[2],mdl.res.x[3],mdl.res.x[4],winlen=winlen)
    
    
    x = np.arange(0,mdl.winlen)
    kernel = stats.norm.pdf(x,loc=mdl.res.x[1],scale=mdl.res.x[2])*mdl.res.x[3]
    plt.figure()
    plt.subplot(2,2,1)
    plt.scatter(dx,phase[dx],color='k',s=2)
    plt.scatter(dx[mdl.winlen:],phase_pred,s=2)
    
    plt.subplot(2,2,2)
    plt.scatter(x,kernel,s=2,color='k')
#%% HMM 
from analysis_funs.CX_HMM import CX_HMM
hmm = CX_HMM(2) 
phase = cxa.pdat['phase_fsb_upper']
phase_eb = cxa.pdat['phase_eb']
hmm.fit(phase,phase_eb,verbose=True)
#%%
phase_short = phase[mdl.winlen:]
states = hmm.viterbi(phase,phase_eb)
x = np.arange(0,len(phase_short))
ustates = np.unique(states)
plt.scatter(x,phase_short,color='k',s=3)
#plt.scatter(x,phase_eb,color='r',s=3)
ypred = np.zeros_like(phase_short)
for u in ustates:
    dx = states==u
    plt.scatter(x[dx],hmm.wrap_ang(phase_pred[dx]+hmm.angles[u]),s=2)
    ypred[dx] = phase_pred[dx]+hmm.angles[u]

ug.circ_mse(phase_short,ypred)
u = ug()
vx,vy,vd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'])
vd = vd/np.std(vd)
vs = np.append(0,vd)
plt.plot(x[1:],vd[mdl.winlen:]/2 -6,color=[0.3,0.3,0.3])
#%%
#%% Guassian kernel with states
class model7s:
    def __init__(self):
        self.version=1
    def predict_phase(self,phase_eb,states,offsets,mus,sigmas,ws,winlen=100):
        ustates = np.unique(states)
        total_convs = np.zeros((len(phase_eb),2,len(ustates)))
        phase_pred = np.zeros_like(phase_eb)
        for u in ustates:
            phase_eb_o = ug.circ_subtract(phase_eb,offsets[u])
            psin = np.sin(phase_eb_o)
            pcos = np.cos(phase_eb_o)
            kernel = stats.norm.pdf(np.arange(0,winlen),loc=mus[u],scale = sigmas[u])*ws[u]
            psin_c = np.convolve(psin,kernel,mode='full')
            pcos_p = np.convolve(pcos,kernel,mode='full')
            total_convs[:,0,u] = psin_c[:-winlen+1]
            total_convs[:,1,u] = pcos_p[:-winlen+1]
        
            dx = states==u
            phase_pred[dx] = np.arctan2(total_convs[dx,0,u],total_convs[dx,1,u])
        
        return phase_pred[winlen:]
    
    def fit_model(self,phase,phase_eb,states,winlen=100):
        nstates = len(np.unique(states))
        params = np.concatenate([
            np.zeros(nstates),
            np.zeros(nstates)+10,
            np.zeros(nstates)+10 ,
            np.zeros(nstates)+.1
            
            ])
        
        bounds = ([(-np.pi,np.pi)]*nstates+
                  [(0,winlen)]*nstates+
                  [(.5,winlen)]*nstates+
                  [(-5,5)]*nstates)
        #print(bounds)
        
        def objective(params):
            phase_pred = self.predict_phase(phase_eb,states,params[0:nstates],params[nstates:nstates*2],
                                            params[nstates*2:nstates*3],params[nstates*3:],winlen)
            return ug.circ_mse(phase[winlen:],phase_pred)
        self.res = minimize(objective,params,bounds=bounds,method='L-BFGS-B')
        self.winlen = winlen

mdl = model7s()
states = hmm.viterbi(phase,phase_eb)
mdl.fit_model(phase,phase_eb,states)
params = mdl.res.x
ins = cxa.ft2['instrip']

nstates  = len(np.unique(states))
phase_pred = mdl.predict_phase(phase_eb,states,params[0:nstates],params[nstates:nstates*2],
                                params[nstates*2:nstates*3],params[nstates*3:])


x = np.arange(0,len(phase))
plt.scatter(x,phase,color='k',s=2)
state_short = states[mdl.winlen:]
plt.figure(1)
plt.plot(x,ins*np.pi,color='r')

vx,vy,vd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'])
vd = vd/np.std(vd)
vd = np.append(0,vd)
plt.plot(x,vd-8,color=[0.5,0.5,0.5])

for i in range(nstates):
    plt.figure(1)
    x = np.arange(0,len(phase))
    
    plt.scatter(x[mdl.winlen:][state_short==i],phase_pred[state_short==i],s=2)
    
    
    
    x = np.arange(0,mdl.winlen)
    
    plt.figure(2)
    offset = 180*params[i]/np.pi
    print(offset)
    kernel = stats.norm.pdf(x,loc=mdl.res.x[nstates+i],scale=mdl.res.x[2*nstates+i])*mdl.res.x[3*nstates+i]
    plt.scatter(x,kernel,s=2)
    plt.figure(3)
    plt.scatter(phase[mdl.winlen:][state_short==i],phase_eb[mdl.winlen:][state_short==i],s=2)
#%% Gaussian kernel with states plus kernel of itself
class model7s2:
    def __init__(self):
        self.version=1
    def predict_phase(self,phase_eb,states,offsets,mus,sigmas,ws,winlen=100):
        ustates = np.unique(states)
        total_convs = np.zeros((len(phase_eb),2,len(ustates)))
        phase_pred = np.zeros_like(phase_eb)
        for u in ustates:
            phase_eb_o = ug.circ_subtract(phase_eb,offsets[u])
            psin = np.sin(phase_eb_o)
            pcos = np.cos(phase_eb_o)
            kernel = stats.norm.pdf(np.arange(0,winlen),loc=mus[u],scale = sigmas[u])*ws[u]
            psin_c = np.convolve(psin,kernel,mode='full')
            pcos_p = np.convolve(pcos,kernel,mode='full')
            total_convs[:,0,u] = psin_c[:-winlen+1]
            total_convs[:,1,u] = pcos_p[:-winlen+1]
        
            dx = states==u
            phase_pred[dx] = np.arctan2(total_convs[dx,0,u],total_convs[dx,1,u])
        
        return phase_pred[winlen:]
    
    def fit_model(self,phase,phase_eb,states,winlen=100):
        nstates = len(np.unique(states))
        params = np.concatenate([
            np.zeros(nstates),
            np.zeros(nstates)+10,
            np.zeros(nstates)+10 ,
            np.zeros(nstates)+.1
            
            ])
        
        bounds = ([(-np.pi,np.pi)]*nstates+
                  [(0,winlen)]*nstates+
                  [(.5,winlen)]*nstates+
                  [(-5,5)]*nstates)
        #print(bounds)
        
        def objective(params):
            phase_pred = self.predict_phase(phase_eb,states,params[0:nstates],params[nstates:nstates*2],
                                            params[nstates*2:nstates*3],params[nstates*3:],winlen)
            return ug.circ_mse(phase[winlen:],phase_pred)
        self.res = minimize(objective,params,bounds=bounds,method='L-BFGS-B')
        self.winlen = winlen

mdl = model7s()
mdl.fit_model(phase,phase_eb,states)
params = mdl.res.x


nstates  = len(np.unique(states))
phase_pred = mdl.predict_phase(phase_eb,states,params[0:nstates],params[nstates:nstates*2],
                                params[nstates*2:nstates*3],params[nstates*3:])


x = np.arange(0,len(phase))
plt.scatter(x,phase,color='k',s=2)
state_short = states[mdl.winlen:]
for i in range(nstates):
    plt.figure(1)
    x = np.arange(0,len(phase))
    
    plt.scatter(x[mdl.winlen:][state_short==i],phase_pred[state_short==i],s=2)
    
    
    
    x = np.arange(0,mdl.winlen)
    
    plt.figure(2)
    offset = 180*params[i]/np.pi
    print(offset)
    kernel = stats.norm.pdf(x,loc=mdl.res.x[nstates+i],scale=mdl.res.x[2*nstates+i])*mdl.res.x[3*nstates+i]
    plt.scatter(x,kernel,s=2)
#%%
class model7_3:
    def __init__(self):
        self.version=1
    def predict_phase(self,phase_eb,velocity,ins,offset_in,offset_out,
                      mu_in,mu_out,sigma_in,sigma_out,w_in,w_out,gamma,winlen=100):
        
        vel_g = velocity*gamma
        phase_eb_o = ug.circ_subtract(phase_eb,offset_out)
        phase_eb_in = ug.circ_subtract(phase_eb,offset_in)
        
        psino = np.sin(phase_eb_o)
        pcoso= np.cos(phase_eb_o)
        #psino[ins>0] = 0
        #pcoso[ins>0] = 0
        
        psini = np.sin(phase_eb_in)
        pcosi = np.cos(phase_eb_in)
        #psini[ins<1] = 0 
       # pcosi[ins<1] = 0 
        
        
        
        kernel_in = stats.norm.pdf(np.arange(0,winlen),loc=mu_in,scale=sigma_in)*w_in
        kernel_out = stats.norm.pdf(np.arange(0,winlen),loc=mu_out,scale=sigma_in)*w_out
        
        psin_c = np.convolve(psino,kernel_out,mode='full')[:-winlen+1]
        psin_c[ins>0] = np.convolve(psini,kernel_in,mode='full')[:-winlen+1][ins>0] 
        pcos_p = np.convolve(pcoso*vel_g,kernel_out,mode='full')[:-winlen+1]
        pcos_p[ins>0] = +np.convolve(pcosi*vel_g,kernel_in,mode='full')[:-winlen+1][ins>0] 
        
        phase_pred = np.arctan2(psin_c,pcos_p)
        phase_pred = phase_pred[winlen:]
        
        return phase_pred
    
    def fit_model(self,phase,phase_eb,velocity,ins,winlen=100):
        if isinstance(winlen, (np.ndarray, list)):
            winlen = float(np.squeeze(winlen))
        params = [0,np.pi,
                  1,1,
                  5,5,
                  .1,.1,
                  1]
        bounds = [(-np.pi,np.pi),(-np.pi,np.pi),
                  (0,winlen),(0,winlen),
                  (.5,winlen),(.5,winlen),
                  (-5,5),(-5,5),
                  (0,3)]
        #print(bounds)
        
        def objective(params):
            phase_pred = self.predict_phase(phase_eb,velocity,ins,
                                            params[0],
                                            params[1],
                                            params[2],
                                            params[3],
                                            params[4],
                                            params[5],
                                            params[6],
                                            params[7],
                                            params[8],
                                            winlen)
            return ug.circ_mse(phase[winlen:],phase_pred)
        self.res = minimize(objective,params,bounds=bounds,method='L-BFGS-B')
        self.winlen = winlen
            
            
phase = cxa.pdat['phase_fsb_upper']
phase_eb = cxa.pdat['phase_eb'] 
ins  = cxa.ft2['instrip'].to_numpy()
u = ug()
vx,vy,vd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'])

vd = np.append(0,vd)
vd = vd+ 1e-6
vd = .01*vd/np.max(vd)
mdl = model7_3()
mdl.fit_model(phase,phase_eb,vd,ins)


phase_pred = mdl.predict_phase(phase_eb,vd,mdl.res.x[0],mdl.res.x[1],mdl.res.x[2],mdl.res.x[3],mdl.res.x[4],
                               mdl.res.x[5],mdl.res.x[6],mdl.res.x[7],mdl.res.x[8],mdl.winlen)
x = np.arange(0,len(phase))
plt.scatter(x,phase,s=1)
plt.scatter(x[mdl.winlen:],phase_pred,s=1)
plt.figure()
x = np.arange(0,mdl.winlen)
kernel = stats.norm.pdf(x,loc=mdl.res.x[2],scale=mdl.res.x[4])
plt.scatter(x,kernel,s=2,color='k')
kernel = stats.norm.pdf(x,loc=mdl.res.x[3],scale=mdl.res.x[5])
plt.scatter(x,kernel,s=2,color='k')
#%%
y = np.zeros(100)
y[0] = 1
y[50] = 1.0 
kernel = np.array([-1,1,-1])
c = np.convolve(y,kernel,mode='valid')
plt.plot(c)
plt.plot(y)


#%% model 7 - chat gpt optimised
import numpy as np
from numba import njit
@njit
def circ_subtract(a, b):
    """Fast circular subtraction in [-pi, pi]."""
    return np.angle(np.exp(1j * (a - b)))

@njit
def predict_phase_numba(phase, phase_eb, offset, w):
    wlen = len(w)
    n = len(phase)
    phase_pred = np.zeros(n)
    phase_eb_offset = circ_subtract(phase_eb, offset)
    
    for i in range(wlen, n):
        dphase = circ_subtract(phase_eb_offset[i-wlen:i], phase_pred[i-1])
        #circ_std = np.std(dphase)+1e-6  # approximate circstd for speed
        #scale = (velocity[i]**gamma) / (circ_std**alpha)
        delta = np.dot(w, dphase)
        phase_pred[i] = circ_subtract(phase_pred[i-1], delta)
    return phase_pred

class model7:
    def __init__(self):
        self.version = 2

    def predict_phase(self, phase, phase_eb, offset, w):
        return predict_phase_numba(phase, phase_eb, offset, w)

    def circ_mse(self, y, ypred):
        return np.mean(1 - np.cos(circ_subtract(y, ypred)))**.5

    def fit_model(self, phase, phase_eb, winlength=30):
        def objective(params):
            w = params[:winlength]
            #alpha, beta, gamma, offset = params[winlength:]
            offset = params[-1]
            ypred = self.predict_phase(phase, phase_eb, offset, w)
            if np.any(np.isnan(ypred)) or np.any(np.isinf(ypred)):
                print("⚠️ Numerical overflow detected in predict_phase")
            return self.circ_mse(phase[winlength:], ypred[winlength:])
        
        def verbose_callback(params):
            print("Iteration params summary:")
            print(f"  mean(w): {np.mean(params[:winlength]):.4f}, "
                  f"offset: {params[-1]:.3f}")
              #f"alpha: {params[winlength]:.3f}, "
            # print(f"  mean(w): {np.mean(params[:winlength]):.4f}, "
            
            #       f"alpha: {params[winlength]:.3f}, "
            #       f"beta: {params[winlength+1]:.3f}, "
            #       f"gamma: {params[winlength+2]:.3f}, "
            #       f"offset: {params[winlength+3]:.3f}")
        bounds = [(-1, 1)]*winlength + [ (-np.pi, np.pi)]
        best_fun = np.inf
        best_res = None
        
        for _ in range(5):
            print('Iteration ',_)
            init_params = np.concatenate([
                np.random.uniform(-0.1, 0.1, winlength),
                [np.random.uniform(-np.pi, np.pi)]
            ])
            res = minimize(objective, init_params, bounds=bounds, method='L-BFGS-B',callback=verbose_callback)
            if res.fun < best_fun:
                best_fun = res.fun
                best_res = res
        self.res = best_res
        
        
        
        #init_params = np.concatenate([np.ones(winlength)*0.01, [1.0, 1.0, 1.0, np.pi]])
        

        #self.res = minimize(objective, init_params, bounds=bounds, method='Nelder-Mead', options={'maxiter': 500, 'disp': True},callback=verbose_callback)
    
phase = cxa.pdat['phase_fsb_upper']
phase_eb = cxa.pdat['phase_eb']  
offset = np.pi
w  =np.ones(50)
u = ug()
vx,vy,vd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'])
vd = vd+ 1e-6
mdl = model7()
mdl.fit_model(phase,phase_eb,winlength=50)
pred = mdl.predict_phase(phase,phase_eb,mdl.res.x[-1],mdl.res.x)
x = np.arange(0,len(phase))
plt.scatter(x,phase,s=1)
plt.scatter(x,pred,s=1)
#%%
datadirs_hdj =  [
                r"Y:\Data\FCI\Hedwig\hDeltaK_SS63089\250807\f2\Trial4", #hDeltaK benchmark
                
                r"Y:\Data\FCI\Hedwig\hDeltaJ\240529\f1\Trial3",#Good pointer
                r"Y:\Data\FCI\Hedwig\hDeltaJ\251011\f1\Trial1",
                r"Y:\Data\FCI\Hedwig\hDeltaJ\251022\f1\Trial2",
                r"Y:\Data\FCI\Hedwig\hDeltaJ\251023\f1\Trial1", # Running through plumes, activity becomes backwards pointing vector
                r"Y:\Data\FCI\Hedwig\hDeltaJ\251023\f1\Trial3",
                
                r"Y:\Data\FCI\Hedwig\hDeltaJ\251023\f2\Trial2", #Problems with image registration. Multiple plumes, very interesting dataset, looks like neurons poitn backwards after exit to where first turn was. Some integration of stim
                r"Y:\Data\FCI\Hedwig\hDeltaJ\251023\f2\Trial3",
                r"Y:\Data\FCI\Hedwig\hDeltaJ\251028\f1\Trial2",
                r"Y:\Data\FCI\Hedwig\hDeltaJ\251028\f2\Trial2",
                r'Y:\Data\FCI\Hedwig\hDeltaJ\251029\f1\Trial2',
                ]
all_flies_hdj = {}
etp_hdj = {}
for i,datadir in enumerate(datadirs_hdj):
    print(datadir)
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    all_flies_hdj.update({str(i):cxa})
    
    
#%% 
data = np.zeros((len(datadirs_hdj),5))
names = np.array([])
all_models = {}
for i in all_flies_hdj:
    print(i)
    cxa = all_flies_hdj[i]
    names = np.append(names,cxa.name)
    print(cxa.name)
    phase = cxa.pdat['phase_fsb_upper']
    phase_eb = cxa.pdat['phase_eb']
    ins = cxa.ft2['instrip'].to_numpy()
    vx,vy,vd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'])
    vd[vd<.5] = 0
    vd = vd/np.std(vd)
    
    #Model 1
    print('Model1')
    mdl1 = model1()
    mdl1.fit_model(phase,phase_eb)
    
    data[int(i),0] = mdl1.res.fun
    
    #Model 2
    print('Model2')
    mdl2 = model2()
    mdl2.fit_model(phase,phase_eb,30)
    
    data[int(i),1] = mdl2.res.fun
    
    
    
    print('Model3')
    mdl3 = model3()
    mdl3.fit_model(phase,phase_eb,ins,30)
    data[int(i),2] = mdl3.res.fun
    phase_pred = mdl3.predict_phase(phase[mdl3.delay:],phase_eb[:-mdl3.delay],mdl3.res.x[0],mdl3.res.x[1],ins[mdl3.delay:])
    phase_pred3 = phase_pred.copy()
    
    print('Model3_2')
    mdl3_2 = model3_2()
    mdl3_2.fit_model(phase,phase_eb,ins,30)
    data[int(i),3] = mdl3_2.res.fun
    
    print('Model5_3 parallel')
    mdl5_3p = model5_3p()
    mdl5_3p.fit_model(phase,phase_eb,15,vd,ins)
    data[int(i),4] = mdl5_3p.res.fun
    # print('Model6')
    # mdl6.fit_model(phase,phase_eb,5,vd,ins)
    # phase_pred = mdl6.predict_phase(phase[mdl6.delay:],phase_eb[:-mdl6.delay],mdl6.res.x[3],mdl6.res.x[4],
    #                                 mdl6.res.x[0],mdl6.res.x[1],vd[:-mdl5.delay],mdl6.res.x[2],ins[:-mdl5.delay])
    # data[int(i),2] = mdl6.res.fun
    
    
    # x = np.arange(0,len(phase))
    # plt.figure()
    # plt.scatter(x,phase,color='k',s=2)
    # plt.scatter(x[mdl6.delay:],phase_pred,color='r',s=2)
    # plt.scatter(x[mdl3.delay:],phase_pred3,color='g',s=2)
    # plt.plot(x,ins*np.pi*2 - np.pi,color='b')
    all_models.update({i:{'mdl1':mdl1,'mdl2':mdl2,'mdl3':mdl3,'mdl3_2':mdl3_2,'mdl5_3p':mdl5_3p}})
    
data_an = 180*np.arccos(1-data)/np.pi
plt.plot(data_an.T)
plt.xticks(np.arange(0,5),labels=['Model1','Model 2','Model 3','Model 3 +','Model 5 +'])
plt.ylabel('Average error (degrees)')
plt.legend(names)
#%% Plot model data for each trial
plt.close('all')
data_an = 180*np.arccos(1-data)/np.pi
plt.plot(data_an.T)
plt.xticks(np.arange(0,5),labels=['Model1','Model 2','Model 3','Model 3 +','Model 5 +'])
plt.ylabel('Average error (degrees)')
plt.legend(names)
for i in all_flies_hdj:
    cxa = all_flies_hdj[i]
    #mdl5_3p = all_models[i]['mdl5_3p']
    mdl3_2 = all_models[i]['mdl3_2']
    phase = cxa.pdat['phase_fsb_upper']
    phase_eb = cxa.pdat['phase_eb']
    ins = cxa.ft2['instrip'].to_numpy()
    heading = cxa.ft2['ft_heading'].to_numpy()
    vx,vy,vd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'])
    vd[vd<.5] = 0
    vd = vd/np.std(vd)
    # phase_pred = mdl5_3p.predict_phase(phase[mdl5_3p.delay:],phase_eb[:-mdl5_3p.delay],mdl5_3p.res.x[0],mdl5_3p.res.x[1]
    #                                 ,mdl5_3p.res.x[2],mdl5_3p.res.x[3],mdl5_3p.res.x[4],mdl5_3p.res.x[5],vd[:-mdl5_3p.delay],ins[:-mdl5_3p.delay])
    
    phase_pred = mdl3_2.predict_phase(phase[mdl3_2.delay:],phase_eb[:-mdl3_2.delay],mdl3_2.res.x[0],mdl3_2.res.x[1],mdl3_2.res.x[2],
                                ins[:-mdl3_2.delay])
    
    error = ug.circ_subtract(phase[mdl3_2.delay:],phase_pred)
    
    x = np.arange(0,len(phase))
    plt.figure()
    plt.scatter(x,heading,s=5,color=[0.5,0.5,0.5])
    plt.scatter(x,phase,s=2,color='k')
    plt.scatter(x[mdl3_2.delay:],phase_pred,s=2,color='r')
    #plt.scatter(x,phase_eb,color='g',s=2)
    plt.plot(x,ins*np.pi*2-np.pi,color='b')
    #plt.plot(x[1:],vd/2-4,color=[0.5,.5,.5])
    plt.title(cxa.name)
    
    
    #plt.scatter(x,phase,s=2,color='k')
    #plt.scatter(x[mdl3_2.delay:],np.abs(error)-7,color='r',s=2)
    plt.plot(x[mdl3_2.delay:],np.abs(error)-7,color='g')
    plt.plot([x[0],x[-1]],np.array([np.pi/2,np.pi/2])-7,color='k',linestyle='--')
    #plt.scatter(x[mdl3_2.delay:],phase_pred,s=2,color='r')
    #plt.scatter(x,phase_eb,color='g',s=2)
   # plt.plot(x,ins*np.pi-7,color='b')
    plt.plot(x[1:],vd/2-11,color=[0.5,.5,.5])
    plt.title(cxa.name)

#%%


data2 = np.zeros((3,3))
for i in range(3):
   
    cxa = all_flies_fc2[str(i)]
    print(cxa.name)
    phase = cxa.pdat['phase_fsb_upper']
    phase_eb = cxa.pdat['phase_eb']
    ins = cxa.ft2['instrip'].to_numpy()
    vx,vy,vd = u.get_velocity(cxa.ft2['ft_posx'].to_numpy(),cxa.ft2['ft_posy'].to_numpy(),cxa.pv2['relative_time'])
    vd = vd/np.std(vd)
    
    #Model 1
    print('Model1')
    mdl1 = model1()
    mdl1.fit_model(phase,phase_eb)
    
    data2[i,0] = mdl1.res.fun
    
    print('Model3')
    mdl3 = model3()
    mdl3.fit_model(phase,phase_eb,ins,30)
    data2[i,1] = mdl3.res.fun
    phase_pred = mdl3.predict_phase(phase[mdl3.delay:],phase_eb[:-mdl3.delay],mdl3.res.x[0],mdl3.res.x[1],ins[mdl3.delay:])
    phase_pred3 = phase_pred.copy()
    
    print('Model6')
    mdl6.fit_model(phase,phase_eb,5,vd,ins)
    phase_pred = mdl6.predict_phase(phase[mdl6.delay:],phase_eb[:-mdl6.delay],mdl6.res.x[3],mdl6.res.x[4],
                                    mdl6.res.x[0],mdl6.res.x[1],vd[:-mdl5.delay],mdl6.res.x[2],ins[:-mdl5.delay])
    data2[i,2] = mdl6.res.fun
    
    
    x = np.arange(0,len(phase))
    plt.figure()
    plt.scatter(x,phase,color='k',s=2)
    plt.scatter(x[mdl6.delay:],phase_pred,color='r',s=2)
    plt.scatter(x[mdl3.delay:],phase_pred3,color='g',s=2)
    plt.plot(x,ins*np.pi*2 - np.pi,color='b')
    
plt.figure()
plt.plot(data2.T)
plt.xticks(np.arange(0,3),labels=['Model1','Model 3','Model 6'])
plt.ylabel('Mean squared error (radians**2)')
