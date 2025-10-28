# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 16:32:36 2025

@author: dowel
"""

import sklearn.linear_model as lm
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import statsmodels.api as sm
from analysis_funs.utilities import funcs as fn
from scipy import signal
from Utilities.utils_general import utils_general as ug
from analysis_funs.regression import fci_regmodel
from scipy.optimize import minimize
#%%
class CX_corr:
    def __init__(self,cxa):
        self.cxa = cxa
        self.dmx = 0
    def set_up_regressors(self,regchoice,delays=[0],use_odour_delay=False,offset=False): 
        # To do: determine odour onset delay, add in 
        # Movement, in odour bias, in plume goals, out odour bias
        
        dmx= self.dmx
        regions = self.cxa.regions
        self.offset = offset
        ins = self.cxa.ft2['instrip'].to_numpy()
        expdims = len(ins)
        e_e = self.cxa.get_entries_exits_like_jumps()
        self.col_dim=np.nan
        self.param_ticks = np.array([0])
        if use_odour_delay:
            fci = fci_regmodel(np.mean(self.cxa.pdat['wedges_fsb_upper'],axis=1),self.cxa.ft2,self.cxa.pv2)
            fci.run(['in odour'])
            regdelay = fci.delay
        
        for i,r in enumerate(regchoice):
            print(i)
            if r in regions:
                if offset==True:
                    tX = self.cxa.pdat['wedges_offset_' + r]
                else:
                    tX = self.cxa.pdat['wedges_'+r]
                tX = (tX-np.mean(tX,axis=0)/np.std(tX,axis=0)) # z score
                #tX = tX/np.std(tX,axis=0) # divide by variance
                dmx = np.max(delays)
                self.dmx =dmx
                d= delays[0]
                X2 = tX[dmx-d:,:].copy()
                self.param_ticks = np.append(self.param_ticks,np.max(self.param_ticks)+16)
                if len(delays)>1:
                    for d in delays[1:]:
                        X2 = np.append(X2,tX[dmx-d:-d,:],axis=1)
                        self.param_ticks = np.append(self.param_ticks,np.max(self.param_ticks)+16)
                
                ddim = X2.shape[1]
                # X2 = np.append(X2,X2,axis=1)
                # if use_odour_delay:
                #     ins2 = np.append(np.zeros(regdelay),ins[:-regdelay])
                #     ins2 = ins[dmx:]
                # else:
                #     ins2 = ins[dmx:]
                # self.param_ticks = np.append(self.param_ticks,np.max(self.param_ticks)+self.param_ticks[1:])
                # X2[ins2==1,0:ddim] = 0 
                # X2[ins2==0,ddim:] = 0
                self.col_dim = ddim
            elif r=='ret goal':
                
                X2 = np.zeros((expdims,len(e_e)))
                for i2,e in enumerate(e_e):
                    if use_odour_delay:
                        dx = np.arange(e[1]+regdelay,e[2]+regdelay)
                    else:
                        dx = np.arange(e[1],e[2])
                    X2[dx,i2] = 1
                X2 = X2/np.std(X2,axis=0)
                X2 = X2[dmx:,:]
                self.param_ticks = np.append(self.param_ticks,self.param_ticks[-1]+X2.shape[1])
            elif r == 'leave goal':
                X2 = np.zeros((expdims,len(e_e)))
                for i2,e in enumerate(e_e):
                    if use_odour_delay:
                        dx = np.arange(e[0]+regdelay,e[1]+regdelay)
                    else:
                        dx = np.arange(e[0],e[1])
                    X2[dx,i2] = 1
                X2 = X2/np.std(X2,axis=0)
                X2 = X2[dmx:,:]
                self.param_ticks = np.append(self.param_ticks,self.param_ticks[-1]+X2.shape[1])
            elif r=='pre bias':
                X2 = np.zeros((expdims,1))
                dx = np.arange(0,e_e[0,0])
                X2[dx] = 1
                X2 = X2[dmx:]
                X2 = X2/np.std(X2)
                
            elif r =='translational vel':
                fci = fci_regmodel(np.ones(len(ins)),self.cxa.ft2,self.cxa.pv2)
                X2,Xo = fci.set_up_regressors([r])
                X2 = X2[dmx:,0]
                X2 = X2[:,np.newaxis]
                
            if i==0:
                X = X2.copy()
            else:
                X = np.append(X,X2,axis=1)
        self.X = X
        self.dmx =dmx
    def run(self,fit_region,plot_diagnostic=False,fit_type='OLS',enforce_conn = False,conn_off = 0,dilation=1):
        if self.offset:
            Y = self.cxa.pdat['wedges_offset_'+fit_region]
        else:
            Y = self.cxa.pdat['wedges_'+fit_region]
        Y = (Y-np.mean(Y,axis=0))/np.std(Y,axis=0)
        #Y = (Y)/np.std(Y,axis=0)
        Y = Y[self.dmx:,:]
        self.Y = Y.copy()
        X = self.X.copy()
        all_params= np.zeros((16,X.shape[1]))
        wedpred = np.zeros_like(Y)
        scores = np.zeros(Y.shape[1])
        if not np.isnan(self.col_dim):
            col_dim = self.col_dim*2
            num_cols = int(col_dim/16)
            dilarray = np.arange(-dilation,dilation+1)
        for i in range(Y.shape[1]):
            y = Y[:,i]
            X = self.X.copy()
            if enforce_conn:
                dx = np.arange(0,col_dim)
                idx = np.mod(i+conn_off,16)+dilarray
                
                idx[idx<0] = 16+idx[idx<0]
                idx[idx>15] = idx[idx>15]-16

                for c in range(num_cols-1):
                    idx= np.append(idx,idx+(c+1)*16)
               
                
                dx = dx[~np.in1d(dx,idx)] 
                
                X[:,dx] = 0
            
            if fit_type=='ridge':
                reg = lm.Ridge(alpha=0.1)
                reg.fit(X,y)
                all_params[i,:] = reg.coef_
                wedpred[:,i] = reg.predict(X)
                scores[i] = reg.score(X,y)
                
            elif fit_type=='OLS':
                reg = lm.LinearRegression(fit_intercept=True)
                reg.fit(X,y)
                all_params[i,:] = reg.coef_
                wedpred[:,i] = reg.predict(X)
                scores[i] = reg.score(X,y)
            elif fit_type == 'GLM exp':
                self.fit_exp_glm(X,y)
                scores[i] =self.r2
                all_params[i,:] = self.beta_hat
                wedpred[:,i] = self.predy
                
                
            
            print(scores[i])
            
        self.all_params = all_params
        self.wedpred=wedpred
        self.R2 = scores
        
        predphase = ug.phase_from_wed(self.wedpred)
        predphaseinput = np.append(np.zeros(self.dmx),predphase)
        wedpredinput = np.append(np.zeros((self.dmx,16)),self.wedpred,axis=0)
        self.cxa.pdat['phase_fsb_upper_pred'] = predphaseinput
        if self.offset:
            self.cxa.pdat['offset_fsb_upper_pred_phase'] = pd.Series(predphaseinput)
        else:  
            self.cxa.pdat['offset_fsb_upper_pred_phase'] = pd.Series(ug.circ_subtract(predphaseinput,self.cxa.pdat['offset'].to_numpy()))
        self.cxa.pdat['wedges_fsb_upper_pred'] = (wedpredinput-np.min(wedpredinput))*.2
        
        if plot_diagnostic:
            self.cxa.plot_traj_arrow_new(['fsb_upper','fsb_upper_pred'],a_sep=5)
            self.diagnostic_plots()
    def fit_exp_glm(self,X,y,tol=1e-6):
        # Function fits a glm with an exponential link function, similar to Pillow et al. 2008
        n, p = X.shape
        init_params = np.zeros(p+1)
        init_params[-1] = np.log(np.std(y))
        
        def neg_log_likelihood(params):
            beta = params[:-1]
            log_sigma = params[-1]
            sigma = np.exp(log_sigma)
            
            mu = np.exp(X@beta) # link function
            resid = y-mu
            nll = 0.5*np.sum((resid/sigma))**2 + np.log(2*np.pi*sigma**2) # log of gaussian
            return nll
        result = minimize(neg_log_likelihood,init_params,method='BFGS')
        self.beta_hat = result.x[:-1]
        self.sigma_hat = np.exp(result.x[-1])
        self.predy = np.exp(X@self.beta_hat)
        self.r2 = ug.r2(y,self.predy)
        
    def diagnostic_plots(self):
        plt.figure()
        plt.imshow(self.all_params,vmax=np.percentile(self.all_params[:],95),vmin = -np.percentile(self.all_params[:],95),cmap='coolwarm')
        plt.xticks(np.arange(0,16*2,16));
        plt.figure()

        plt.imshow(np.append(np.append(self.wedpred,self.Y,axis=1),self.X[:,:16],axis=1),interpolation='None',aspect='auto')

        error= self.Y-self.wedpred
        msq = np.sum(error**2,axis=1)
        plt.figure()
        plt.plot(msq)
        ins = self.cxa.ft2['instrip'].to_numpy()[self.dmx:]
        plt.plot(ins*50,color='r')

        predphase = ug.phase_from_wed(self.wedpred)
        fsbphase = self.cxa.pdat['phase_fsb_upper'][self.dmx:]
        ebphase = self.cxa.pdat['phase_eb'][self.dmx:]
        plt.figure()
        plt.plot(predphase)
        plt.plot(fsbphase)
        plt.figure()
        plt.scatter(predphase,fsbphase,s=1)
        plt.figure()
        plt.scatter(ebphase,fsbphase,s=1)
        msqplt = np.append(np.zeros(self.dmx),msq)
        fci2 = fci_regmodel(msqplt,self.cxa.ft2,self.cxa.pv2)
        fci2.example_trajectory_jump(msqplt,self.cxa.ft,cmin=0,cmax=20)
        
        

        plt.figure()
        plt.plot(self.cxa.pdat['offset_fsb_upper_phase'],color='b')
        plt.plot(self.cxa.pdat['offset_fsb_upper_pred_phase'],color='r' )
        plt.plot(self.cxa.pdat['offset_eb_phase'],color='k')
        plt.plot(self.cxa.ft2['instrip'])
        
        
        