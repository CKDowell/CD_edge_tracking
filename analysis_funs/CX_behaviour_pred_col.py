# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 10:36:02 2025

@author: dowel

Objective:
    1. To use regression methods to predict the future trajectory/velocity of the fly using imaging data
    2. To use more advanced machine learning based methods to do the same


"""

import sklearn.linear_model as lm
from analysis_funs.CX_imaging import CX
import os
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from scipy.stats import circmean, circstd
from scipy import signal as sg
from analysis_funs.utilities import funcs as fn
import pickle
from scipy.optimize import curve_fit,minimize
from analysis_funs.CX_imaging import CX
from analysis_funs.CX_analysis_col import CX_a

class CX_b:
    def __init__(self,datadir,regions):
        self.cxa = CX_a(datadir,regions=regions,denovo=False)
    
    def reg_traj_model(self,twindow,regions,mtype='Phase_amp'):
        # Here we are going to make a simple regression based model to predict the 
        # New direction of the fly based upon neural data.
        # Model types:
            # 1. Phase and amplitude models
            # 2. Raw wedges
        cxa = self.cxa
        ft2 = cxa.ft2
        pv2 = cxa.pv2
        heading= ft2['ft_heading'].to_numpy()
        tt = pv2['relative_time'].to_numpy()
        tstep = np.mean(np.diff(tt))
        winlen =np.round( twindow/tstep).astype('int')
        if mtype=='Phase_amp':
            t_phase = np.zeros((len(heading),len(regions)))
            t_amp_mean = np.zeros((len(heading),len(regions)))
            t_amp_pva =np.zeros((len(heading),len(regions)))
            
            for i,r in enumerate(regions):
                tp = cxa.pdat['offset_'+r+'_phase'].to_numpy()
                t_phase[:,i] = tp
                twed  = cxa.pdat['wedges_'+r]
                ymn_z,pvan_z,pva_z = self.output_amps(twed)
                t_amp_mean[:,i] = ymn_z
                t_amp_pva[:,i] = pva_z
            
            # Make regression matrix
            input_mat = np.zeros((len(heading)-winlen,winlen*len(regions)*3))
            for i  in range(len(heading)-winlen):
                for ir,r in enumerate(regions):
                    
                    p = t_phase[i:i+winlen,ir]
                    p = (p-np.mean(p))/np.std(p)
                    pdx = np.arange(ir*winlen,ir*winlen+winlen)
                    input_mat[i,pdx] = p
                    
                    am = t_amp_mean[i:i+winlen,ir]
                    am = (am-np.mean(am))/np.std(am)
                    adx = np.arange(ir*winlen,ir*winlen+winlen)+len(regions)*winlen
                    input_mat[i,adx] = am
                    
                    av = t_amp_pva[i:i+winlen,ir]
                    av = (av-np.mean(av)/np.std(av))
                    adx = np.arange(ir*winlen,ir*winlen+winlen)+len(regions)*winlen*2
                    input_mat[i,adx] = av
            
        ymatrix = np.zeros((len(input_mat),2))
        uheading = fn.unwrap(heading)
        uheadings = sg.savgol_filter(uheading,10,3)
        plt.plot(uheading)
        plt.plot(uheadings)
        dheading = np.diff(uheading)
        #plt.plot(dheading)
                    
                    
    def output_amps(self,wedges):
        angles = np.linspace(-np.pi,np.pi,16)
        weds = np.sum(wedges*np.sin(angles),axis=1)
        wedc = np.sum(wedges*np.cos(angles),axis=1)
        pva  = np.sqrt(weds**2+wedc**2)
        p0 = np.mean(pva[pva<np.percentile(pva,10)])
        pva = (pva-p0)/p0

        # pva_norm - measure of coherence

        wednorm = wedges
        wednorm = wednorm/np.max(wednorm,axis=1)[:,np.newaxis]

        weds = np.sum(wednorm*np.sin(angles),axis=1)
        wedc = np.sum(wednorm*np.cos(angles),axis=1)
        pva_norm  = np.sqrt(weds**2+wedc**2)
        p0 = np.mean(pva_norm[pva_norm<np.percentile(pva_norm,10)])
        pva_norm = (pva_norm-p0)/p0



        ymn = np.mean(wedges,axis=1)
        y0 = np.mean(ymn[ymn<np.percentile(ymn,10)])
        ymn = (ymn-y0)/y0
        pva_z = pva/np.std(pva)
        pvan_z = pva_norm/np.std(pva_norm)
        ymn_z = ymn/np.std(ymn)
        
        return ymn_z,pvan_z,pva_z
                