# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 17:08:27 2025

@author: dowel

Description:
    Class object to conduct analysis of LAL recordings of PFL3 neurons
    



"""
import numpy as np
import pandas as pd
import analysis_funs.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt 
from analysis_funs.CX_imaging import CX
from Utils.utils_general import utils_general as ug
from scipy.optimize import minimize
#%%
class PFL3_analysis():
    def __init__(self,datadir):
        regions = ['LAL']
        d = datadir.split("\\")
        name = d[-3] + '_' + d[-2] + '_' + d[-1]
        cx = CX(name,regions,datadir)
        self.pv2, self.ft, self.ft2, ix = cx.load_postprocessing()
        
        
    def PFL3_function(self,heading,goal):
        """
        Simple PFL3 function abstracting anatomy.
        Input is heading and a singular goal
        
        """
        x = np.linspace(-np.pi,np.pi,16)
        xblock = np.ones((len(heading),len(x)))
        xblock = xblock*x.T
        goal = np.tile(goal,[16,1]).T
        heading = np.tile(heading,[16,1]).T
        
        gsignal = np.cos(goal+xblock)
        hsignal_L = np.cos(heading+xblock+np.pi/4)
        hsignal_R = np.cos(heading+xblock-np.pi/4)
        
        PFL3_L = np.exp(gsignal+hsignal_L)-1
        PFL3_R = np.exp(gsignal+hsignal_R)-1
        
        R_Lal = np.sum(PFL3_L,axis=1)/18.44
        L_Lal = np.sum(PFL3_R,axis= 1)/18.44
        return R_Lal,L_Lal
    def min_PFL3_function(self,goal,heading,RL,LL):
        predicted_R, predicted_L = self.PFL3_function(heading, np.array([goal]))
        error_R = np.sum((predicted_R - RL) ** 2)
        error_L = np.sum((predicted_L - LL) ** 2)
        return error_R + error_L
    
    def fit_PFL3(self):
        heading = self.ft2['ft_heading'].to_numpy()
        
        L = self.pv2['0_lal'].to_numpy()
        R = self.pv2['1_lal'].to_numpy()
        L[np.isnan(L)] = 0
        R[np.isnan(R)] = 0
        
        L = L#/np.max(L)
        R = R#/np.max(R)
        ydat = R-L
        ydat[np.isnan(R-L)] = 0
        infgoal = np.zeros_like(heading)
        print('Fitting')
        for i in range(len(infgoal)):
            result = minimize(self.min_PFL3_function,0,args=([heading[i]],[R[i]],[L[i]]),bounds=[(-np.pi,np.pi)])
            infgoal[i] = result.x[0]
        print('Done')
        self.infgoal = infgoal
        
    def plot_goal_arrows(self,a_sep= 5):
        heading = self.ft2['ft_heading'].to_numpy()
        goal = self.infgoal
        
        x = self.ft2['ft_posx'].to_numpy()
        y = self.ft2['ft_posy'].to_numpy()
        
        
        x,y = ug.fictrac_repair(x,y)
        instrip = self.ft2['instrip'].to_numpy()

        is1 =np.where(instrip)[0][0]
        dist = np.sqrt(x**2+y**2)
        dist = dist-dist[0]
        plt.figure()
        
        #x = x[is1:]
        #y = y[is1:]
        #dist = dist[is1:]
        
        #instrip = instrip[is1:]
        
        #goal = goal[is1:]
        #heading = heading[is1:]    
        plt.scatter(x[instrip>0],y[instrip>0],color=[0.6,0.6,0.6])
        
        
        plt.plot(x,y,color='k')
        t_sep = a_sep
        for i,d in enumerate(dist):
            if np.abs(d-t_sep)>a_sep:
                t_sep = d
                
                xa = 10*np.sin(heading[i])
                ya = 10*np.cos(heading[i])
                plt.arrow(x[i],y[i],xa,ya,length_includes_head=True,head_width=1,color=[0.1,0.1,0.1])
                
                xa = 10*np.sin(goal[i])
                ya = 10*np.cos(goal[i])
                plt.arrow(x[i],y[i],xa,ya,length_includes_head=True,head_width=1,color=[0.3,0.3,1])
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    