# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:07:51 2024

@author: dowel

Script goes through all neurons and outputs the results of regression modelling
and saves them
"""

from analysis_funs.regression import fci_regmodel
import os
import matplotlib.pyplot as plt 
from src.utilities import imaging as im
from skimage import io, data, registration, filters, measure
from scipy import signal as sg
from analysis_funs.CX_imaging import CX
import numpy as np
import pickle
import pandas as pd
from analysis_funs.CX_analysis_col import CX_a
from analysis_funs.CX_analysis_tan import CX_tan
#%% Initialise save directory and regression types
savedir = 'Y:\\Data\FCI\\ConsolidatedData\\Regression'
regchoice = ['odour onset', 'odour offset', 'in odour', 
                                 'cos heading pos','cos heading neg', 'sin heading pos', 'sin heading neg',
                                 'angular velocity pos','angular velocity neg','translational vel','ramp down since exit','ramp to entry']
#%% FB4X
datadirs = ["Y:\Data\FCI\Hedwig\\SS70711_FB4X\\240307\\f1\\Trial3",
            "Y:\Data\FCI\Hedwig\\SS70711_FB4X\\240313\\f1\\Trial3",
            "Y:\Data\FCI\Hedwig\\SS70711_FB4X\\240531\\f1\\Trial3"]

d_R2s = np.zeros((len(datadirs),len(regchoice)))
coeffs = np.zeros((len(datadirs),len(regchoice)))
rsq = np.zeros(len(datadirs))
for i, d in enumerate(datadirs):
    
    dspl = d.split("\\")
    name = dspl[-3] + '_' + dspl[-2] + '_' + dspl[-1]
    cxt = CX_tan(d)
    pv2 = cxt.pv2
    ft2 = cxt.ft2
    fc = fci_regmodel(pv2[['0_fsbtn']].to_numpy().flatten(),ft2,pv2)
    fc.rebaseline(span=500,plotfig=False)
    fc.run(regchoice)
    fc.run_dR2(20,fc.xft)
    d_R2s[i,:] = fc.dR2_mean
    coeffs[i,:] = fc.coeff_cv[:-1]
    rsq[i] = fc.cvR2
    
    
    
    trj,ca = cxt.mean_traj_nF()
    if i==0:
        T = np.expand_dims(trj,2)
        C = np.expand_dims(ca,1)
    else:
        T = np.append(T,np.expand_dims(trj,2),axis=2)
        C = np.append(C,np.expand_dims(ca,1),axis=1)
    

output_dR2 = (-d_R2s.T)*np.sign(coeffs.T)

savedict = {'deltaR2_norm':output_dR2,'cvR2':rsq,'regnames':regchoice,'mean_traj':T,'mean_Ca':C}
savename = os.path.join(savedir,'FB4X.pkl')
with open(savename,'wb') as fp:
    pickle.dump(savedict,fp)
#%% FB4R
datadirs = ['Y:\\Data\FCI\\AndyData\\47H09_GDrive\\47H09\\20220128_47H09jGCaMP7f_Fly1-002\\processed\\processed',
         #'Y:\\Data\FCI\\AndyData\\47H09_GDrive\\47H09\\20220130_47H09sytjGCaMP7f_Fly2-001\\processed\\processed',
         'Y:\\Data\FCI\\AndyData\\47H09_GDrive\\47H09\\20220130_47H09sytjGCaMP7f_Fly1-001\\processed\\processed',
         'Y:\\Data\FCI\\AndyData\\47H09_GDrive\\47H09\\20220128_47H09jGCaMP7f_Fly2-001\\processed\\processed',
         'Y:\\Data\FCI\\AndyData\\47H09_GDrive\\47H09\\20220127_47H09jGCaMP7f_Fly1-001\\processed\\processed'
         ]
d_R2s = np.zeros((len(datadirs),len(regchoice)))
coeffs = np.zeros((len(datadirs),len(regchoice)))
rsq = np.zeros(len(datadirs))
for i, d in enumerate(datadirs):
    #post_processing_file = os.path.join(d,'postprocessing.h5')
    #pv2 = pd.read_hdf(post_processing_file, 'pv2')
    #ft2 = pd.read_hdf(post_processing_file, 'ft2')
    cxt = CX_tan(d,tnstring='fb4r_dff',Andy=True)
    pv2 = cxt.pv2
    ft2 = cxt.ft2
    fc = fci_regmodel(pv2['fb4r_dff'].to_numpy(),ft2,pv2)
    fc.rebaseline(span=500,plotfig=False)
    fc.run(regchoice)
    fc.run_dR2(20,fc.xft)
    d_R2s[i,:] = fc.dR2_mean
    coeffs[i,:] = fc.coeff_cv[:-1]
    rsq[i] = fc.cvR2
    
    trj,ca = cxt.mean_traj_nF()
    if i==0:
        T = np.expand_dims(trj,2)
        C = np.expand_dims(ca,1)
    else:
        T = np.append(T,np.expand_dims(trj,2),axis=2)
        C = np.append(C,np.expand_dims(ca,1),axis=1)
    

output_dR2 = (-d_R2s.T)*np.sign(coeffs.T)

savedict = {'deltaR2_norm':output_dR2,'cvR2':rsq,'regnames':regchoice,'mean_traj':T,'mean_Ca':C}
savename = os.path.join(savedir,'FB4R.pkl')
with open(savename,'wb') as fp:
    pickle.dump(savedict,fp)
#%% FB4P_b
datadirs = ["Y:\\Data\\FCI\\Hedwig\\FB4P_b_SS67631\\240720\\f1\\Trial3"]
d_R2s = np.zeros((len(datadirs),len(regchoice)))
coeffs = np.zeros((len(datadirs),len(regchoice)))
rsq = np.zeros(len(datadirs))
for i, d in enumerate(datadirs):
    cxt = CX_tan(d)
    pv2 = cxt.pv2
    ft2 = cxt.ft2
    fc = fci_regmodel(pv2[['0_fsbtn']].to_numpy().flatten(),ft2,pv2)
    fc.rebaseline(span=500,plotfig=False)
    fc.run(regchoice)
    fc.run_dR2(20,fc.xft)
    d_R2s[i,:] = fc.dR2_mean
    coeffs[i,:] = fc.coeff_cv[:-1]
    rsq[i] = fc.cvR2
    
    
    
    trj,ca = cxt.mean_traj_nF()
    if i==0:
        T = np.expand_dims(trj,2)
        C = np.expand_dims(ca,1)
    else:
        T = np.append(T,np.expand_dims(trj,2),axis=2)
        C = np.append(C,np.expand_dims(ca,1),axis=1)
    

output_dR2 = (-d_R2s.T)*np.sign(coeffs.T)

savedict = {'deltaR2_norm':output_dR2,'cvR2':rsq,'regnames':regchoice,'mean_traj':T,'mean_Ca':C}
savename = os.path.join(savedir,'FB4P_b.pkl')
with open(savename,'wb') as fp:
    pickle.dump(savedict,fp)
#%% FB5AB
datadirs =['Y:\\Data\\FCI\\AndyData\\21D07\\20220103_21D07_sytjGCaMP7f_Fly1_001\\processed',
        'Y:\\Data\\FCI\\AndyData\\21D07\\20220107_21D07_sytjGCaMP7f_Fly1_001\\processed',
        'Y:\\Data\\FCI\\AndyData\\21D07\\20220110_21D07_sytjGCaMP7f_Fly1_001\\processed',
        'Y:\\Data\\FCI\\AndyData\\21D07\\20220111_21D07_sytjGCaMP7f_Fly1_001\\processed']
d_R2s = np.zeros((len(datadirs),len(regchoice)))
coeffs = np.zeros((len(datadirs),len(regchoice)))
rsq = np.zeros(len(datadirs))
for i, d in enumerate(datadirs):
    cxt = CX_tan(d,tnstring='fb5ab_dff',Andy=True)
    pv2 = cxt.pv2
    ft2 = cxt.ft2
    fc = fci_regmodel(pv2['fb5ab_dff'].to_numpy(),ft2,pv2)
    fc.rebaseline(span=500,plotfig=False)
    fc.run(regchoice)
    fc.run_dR2(20,fc.xft)
    d_R2s[i,:] = fc.dR2_mean
    coeffs[i,:] = fc.coeff_cv[:-1]
    rsq[i] = fc.cvR2
    
    trj,ca = cxt.mean_traj_nF()
    if i==0:
        T = np.expand_dims(trj,2)
        C = np.expand_dims(ca,1)
    else:
        T = np.append(T,np.expand_dims(trj,2),axis=2)
        C = np.append(C,np.expand_dims(ca,1),axis=1)
    
    

output_dR2 = (-d_R2s.T)*np.sign(coeffs.T)

savedict = {'deltaR2_norm':output_dR2,'cvR2':rsq,'regnames':regchoice,'mean_traj':T,'mean_Ca':C}
savename = os.path.join(savedir,'FB5AB.pkl')
with open(savename,'wb') as fp:
    pickle.dump(savedict,fp)
#%% FB5I
datadirs = [
    "Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553\\240523\\f1\\Trial4",
    "Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553\\240628\\f1\\Trial2"]
d_R2s = np.zeros((len(datadirs),len(regchoice)))
coeffs = np.zeros((len(datadirs),len(regchoice)))
rsq = np.zeros(len(datadirs))
for i, d in enumerate(datadirs):
    cxt = CX_tan(d)
    pv2 = cxt.pv2
    ft2 = cxt.ft2
    fc = fci_regmodel(pv2[['0_fsbtn']].to_numpy().flatten(),ft2,pv2)
    fc.rebaseline(span=500,plotfig=False)
    fc.run(regchoice)
    fc.run_dR2(20,fc.xft)
    d_R2s[i,:] = fc.dR2_mean
    coeffs[i,:] = fc.coeff_cv[:-1]
    rsq[i] = fc.cvR2
    
    
    
    trj,ca = cxt.mean_traj_nF()
    if i==0:
        T = np.expand_dims(trj,2)
        C = np.expand_dims(ca,1)
    else:
        T = np.append(T,np.expand_dims(trj,2),axis=2)
        C = np.append(C,np.expand_dims(ca,1),axis=1)
    

output_dR2 = (-d_R2s.T)*np.sign(coeffs.T)

savedict = {'deltaR2_norm':output_dR2,'cvR2':rsq,'regnames':regchoice,'mean_traj':T,'mean_Ca':C}
savename = os.path.join(savedir,'FB5I.pkl')
with open(savename,'wb') as fp:
    pickle.dump(savedict,fp)
#%% FB4M
datadirs = ['Y:\\Data\\FCI\\AndyData\\FB4M_jGCaMP7f\\20220510_FB4MjGCaMP7f_Fly1_001\\processed',
        'Y:\\Data\\FCI\\AndyData\\FB4M_jGCaMP7f\\20220510_FB4MjGCaMP7f_Fly2_002\\processed']
d_R2s = np.zeros((len(datadirs),len(regchoice)))
coeffs = np.zeros((len(datadirs),len(regchoice)))
rsq = np.zeros(len(datadirs))
for i, d in enumerate(datadirs):
    cxt = CX_tan(d,tnstring='fb4m_dff',Andy=True)
    pv2 = cxt.pv2
    ft2 = cxt.ft2
    fc = fci_regmodel(pv2['fb4m_dff'].to_numpy(),ft2,pv2)
    fc.rebaseline(span=500,plotfig=False)
    fc.run(regchoice)
    fc.run_dR2(20,fc.xft)
    d_R2s[i,:] = fc.dR2_mean
    coeffs[i,:] = fc.coeff_cv[:-1]
    rsq[i] = fc.cvR2
    
    trj,ca = cxt.mean_traj_nF()
    if i==0:
        T = np.expand_dims(trj,2)
        C = np.expand_dims(ca,1)
    else:
        T = np.append(T,np.expand_dims(trj,2),axis=2)
        C = np.append(C,np.expand_dims(ca,1),axis=1)
    
    

output_dR2 = (-d_R2s.T)*np.sign(coeffs.T)

savedict = {'deltaR2_norm':output_dR2,'cvR2':rsq,'regnames':regchoice,'mean_traj':T,'mean_Ca':C}
savename = os.path.join(savedir,'FB4M.pkl')
with open(savename,'wb') as fp:
    pickle.dump(savedict,fp)
#%% hDeltaB
datadirs = ["Y:\Data\FCI\AndyData\hdb\\20220406_Fly2_001"]
coeffs = np.zeros((len(datadirs),len(regchoice)))
rsq = np.zeros(len(datadirs))
for ir, datadir in enumerate(datadirs):
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    cxa = CX_a(datadir,Andy='hDeltaB')
       
    y = np.mean(cxa.pdat['wedges_offset_fb'],axis=1)
    ft2 = cxa.ft2
    pv2 = cxa.pv2
    fc = fci_regmodel(y,ft2,pv2)
    fc.run(regchoice)
    fc.run_dR2(20,fc.xft)
    d_R2s[ir,:] = fc.dR2_mean
    coeffs[ir,:] = fc.coeff_cv[:-1]
    rsq[ir] = fc.cvR2
    
output_dR2 = (-d_R2s.T)*np.sign(coeffs.T)

savedict = {'deltaR2_norm':output_dR2,'cvR2':rsq,'regnames':regchoice}
savename = os.path.join(savedir,'hDeltaB.pkl')
with open(savename,'wb') as fp:
    pickle.dump(savedict,fp)
#%% hDeltaC
rootdir = 'Y:\\Data\\FCI\\AndyData\\hDeltaC_imaging\\csv'
datadirs = ['20220517_hdc_split_60d05_sytgcamp7f',
 '20220627_hdc_split_Fly1',
 '20220627_hdc_split_Fly2',
 '20220628_HDC_sytjGCaMP7f_Fly1',
 #'20220628_HDC_sytjGCaMP7f_Fly1_45-004', 45 degree plume
 '20220629_HDC_split_sytjGCaMP7f_Fly1',
 '20220629_HDC_split_sytjGCaMP7f_Fly3']


d_R2s = np.zeros((len(datadirs),len(regchoice)))
coeffs = np.zeros((len(datadirs),len(regchoice)))
rsq = np.zeros(len(datadirs))
for ir, ddir in enumerate(datadirs):
    datadir = os.path.join(rootdir,ddir,"et")
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    cxa = CX_a(datadir,Andy='hDeltaC')
    y = np.mean(cxa.pdat['wedges_fsb_upper'],axis=1)
    ft2 = cxa.ft2
    pv2 = cxa.pv2
    fc = fci_regmodel(y,ft2,pv2)
    fc.run(regchoice)
    fc.run_dR2(20,fc.xft)
    d_R2s[ir,:] = fc.dR2_mean
    coeffs[ir,:] = fc.coeff_cv[:-1]
    rsq[ir] = fc.cvR2
output_dR2 = (-d_R2s.T)*np.sign(coeffs.T)

savedict = {'deltaR2_norm':output_dR2,'cvR2':rsq,'regnames':regchoice}
savename = os.path.join(savedir,'hDeltaC.pkl')
with open(savename,'wb') as fp:
    pickle.dump(savedict,fp)
#%% hDeltaJ
datadirs= [                   "Y:\\Data\\FCI\\Hedwig\\hDeltaJ\\240529\\f1\\Trial3"
                   ]
d_R2s = np.zeros((len(datadirs),len(regchoice)))
coeffs = np.zeros((len(datadirs),len(regchoice)))
rsq = np.zeros(len(datadirs))
for ir, datadir in enumerate(datadirs):
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
       
    y = np.mean(cxa.pdat['wedges_offset_fsb_upper'],axis=1)
    ft2 = cxa.ft2
    pv2 = cxa.pv2
    fc = fci_regmodel(y,ft2,pv2)
    fc.run(regchoice)
    fc.run_dR2(20,fc.xft)
    d_R2s[ir,:] = fc.dR2_mean
    coeffs[ir,:] = fc.coeff_cv[:-1]
    rsq[ir] = fc.cvR2
    
output_dR2 = (-d_R2s.T)*np.sign(coeffs.T)

savedict = {'deltaR2_norm':output_dR2,'cvR2':rsq,'regnames':regchoice}
savename = os.path.join(savedir,'hDeltaJ.pkl')
with open(savename,'wb') as fp:
    pickle.dump(savedict,fp)
#%% FC2
datadirs = [
    "Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f1\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f2\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240502\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240514\\f1\\Trial2"]
d_R2s = np.zeros((len(datadirs),len(regchoice)))
coeffs = np.zeros((len(datadirs),len(regchoice)))
rsq = np.zeros(len(datadirs))
for ir, datadir in enumerate(datadirs):
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
       
    y = np.mean(cxa.pdat['wedges_offset_fsb_upper'],axis=1)
    ft2 = cxa.ft2
    pv2 = cxa.pv2
    fc = fci_regmodel(y,ft2,pv2)
    fc.run(regchoice)
    fc.run_dR2(20,fc.xft)
    d_R2s[ir,:] = fc.dR2_mean
    coeffs[ir,:] = fc.coeff_cv[:-1]
    rsq[ir] = fc.cvR2
    
output_dR2 = (-d_R2s.T)*np.sign(coeffs.T)

savedict = {'deltaR2_norm':output_dR2,'cvR2':rsq,'regnames':regchoice}
savename = os.path.join(savedir,'FC2.pkl')
with open(savename,'wb') as fp:
    pickle.dump(savedict,fp)
#%% 
import pandas as pd
pd.read_pickle(savename)
#%% Plot mean of selected neurons

def plot_dr2(neurons,colourindex=[False]):
    colours = np.array([[228,26,28],
    [55,126,184],
    [77,175,74],
    [152,78,163],
    [255,127,0],
    [166,86,40],
    [247,129,191],
    [153,153,153]])/255
    for i,n in enumerate(neurons):
        savename = os.path.join(savedir,n+ '.pkl')
        dat = pd.read_pickle(savename)
        dr2 = dat['deltaR2_norm']/dat['cvR2']
        dmn = np.mean(dr2,axis=1)
        plt.figure(1)
        if colourindex[0]==False:
            plt.plot(dmn,color=colours[i,:])
        else:
            plt.plot(dmn,color=colours[colourindex[i],:])
        plt.figure(2)
        dr2 = dat['deltaR2_norm']
        dmn = np.mean(dr2,axis=1)
        if colourindex[0]==False:
            plt.plot(dmn,color=colours[i,:])
        else:
            plt.plot(dmn,color=colours[colourindex[i],:])
    for i in range(0,2):
        plt.figure(i+1)
        plt.xticks(np.arange(0,len(dmn)),labels=dat['regnames'],rotation=70)
        plt.legend(neurons,loc='upper left')
        if i==0:
            plt.ylabel('normalised signed dR2')
        else:
            plt.ylabel('signed dR2')
        
        plt.subplots_adjust(bottom=0.4)
#%% All TNs
neurons = ['FB5I','FB4X','FB4P_b','FB4R','FB5AB','FB4M']
plot_dr2(neurons)
ylims = np.array([[-0.4,0.4],[-0.17,0.17]])
for i in range(0,2):
    plt.figure(i+1)
    plt.ylim(ylims[i,:])
    plt.savefig(os.path.join(savedir,'AllTNs.png'))
    plt.savefig(os.path.join(savedir,'AllTNs.pdf'))
#%% Turners
plt.close('all')
neurons = ['FB5I','FB4X']
plot_dr2(neurons)
ylims = np.array([[-0.4,0.4],[-0.17,0.17]])
for i in range(0,2):
    plt.figure(i+1)
    plt.ylim(ylims[i,:])
    plt.savefig(os.path.join(savedir,'TurnTNs.png'))
    plt.savefig(os.path.join(savedir,'TurnTNs.pdf'))
#%% Straight walkers
plt.close('all')
neurons = ['FB4P_b','FB4R','FB5AB','FB4M']
colourindex= np.arange(2,6)
plot_dr2(neurons,colourindex)
ylims = np.array([[-0.4,0.4],[-0.17,0.17]])
for i in range(0,2):
    plt.figure(i+1)
    plt.ylim(ylims[i,:])
    plt.savefig(os.path.join(savedir,'StraightTNs.png'))
    plt.savefig(os.path.join(savedir,'StraightTNs.pdf'))
#%% Plot trajs
import matplotlib as mpl
from matplotlib import cm
def plot_trajs(neurons,cmx):
    plt.figure(figsize=(10,10))
    xoffset = 0
    c_map = plt.get_cmap('coolwarm')
    cnorm = mpl.colors.Normalize(vmin=-cmx, vmax=cmx)
    scalarMap = cm.ScalarMappable(cnorm, c_map)
    offsets = np.array(xoffset)
    for i,n in enumerate(neurons):
        savename = os.path.join(savedir,n+ '.pkl')
        dat = pd.read_pickle(savename)
        ca = dat['mean_Ca']
        cas = np.shape(ca)
        if len(cas)>1:
            neurons[i] = n+' (n=' +str(np.shape(ca)[1]) +')'
        else:
            neurons[i] = n+' (n=1)'
        traj = dat['mean_traj']
        ca = ca/np.std(ca,axis=0)
        mca = np.mean(ca,axis=1)
        colour = mca
        c_map_rgb = scalarMap.to_rgba(colour)
        mtraj = np.mean(traj,axis=2)
        yrange = np.array([min(mtraj[:,1]),max(mtraj[:,1])])
        plt.fill([-5+xoffset,5+xoffset,5+xoffset,-5+xoffset],yrange[[0,0,1,1]],color=[0.7,0.7,0.7])
        for i in range(len(ca)-1):
            x = mtraj[i:i+2,0]
            y = mtraj[i:i+2,1]
            #ca = np.mean(ca[i:i+2])
            plt.plot(x+xoffset,y,color=c_map_rgb[i,:])
        xoffset = xoffset+25
        offsets = np.append(offsets,xoffset)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.xticks(offsets[:-1],labels=neurons)
    plt.show()
neurons = ['FB5I','FB4X','FB4P_b','FB4R','FB5AB','FB4M']
plot_trajs(neurons,cmx=2)
plt.savefig(os.path.join(savedir,'TN_mean_Traj.png'))
plt.savefig(os.path.join(savedir,'TN_mean_Traj.pdf'))

#%%

fc.plot_mean_flur('odour_onset')
fc.plot_example_flur()

plt.figure()
plt.plot(d_R2s.T,color='k')
plt.plot([0,len(regchoice)],[0,0],color='k',linestyle='--')
plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('delta R2')
plt.xlabel('Regressor name')
plt.show()
#plt.savefig(os.path.join(savedir,'dR2.png'))

plt.figure()
plt.plot(-d_R2s.T*np.sign(coeffs.T),color='k')
plt.plot([0,len(regchoice)],[0,0],color='k',linestyle='--')
plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('delta R2* sign(coeffs)')
plt.xlabel('Regressor name')
plt.show()
#plt.savefig(os.path.join(savedir,'dR2_mult_coeff.png'))

plt.figure()
plt.plot(coeffs.T,color='k')
plt.plot([0,len(regchoice)],[0,0],color='k',linestyle='--')
plt.xticks(np.arange(0,len(regchoice)),labels=regchoice,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.ylabel('Coefficient weight')
plt.xlabel('Regressor name')
plt.show()
#plt.savefig(os.path.join(savedir,'Coeffs.png'))

plt.figure()
plt.scatter(rsq_t_t[:,0],rsq_t_t[:,1],color='k')
plt.plot([np.min(rsq_t_t[:]),np.max(rsq_t_t[:])], [np.min(rsq_t_t[:]),np.max(rsq_t_t[:])],color='k',linestyle='--' )
plt.xlabel('R2 pre air')
plt.ylabel('R2 live air')
plt.title('Model trained on pre air period')