# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:00:04 2023

@author: dowel
"""
#%% Plan
# just load some of Andy's processed data and see what it looks like


#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import h5py
from matplotlib import cm
import matplotlib as mpl
from analysis_funs.regression import fci_regmodel



#%% Example data load
#datadir = 'Y:\\Data\\FCI\\AndyData\\21D07\\20220103_21D07_sytjGCaMP7f_Fly1_001\\processed'
datadir = 'Y:\\Data\FCI\\AndyData\\47H09_GDrive\\47H09\\20220128_47H09jGCaMP7f_Fly1-002\\processed\\processed'
#datadir = 'Y:\\Data\\FCI\\AndyData\\FB4M_jGCaMP7f\\20220510_FB4MjGCaMP7f_Fly2_002\\processed'
post_processing_file = os.path.join(datadir,'postprocessing.h5')
pv2 = pd.read_hdf(post_processing_file, 'pv2')
ft = pd.read_hdf(post_processing_file, 'ft')
ft2 = pd.read_hdf(post_processing_file, 'ft2')

ix = pd.read_hdf(post_processing_file, 'ix')
#%% plot example plume experiment
#%%
colour = pv2['fb4r_dff']
x = ft2['ft_posx']
y = ft2['ft_posy']
xrange = np.max(x)-np.min(x)
yrange = np.max(y)-np.min(y)
mrange = np.max([xrange,yrange])+100
ylims = [np.median(y)-mrange/2, np.median(y)+mrange/2]
xlims = [np.median(x)-mrange/2, np.median(x)+mrange/2]

acv = ft2['mfc2_stpt']
inplume = acv>0
c_map = plt.get_cmap('coolwarm')

cmax = np.round(np.percentile(colour[~np.isnan(colour)],95),decimals=1)
cnorm = mpl.colors.Normalize(vmin=0, vmax=cmax)
scalarMap = cm.ScalarMappable(cnorm, c_map)
c_map_rgb = scalarMap.to_rgba(colour)
x = x-x[0]
y = y -y[0]
fig = plt.figure(figsize=(15,15))

ax = fig.add_subplot(111)
ax.scatter(x[inplume],y[inplume],color=[0.5, 0.5, 0.5])
for i in range(len(x)-1):
    ax.plot(x[i:i+2],y[i:i+2],color=c_map_rgb[i+1,:3])
plt.xlim(xlims)
plt.ylim(ylims)
plt.xlabel('x position (mm)')
plt.ylabel('y position (mm)')
plt.show()
#%% Linear regression model to understand activity %%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%
#%% Test with toy data - looks like it is a bit off
from analysis_funs.regression import fci_regmodel
fc = fci_regmodel(pv2['fb4r_dff'],ft2,pv2)
regchoice = ['odour onset', 'odour offset', 'in odour', 
                                'cos heading pos','cos heading neg', 'sin heading pos', 'sin heading neg',
                                'angular velocity pos','x pos','x neg','y pos', 'y neg','ramp down since exit','ramp to entry'] 
x = fc.set_up_regressors(regchoice)
fc.run(regchoice)

fc.plot_example_flur()
fc.plot_flur_w_regressors(['in odour'])
fc.example_trajectory()
fc.plot_all_regressors(regchoice)
#%% Run regression model by all fb4r neurons
savedir = "Y:\Data\FCI\FCI_summaries\TangentialRegression"
savedir_n = "Y:\Data\FCI\FCI_summaries\TangentialRegression\FB4R"
plt.close('all')
regchoice = ['odour onset', 'odour offset', 'in odour', 
                                'cos heading pos','cos heading neg', 'sin heading pos', 'sin heading neg',
                                'angular velocity pos','x pos','x neg','y pos', 'y neg','ramp down since exit','ramp to entry'] 

flies = ['Y:\\Data\FCI\\AndyData\\47H09_GDrive\\47H09\\20220128_47H09jGCaMP7f_Fly1-002\\processed\\processed',
         #'Y:\\Data\FCI\\AndyData\\47H09_GDrive\\47H09\\20220130_47H09sytjGCaMP7f_Fly2-001\\processed\\processed',
         'Y:\\Data\FCI\\AndyData\\47H09_GDrive\\47H09\\20220130_47H09sytjGCaMP7f_Fly1-001\\processed\\processed',
         'Y:\\Data\FCI\\AndyData\\47H09_GDrive\\47H09\\20220128_47H09jGCaMP7f_Fly2-001\\processed\\processed',
         'Y:\\Data\FCI\\AndyData\\47H09_GDrive\\47H09\\20220127_47H09jGCaMP7f_Fly1-001\\processed\\processed'
         ]
weights = np.zeros([len(flies),len(regchoice)],dtype='float')
weights_c = np.zeros([len(flies),len(regchoice)+1],dtype='float')
r2 = np.zeros(len(flies),dtype='float')
p = np.zeros([len(flies),len(regchoice)],dtype='float')
for i,f in enumerate(flies):
    print('Fly',i)
    post_processing_file = os.path.join(f,'postprocessing.h5')
    pv2 = pd.read_hdf(post_processing_file, 'pv2')
    ft2 = pd.read_hdf(post_processing_file, 'ft2')
    fc = fci_regmodel(pv2['fb4r_dff'],ft2,pv2)
    fc.run(regchoice)
    fc.run_dR2(20,fc.xft)
    weights_c[i,:] = fc.coeff_cv
    weights[i,:] = fc.dR2_mean
    r2[i] = fc.r2
    p[i,:] = fc.dR2_ttest
    
    # Plot example fluorescence and fit
    fc.plot_example_flur()
    plt.savefig(os.path.join(savedir_n,'Example_Fit_F' + str(i) + '.png'))
    plt.rcParams['pdf.fonttype'] = 42 
    plt.savefig(os.path.join(savedir_n,'Example_Fit_F' + str(i) + '.pdf'))
    plt.xlim([0,200])
    plt.savefig(os.path.join(savedir_n,'Example_Fit_F' + str(i) + '_200secs.png'))
    # Plot in and outside odour
    fc.plot_flur_w_regressors(['in odour','ramp down since exit'])
    plt.savefig(os.path.join(savedir_n,'Example_F' + str(i) + '.png'))
    
    plt.xlim([0,200])
    plt.savefig(os.path.join(savedir_n,'Example_F' + str(i) + '_200secs.png'))
    
    plt.close('all')
    fc.example_trajectory()
    plt.savefig(os.path.join(savedir_n,'Trajectory_F' + str(i) + '_.png'))
    plt.rcParams['pdf.fonttype'] = 42 
    plt.rcParams['ps.fonttype'] = 42 
    plt.savefig(os.path.join(savedir_n,'Trajectory_F' + str(i) + '_.eps'), format='eps')
    plt.rcParams['pdf.fonttype'] = 42 
    plt.savefig(os.path.join(savedir_n,'Trajectory_F' + str(i) + '_.pdf'))
    plt.savefig(os.path.join(savedir_n,'Trajectory_F' + str(i) + '_.svg'))
    fc.plot_all_regressors(regchoice)
    plt.savefig(os.path.join(savedir_n,'All_regressors' + str(i) + '_.png'))
    plt.rcParams['pdf.fonttype'] = 42
    plt.savefig(os.path.join(savedir_n,'All_regressors' + str(i) + '_.pdf'))
    plt.close('all')
    
    
    
colours = np.array([[247,251,255],
[222,235,247],
[198,219,239],
[158,202,225],
[107,174,214],
[66,146,198],
[33,113,181],
[8,81,156],
[8,48,107]],dtype='float')/255
plt.figure()
for i in range(len(colours)):
    plt.plot([-1, 1],[i,i],color=colours[i,:])
    
plt.yticks(np.linspace(0,8,9),labels = (np.linspace(0,8,9)+1)/10)
plt.xticks([])
plt.ylabel('R squared')
plt.show()
plt.savefig(os.path.join(savedir,'R2_legend_.pdf'))
r2dx = np.round(r2*10).astype(int)



plt.figure()
plt.plot([0, 14],[0, 0],color='k',linestyle='--')   
for a in range(len(r2)):
    plt.plot(np.linspace(0,len(regchoice),len(regchoice)),np.transpose(weights_c[a,:-1]),color=colours[r2dx[a],:])
wm = np.mean(weights_c,axis=0)
plt.plot(np.linspace(0,len(regchoice),len(regchoice)),wm[:-1],color='k',linewidth=2)
plt.xticks(np.linspace(0,len(regchoice),len(regchoice)),labels=regchoice,rotation=45,ha='right')
plt.subplots_adjust(bottom=0.4)
plt.ylabel('beta weights')
plt.title('FB4R')
plt.savefig(os.path.join(savedir,'FB4R_weights.png'))
plt.rcParams['pdf.fonttype'] = 42 
plt.savefig(os.path.join(savedir,'FB4R_weights.pdf'))
plt.show()

plt.figure()
plt.plot([0, 14],[0, 0],color='k',linestyle='--')   
for a in range(len(r2)):
    plt.plot(np.linspace(0,len(regchoice),len(regchoice)),np.transpose(weights[a,:]),color=colours[r2dx[a],:])
wm = np.mean(weights,axis=0)
plt.plot(np.linspace(0,len(regchoice),len(regchoice)),wm[:],color='k',linewidth=2)
plt.xticks(np.linspace(0,len(regchoice),len(regchoice)),labels=regchoice,rotation=45,ha='right')
plt.subplots_adjust(bottom=0.4)
plt.ylabel('delta R2')
plt.title('FB4R')
plt.show()
plt.savefig(os.path.join(savedir,'FB4R_dR2.png'))
plt.rcParams['pdf.fonttype'] = 42 
plt.savefig(os.path.join(savedir,'FB4R_dR2.pdf'))
plt.figure()
plt.plot([0, 14],[0.05, 0.05],color='k',linestyle='--')   
for a in range(len(r2)):
    plt.plot(np.linspace(0,len(regchoice),len(regchoice)),np.transpose(p[a,:]),color=colours[r2dx[a],:])
    
plt.xticks(np.linspace(0,len(regchoice),len(regchoice)),labels=regchoice,rotation=45,ha='right')
plt.subplots_adjust(bottom=0.4)
plt.yscale('log')
plt.show()
plt.ylabel('p Value')
plt.title('FB4R')
plt.savefig(os.path.join(savedir,'FB4R_p_val.png'))
plt.rcParams['pdf.fonttype'] = 42 
plt.savefig(os.path.join(savedir,'FB4R_p_val.pdf'))
#%% Run regression model by all fb5AB neurons
plt.close('all')
savedir_n = "Y:\Data\FCI\FCI_summaries\TangentialRegression\FB5AB"
savedir = "Y:\Data\FCI\FCI_summaries\TangentialRegression"
regchoice = ['odour onset', 'odour offset', 'in odour', 
                                'cos heading pos','cos heading neg', 'sin heading pos', 'sin heading neg',
                                'angular velocity pos','x pos','x neg','y pos', 'y neg','ramp down since exit','ramp to entry']

flies =['Y:\\Data\\FCI\\AndyData\\21D07\\20220103_21D07_sytjGCaMP7f_Fly1_001\\processed',
        'Y:\\Data\\FCI\\AndyData\\21D07\\20220107_21D07_sytjGCaMP7f_Fly1_001\\processed',
        'Y:\\Data\\FCI\\AndyData\\21D07\\20220110_21D07_sytjGCaMP7f_Fly1_001\\processed',
        'Y:\\Data\\FCI\\AndyData\\21D07\\20220111_21D07_sytjGCaMP7f_Fly1_001\\processed']

weights = np.zeros([len(flies),len(regchoice)],dtype='float')
weights_c = np.zeros([len(flies),len(regchoice)+1],dtype='float')
r2 = np.zeros(len(flies),dtype='float')
p = np.zeros([len(flies),len(regchoice)],dtype='float')
for i,f in enumerate(flies):
    print('Fly',i)
    post_processing_file = os.path.join(f,'postprocessing.h5')
    pv2 = pd.read_hdf(post_processing_file, 'pv2')
    ft2 = pd.read_hdf(post_processing_file, 'ft2')
    fc = fci_regmodel(pv2['fb5ab_dff'],ft2,pv2)
    fc.run(regchoice)
    fc.run_dR2(20,fc.xft)
    weights_c[i,:] = fc.coeff_cv
    weights[i,:] = fc.dR2_mean
    r2[i] = fc.r2
    p[i,:] = fc.dR2_ttest
    
    # Plot example fluorescence and fit
    fc.plot_example_flur()
    plt.savefig(os.path.join(savedir_n,'Example_Fit_F' + str(i) + '.png'))
    plt.rcParams['pdf.fonttype'] = 42 
    plt.savefig(os.path.join(savedir_n,'Example_Fit_F' + str(i) + '.pdf'))
    plt.xlim([0,200])
    plt.savefig(os.path.join(savedir_n,'Example_Fit_F' + str(i) + '_200secs.png'))
    # Plot in and outside odour
    fc.plot_flur_w_regressors(['in odour','ramp down since exit'])
    plt.savefig(os.path.join(savedir_n,'Example_F' + str(i) + '.png'))
    
    plt.xlim([0,200])
    plt.savefig(os.path.join(savedir_n,'Example_F' + str(i) + '_200secs.png'))
    
    plt.close('all')
    fc.example_trajectory()
    plt.savefig(os.path.join(savedir_n,'Trajectory_F' + str(i) + '_.png'))
    plt.rcParams['pdf.fonttype'] = 42 
    plt.rcParams['ps.fonttype'] = 42 
    plt.savefig(os.path.join(savedir_n,'Trajectory_F' + str(i) + '_.eps'), format='eps')
    plt.rcParams['pdf.fonttype'] = 42 
    plt.savefig(os.path.join(savedir_n,'Trajectory_F' + str(i) + '_.pdf'))
    plt.savefig(os.path.join(savedir_n,'Trajectory_F' + str(i) + '_.svg'))
    fc.plot_all_regressors(regchoice)
    plt.savefig(os.path.join(savedir_n,'All_regressors' + str(i) + '_.png'))
    plt.rcParams['pdf.fonttype'] = 42
    plt.savefig(os.path.join(savedir_n,'All_regressors' + str(i) + '_.pdf'))
    plt.close('all')
    
colours = np.array([[247,251,255],
[222,235,247],
[198,219,239],
[158,202,225],
[107,174,214],
[66,146,198],
[33,113,181],
[8,81,156],
[8,48,107]],dtype='float')/255
r2dx = np.round(r2*10).astype(int)


plt.figure()
plt.plot([0, 14],[0, 0],color='k',linestyle='--')   
for a in range(len(r2)):
    plt.plot(np.linspace(0,len(regchoice),len(regchoice)),np.transpose(weights_c[a,:-1]),color=colours[r2dx[a],:])
wm = np.mean(weights_c,axis=0)
plt.plot(np.linspace(0,len(regchoice),len(regchoice)),wm[:-1],color='k',linewidth=2)
plt.xticks(np.linspace(0,len(regchoice),len(regchoice)),labels=regchoice,rotation=45,ha='right')
plt.subplots_adjust(bottom=0.4)
plt.ylabel('beta weights')
plt.title('FB5AB')
plt.savefig(os.path.join(savedir,'FB5AB_weights.png'))
plt.show()


plt.figure()
plt.plot([0, 14],[0, 0],color='k',linestyle='--')   
for a in range(len(r2)):
    plt.plot(np.linspace(0,len(regchoice),len(regchoice)),np.transpose(weights[a,:]),color=colours[r2dx[a],:])
wm = np.mean(weights,axis=0)
plt.plot(np.linspace(0,len(regchoice),len(regchoice)),wm[:],color='k',linewidth=2)
plt.xticks(np.linspace(0,len(regchoice),len(regchoice)),labels=regchoice,rotation=45,ha='right')
plt.subplots_adjust(bottom=0.4)
plt.ylabel('delta R2')
plt.title('FB5AB')
plt.show()


plt.savefig(os.path.join(savedir,'FB5AB_dR2.png'))

plt.figure()
plt.plot([0, 14],[0.05, 0.05],color='k',linestyle='--')   
for a in range(len(r2)):
    plt.plot(np.linspace(0,len(regchoice),len(regchoice)),np.transpose(p[a,:]),color=colours[r2dx[a],:])
    
plt.xticks(np.linspace(0,len(regchoice),len(regchoice)),labels=regchoice,rotation=45,ha='right')
plt.subplots_adjust(bottom=0.4)
plt.yscale('log')
plt.show()
plt.ylabel('p Value')
plt.title('FB5AB')
plt.savefig(os.path.join(savedir,'FB5AB_p_val.png'))
#%% Run regression model by all fb4M neurons
plt.close('all')
savedir_n = "Y:\Data\FCI\FCI_summaries\TangentialRegression\FB4M"
regchoice = ['odour onset', 'odour offset', 'in odour', 
                                'cos heading pos','cos heading neg', 'sin heading pos', 'sin heading neg',
                                'angular velocity pos','x pos','x neg','y pos', 'y neg','ramp down since exit','ramp to entry'] 

flies =['Y:\\Data\\FCI\\AndyData\\FB4M_jGCaMP7f\\20220510_FB4MjGCaMP7f_Fly1_001\\processed',
        'Y:\\Data\\FCI\\AndyData\\FB4M_jGCaMP7f\\20220510_FB4MjGCaMP7f_Fly2_002\\processed']
weights = np.zeros([len(flies),len(regchoice)],dtype='float')
weights_c = np.zeros([len(flies),len(regchoice)+1],dtype='float')
r2 = np.zeros(len(flies),dtype='float')
p = np.zeros([len(flies),len(regchoice)],dtype='float')
for i,f in enumerate(flies):
    print('Fly',i)
    post_processing_file = os.path.join(f,'postprocessing.h5')
    pv2 = pd.read_hdf(post_processing_file, 'pv2')
    ft2 = pd.read_hdf(post_processing_file, 'ft2')
    fc = fci_regmodel(pv2['fb4m_dff'],ft2,pv2)
    fc.run(regchoice)
    fc.run_dR2(20,fc.xft)
    weights_c[i,:] = fc.coeff_cv
    weights[i,:] = fc.dR2_mean
    r2[i] = fc.r2
    p[i,:] = fc.dR2_ttest
    
    # Plot example fluorescence and fit
    fc.plot_example_flur()
    plt.savefig(os.path.join(savedir_n,'Example_Fit_F' + str(i) + '.png'))
    plt.rcParams['pdf.fonttype'] = 42 
    plt.savefig(os.path.join(savedir_n,'Example_Fit_F' + str(i) + '.pdf'))
    plt.xlim([0,200])
    plt.savefig(os.path.join(savedir_n,'Example_Fit_F' + str(i) + '_200secs.png'))
    # Plot in and outside odour
    fc.plot_flur_w_regressors(['in odour','ramp down since exit'])
    plt.savefig(os.path.join(savedir_n,'Example_F' + str(i) + '.png'))
    
    plt.xlim([0,200])
    plt.savefig(os.path.join(savedir_n,'Example_F' + str(i) + '_200secs.png'))
    
    plt.close('all')
    fc.example_trajectory()
    plt.savefig(os.path.join(savedir_n,'Trajectory_F' + str(i) + '_.png'))
    plt.rcParams['pdf.fonttype'] = 42 
    plt.rcParams['ps.fonttype'] = 42 
    plt.savefig(os.path.join(savedir_n,'Trajectory_F' + str(i) + '_.eps'), format='eps')
    plt.rcParams['pdf.fonttype'] = 42 
    plt.savefig(os.path.join(savedir_n,'Trajectory_F' + str(i) + '_.pdf'))
    plt.savefig(os.path.join(savedir_n,'Trajectory_F' + str(i) + '_.svg'))
    fc.plot_all_regressors(regchoice)
    plt.savefig(os.path.join(savedir_n,'All_regressors' + str(i) + '_.png'))
    plt.rcParams['pdf.fonttype'] = 42
    plt.savefig(os.path.join(savedir_n,'All_regressors' + str(i) + '_.pdf'))
    plt.close('all')
    
    
colours = np.array([[247,251,255],
[222,235,247],
[198,219,239],
[158,202,225],
[107,174,214],
[66,146,198],
[33,113,181],
[8,81,156],
[8,48,107]],dtype='float')/255
r2dx = np.round(r2*10).astype(int)

plt.figure()
plt.plot([0, 14],[0, 0],color='k',linestyle='--')   
for a in range(len(r2)):
    plt.plot(np.linspace(0,len(regchoice),len(regchoice)),np.transpose(weights_c[a,:-1]),color=colours[r2dx[a],:])
wm = np.mean(weights_c,axis=0)

savedir = "Y:\Data\FCI\FCI_summaries\TangentialRegression"
plt.plot(np.linspace(0,len(regchoice),len(regchoice)),wm[:-1],color='k',linewidth=2)
plt.xticks(np.linspace(0,len(regchoice),len(regchoice)),labels=regchoice,rotation=45,ha='right')
plt.subplots_adjust(bottom=0.4)
plt.ylabel('beta weights')
plt.title('FB4M')
plt.show()
plt.savefig(os.path.join(savedir,'FB4M_dR2_weights.png'))

plt.figure()
plt.plot([0, 14],[0, 0],color='k',linestyle='--')   
for a in range(len(r2)):
    plt.plot(np.linspace(0,len(regchoice),len(regchoice)),np.transpose(weights[a,:]),color=colours[r2dx[a],:])
wm = np.mean(weights,axis=0)
plt.plot(np.linspace(0,len(regchoice),len(regchoice)),wm[:],color='k',linewidth=2)
plt.xticks(np.linspace(0,len(regchoice),len(regchoice)),labels=regchoice,rotation=45,ha='right')
plt.subplots_adjust(bottom=0.4)
plt.ylabel('delta R2')
plt.title('FB4M')
plt.show()

plt.savefig(os.path.join(savedir,'FB4M_dR2.png'))


plt.figure()
plt.plot([0, 14],[0.05, 0.05],color='k',linestyle='--')   
for a in range(len(r2)):
    plt.plot(np.linspace(0,len(regchoice),len(regchoice)),np.transpose(p[a,:]),color=colours[r2dx[a],:])
    
plt.xticks(np.linspace(0,len(regchoice),len(regchoice)),labels=regchoice,rotation=45,ha='right')
plt.subplots_adjust(bottom=0.4)
plt.yscale('log')
plt.show()
plt.ylabel('p Value')
plt.title('FB4M')
plt.savefig(os.path.join(savedir,'FB4M_p_val.png'))

#%%

colours = np.array([[247,251,255],
[222,235,247],
[198,219,239],
[158,202,225],
[107,174,214],
[66,146,198],
[33,113,181],
[8,81,156],
[8,48,107]],dtype='float')/255

x = linspace(0,len(colours)-1,len(colours))
        
        
        
        
        