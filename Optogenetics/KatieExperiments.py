# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:24:56 2026

@author: dowel
"""
import numpy as np
import pandas as pd
import src.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42 
from pathlib import Path
from Utilities.utils_general import utils_general as ug
from scipy import stats
#%% hDeltaC horizontal plume
plt.close('all')
meta_data = {'stim_type':  'alternation',
              'act_inhib':'inhib',            
              'ledOnx':0,
              'ledOffx':0,
              'ledOny':0,
              'ledOffy':0,
              'PlumeWidth': float(50),
              'PlumeAngle': 90,
              
              }

rootdir = r'\\hermes.rockefeller.edu\Katie\Odor\horizontal_plume\HorizPlumeLedAlt\hDeltaC'

flies = [
    # r'260602\f1',
    #      r'260602\f2', # Not horizontal plume
    #     r'260603\f1',
    #     r'260603\f2',
    #     r'260603\f3',
    #      r'260604\f1',
    #      r'260604\f2',
    #      r'260604\f3',
    #      r'260611\f1',
    #      r'260611\f2',
    #      r'260611\f3',
         r'260624\f1',
         r'260624\f2',
         r'260624\f3',
         r'260701\f1',
         r'260701\f2',
         r'260701\f3',
        ]
for i in range(len(flies)):
    sdir = os.path.join(rootdir, flies[i],'trial1')
    lname = np.array( os.listdir(sdir))
    
    lname = ug.get_folder(sdir)   
    print(len(lname))
    datadir = os.path.join(sdir,lname[-1])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    plt.figure()
    op.plot_plume_simple(meta_data,df)
    plt.title(flies[i])
#%% Horizontal plume pairings
plt.close('all')
meta_data = {'stim_type':  'alternation',
              'act_inhib':'inhib',            
              'ledOnx':0,
              'ledOffx':0,
              'ledOny':0,
              'ledOffy':0,
              'PlumeWidth': float(50),
              'PlumeAngle': 90,
              
              }

rootdir = r'\\hermes.rockefeller.edu\Katie\\Odor\Inhibition\PulsePlumePairs\horizontal_plume_pairings\hDeltaC\HorizontalPlumeLEDAlt'

# flies = [
    
#          r'260624\f1', # Decent tracker with large excursion
#          r'260624\f2', # decent tracker with  large excursion
#          r'260624\f3', # Strange tracker close to plume
#          r'260624\f4', # Lovely tracker. Some large excursions
#          #r'260624\f5', # No LEDs
#          r'260624\f6', # Nice tracker, large excursions
         
#          r'260701\f1', # Close to plume not much trac
#          #r'260701\f2', # No LEDs
#          r'260701\f3', # Strange tracker close to plume edge. Large excursion with LED on
#         # r'260701\f4', # No LEDs
#          r'260701\f5', # Large excursion
         
#          r'260714\f1',# Ok tracker
#          r'260714\f2', # Circling buth otherwise a good tracker
#          #r'260714\f3', # No LEDs
#          r'260714\f4',# Lovely tracker
#          r'260714\f5', # Huge excursion from plume when LED comes on
#          r'260714\f6', # Plume cross overs with LED ON and off
         
#          r'260715\f1', # Lovely tracker
#          r'260715\f2', # Through plume then lost at end
#          #r'260715\f3', # No LEds
#          r'260715\f4', # Through plume with LED off - fine
#          r'260715\f5', # Through plume with LED then made it back upwind - assess
#          r'260715\f6' # lovely tracker - fine
#         ]


flies  = [r'260624\f1',
    r'260624\f2',
    r'260624\f3',
    r'260701\f1',
    r'260701\f3',
   # r'260701\f4', No LEDs
    r'260701\f5',
    r'260714\f1', #good
    r'260714\f2',
   # r'260714\f3', No LEDs
    r'260714\f4',
    r'260714\f5',
    r'260715\f1', # not sure if this has the correct pulse number
    r'260715\f2',
    #r'260715\f3',No LEDs
    r'260715\f5',
    r'260715\f6']


def get_led_blocks(led,ins):
    ledson,ledsize = ug.find_blocks(led)
    ledsoff,ledoffsize = ug.find_blocks(~led)
    ledsoffoff = ledsoff+ledoffsize+1
    plumeon,plumesize = ug.find_blocks(ins,mergeblocks=True,merg_threshold=5)
    plumeoff = plumeon+plumesize+1
    output = np.zeros((len(ledson),4),dtype='int')
    for i,le in enumerate(ledson):
        # Get leds off index 
        output[i,0] = max(plumeoff[plumeoff<ledsoffoff[i]])
        output[i,1] = le
        output[i,2] = max(plumeoff[plumeoff<le+ledsize[i]])
        output[i,3] = le+ledsize[i]+1
    return output

plotmat = np.zeros((len(flies),2))
stripwidth = 25
maxy = 400+stripwidth
miny = -200-stripwidth
maxt = 15*60
savedir = r'Y:\Data\Optogenetics\Summaries\hDeltaC'
club400 = []
for i in range(len(flies)):
    sdir = os.path.join(rootdir, flies[i],'trial1')
    lname = np.array( os.listdir(sdir))
    
    lname = ug.get_folder(sdir)   
    print(len(lname))
    datadir = os.path.join(sdir,lname[-1])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    plt.figure(i+1)
    #plt.subplot(1,2,1)
    op.plot_plume_simple(meta_data,df)
    plt.title(flies[i])
    
    # Get metrics
    x = df['ft_posx'].to_numpy()
    y = df['ft_posy'].to_numpy()
    led = df['led1_stpt'].to_numpy()<1
    ins = df['instrip'].to_numpy()
    tt = ug.get_ft_time(df)
    ins1 = np.where(ins>0)[0][0]
    tt = tt-tt[ins1]
    y = y-y[ins1]
    x = x-x[ins1]
    tdx = (tt>0)& (tt<maxt)
    x = x[tdx]; y = y[tdx];ins = ins[tdx];led = led[tdx]
    
    ydx = np.where(y>maxy)[0]
    len(ydx)
    if len(ydx)>0:
        x = x[:ydx[0]];y = y[:ydx[0]];led = led[:ydx[0]]; ins = ins[:ydx[0]]
    
    ydx = np.where(y<miny)[0]
    len(ydx)
    if len(ydx)>0:
        x = x[:ydx[0]];y = y[:ydx[0]];led = led[:ydx[0]]; ins = ins[:ydx[0]]
    
    
    # clip data
    
    
    
    led_index = get_led_blocks(led,ins)
    #plt.subplot(1,2,2)
    
    
    ydata  =np.zeros((len(led_index),2))
    for di in range(len(ydata)):
        dx1 = np.arange(led_index[di,0],led_index[di,1])
        dx2 = np.arange(led_index[di,2],led_index[di,3]-1)
        ty = y[dx1]-y[dx1[0]]
        tyi = np.argmax(np.abs(ty))
        ydata[di,0] = np.max(ty[tyi]);
        
        ty = y[dx2]-y[dx2[0]]
        tyi = np.argmax(np.abs(ty))
        ydata[di,1] = np.max(ty[tyi]);
        
        #ydata[di,1] = np.max(y[dx2]-y[dx2[0]])
    #plt.scatter(ydata[:,0],ydata[:,1])   
     
    if ug.round_to_sig_figs(ydata[-1,-1],2)==400:
        club400.append(flies[i])
    
        
    
    #plt.plot([0,500],[0,500],color='k')
    #plt.gca().set_aspect('equal')
    fname = str.split(flies[i],'\\')
    plt.savefig(os.path.join(savedir,'HorizontalTraj' + fname[0]+fname[1]+'.pdf'))
    plt.figure(101)
    if led_index[-1,-1]>=len(ins):
        
        plt.scatter(ydata[:-1,0],ydata[:-1,1],color='k',s=20,alpha=.25,linewidth=0)
        plt.scatter(ydata[-1,0],ydata[-1,1],color='g',s=20,linewidth=0)
        
        plotmat[i,:] = np.median(ydata[:-1,:],axis=0)
    else :
        plt.scatter(ydata[:,0],ydata[:,1],color='k',s=20,alpha=.25,linewidth=0)
        plt.scatter(y[-1],ydata[-1,1],color='r',s=20,linewidth=0)
        plotmat[i,:] = np.median(ydata,axis=0)
        
    
    
    
    
    
    
    
# plt.plot([0,700],[0,700])
# plt.xlim([-100,1000])
# plt.ylim([-100,1000])
#plt.xlim([-10,410])
#plt.ylim([-10,410])
plt.plot([0,400],[0,400],color='r')
plt.scatter(plotmat[:,0],plotmat[:,1],color='b',linewidth=0)
plt.xlabel('max y pre led')
plt.ylabel('max y led')
#plt.yscale('log')
#plt.xscale('log') 
plt.figure()
x = np.arange(0,2)
xt  =np.tile(x[:,np.newaxis],len(plotmat))
plt.plot(xt,plotmat.T,color='k',alpha=.25)
sat,pval = stats.wilcoxon(plotmat[:,0],plotmat[:,1],nan_policy='omit')
plt.xticks([0,1],labels=['led off','led on'])
plt.plot(x,np.nanmedian(plotmat,axis=0),linewidth=3,color='k')
plt.ylabel('median max y displacement (mm)')
plt.text(.5,110,str(ug.round_to_sig_figs(pval, 2)),horizontalalignment='center')
plt.ylim([-10,120])

#%% plot plume and pulse data for 400 club flies

pulseroot = r'\\hermes.rockefeller.edu\Katie\Odor\PulsePlumePairs\horizontal_plume_pairings\hDeltaC\led_pulses'
plumeroot =  r'\\hermes.rockefeller.edu\Katie\Odor\PulsePlumePairs\horizontal_plume_pairings\hDeltaC\HorizontalPlumeLEDAlt'
meta_data_plume = {'stim_type':  'alternation',
              'act_inhib':'inhib',            
              'ledOnx':0,
              'ledOffx':0,
              'ledOny':0,
              'ledOffy':0,
              'PlumeWidth': float(50),
              'PlumeAngle': 90,
              
              }
meta_data_pulse = {'stim_type':  'pulse',
              'act_inhib':'inhib',            
              'ledOnx':0,
              'ledOffx':0,
              'ledOny':0,
              'ledOffy':0,
              'PlumeWidth': float(50),
              'PlumeAngle': 90,
              
              }
plt.close('all')
data = np.zeros((len(club400),3))
savedir = r'Y:\Data\Optogenetics\Summaries\hDeltaC'

for i in range(len(club400)):
    sdir = os.path.join(plumeroot, club400[i],'trial1')
    lname = np.array( os.listdir(sdir))
    
    lname = ug.get_folder(sdir)   
    print(len(lname))
    datadir = os.path.join(sdir,lname[-1])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    plt.figure(i+1)
    plt.subplot(1,2,1)
    op.plot_plume_simple(meta_data_plume,df)
    plt.title(flies[i])
    
    
    sdir = os.path.join(pulseroot, club400[i],'trial1')
    lname = np.array( os.listdir(sdir))
    
    lname = ug.get_folder(sdir)   
    print(len(lname))
    datadir = os.path.join(sdir,lname[-1])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    
    plt.subplot(1,2,2)
    op.plot_plume_simple(meta_data_pulse,df)
    plt.title(flies[i])
    tt = ug.get_ft_time(df)
    dt = np.mean(np.diff(tt))
    t_int = int(10/dt)
    x = df['ft_posx'].to_numpy()
    y = df['ft_posy'].to_numpy()
    led = df['led1_stpt'].to_numpy()<1
    
    ledblk,ledsize = ug.find_blocks(led)
    ledblk = ledblk[:10];ledsize = ledsize[:10]
    
    blkoff = ledblk+ledsize+1
    
    preon = ledblk-t_int-1
    preoff = ledblk-1
    
    poston = blkoff+1
    postoff = poston+t_int
    
    prey = y[preoff]-y[preon]
    ledy = y[blkoff] - y[ledblk]
    posty = y[postoff] - y[poston]
    
    preym = np.mean(prey)
    ledym = np.mean(ledy)
    postym = np.mean(posty)
    data[i,0] = preym
    data[i,1] = ledym
    data[i,2] = postym

plt.figure()
x = np.arange(0,3)
xgrid = np.tile(x[np.newaxis,:],(len(club400),1))
# plt.figure()
# plt.subplot(1,2,1)
plt.plot(x,data.T,color='k',alpha=.25)
plt.scatter(xgrid[:,0],data[:,0],color=[.7,.7,.7],zorder=2,alpha=1)
plt.scatter(xgrid[:,1],data[:,1],color=[.7,1,.7],zorder=2,alpha=1)
plt.scatter(xgrid[:,2],data[:,2],color=[.7,.7,.7],zorder=2,alpha=1)
plt.xticks(x,labels=['10 s pre','10 s led','10 s post'])
plt.ylabel('mean y displacement (mm)')
yl = plt.gca().get_ylim()
plt.ylim([yl[0],105])
plt.savefig(os.path.join(savedir,'400_club_pulse_data.png'))

#%% pulses
    
plt.close('all')
meta_data = {'stim_type':  'pulse',
              'act_inhib':'inhib',            
              'ledOnx':0,
              'ledOffx':0,
              'ledOny':0,
              'ledOffy':0,
              'PlumeWidth': float(50),
              'PlumeAngle': 90,
              
              }

rootdir = r'\\hermes.rockefeller.edu\Katie\Odor\PulsePlumePairs\horizontal_plume_pairings\hDeltaC\led_pulses'

flies = [
    
         r'260624\f1', # Decent tracker with large excursion
         r'260624\f2', # decent tracker with  large excursion
         r'260624\f3', # Strange tracker close to plume
        
         #r'260624\f4', # Lovely tracker. Some large excursions no pulses
         #r'260624\f5', # No LEDs. No pulses
        # r'260624\f6', # Nice tracker, large excursions. No pulses
         
         r'260701\f1', # Close to plume not much trac
        # r'260701\f2', # No LEDs # No pulses
         r'260701\f3', # Strange tracker close to plume edge. Large excursion with LED on
         r'260701\f4', # No LEDs
         r'260701\f5', # Large excursion
         
         r'260714\f1',# Ok tracker
         r'260714\f2', # Circling buth otherwise a good tracker 
         r'260714\f3', # No LEDs
         r'260714\f4',# Lovely tracker
         r'260714\f5', # Huge excursion from plume when LED comes on
         r'260714\f6', # Plume cross overs with LED ON and off
         
         r'260715\f1', # Lovely tracker
         r'260715\f2', # Through plume then lost at end
         r'260715\f3', # No LEds
         r'260715\f4', # Through plume with LED off - fine
         r'260715\f5', # Through plume with LED then made it back upwind - assess
         r'260715\f6' # lovely tracker - fine
        ]





data = np.zeros((len(flies),3))
for i in range(len(flies)):
    sdir = os.path.join(rootdir, flies[i],'trial1')
    lname = np.array( os.listdir(sdir))
    
    lname = ug.get_folder(sdir)   
    print(len(lname))
    datadir = os.path.join(sdir,lname[-1])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    op.plot_plume_simple(meta_data,df)
    plt.figure()
    plt.title(flies[i])

    tt = ug.get_ft_time(df)
    dt = np.mean(np.diff(tt))
    t_int = int(10/dt)
    x = df['ft_posx'].to_numpy()
    y = df['ft_posy'].to_numpy()
    led = df['led1_stpt'].to_numpy()<1
    
    ledblk,ledsize = ug.find_blocks(led)
    ledblk = ledblk[:10];ledsize = ledsize[:10]
    
    blkoff = ledblk+ledsize+1
    
    preon = ledblk-t_int-1
    preoff = ledblk-1
    
    poston = blkoff+1
    postoff = poston+t_int
    
    prey = y[preoff]-y[preon]
    ledy = y[blkoff] - y[ledblk]
    posty = y[postoff] - y[poston]
    
    preym = np.mean(prey)
    ledym = np.mean(ledy)
    postym = np.mean(posty)
    data[i,0] = preym
    data[i,1] = ledym
    data[i,2] = postym
    
#%% Flies Katie plotted
plt.close('all')
flies = [r'260624\f1\trial1',
    r'260624\f2\trial1',
    r'260624\f3\trial1',
    r'260701\f1\trial1',
    r'260701\f3\trial1',
    r'260701\f4\trial1',
    r'260701\f5\trial1',
    r'260714\f1\trial1', #good
    r'260714\f2\trial1',
    r'260714\f3\trial1',
    r'260714\f4\trial1',
    r'260714\f5\trial1',
    r'260715\f1\trial1', # not sure if this has the correct pulse number
    r'260715\f2\trial1',
    r'260715\f3\trial1',
    r'260715\f5\trial1',
    r'260715\f6\trial1']

# flies = [ '260624/f1/trial1',
#     '260624/f2/trial1',
#     '260624/f3/trial1',
#     '260701/f1/trial1',
#     '260701/f3/trial1',
#     '260701/f4/trial1',
#     '260701/f5/trial1',
#     '260714/f1/trial1', #good
#     '260714/f2/trial1',
#     '260714/f3/trial1',
#     '260714/f4/trial1',
#     '260714/f5/trial1',
#     '260715/f1/trial1', # not sure if this has the correct pulse number
#     '260715/f2/trial1',
#     '260715/f3/trial1',
#     '260715/f5/trial1',
#     '260715/f6/trial1',
# ]
rootdir_pulse = '\\hermes.rockefeller.edu\Katie\Odor\PulsePlumePairs\horizontal_plume_pairings\hDeltaC\led_pulses'
#rootdir = r'\\hermes.rockefeller.edu\Katie\Odor\PulsePlumePairs\horizontal_plume_pairings\hDeltaC\HorizontalPlumeLEDAlt'
data = np.zeros((len(flies),3))
savedir = r'Y:\Data\Optogenetics\Summaries\hDeltaC'
for i in range(len(flies)):
    sdir = os.path.join(rootdir,flies[i])
    
    lname = ug.get_folder(sdir)   
    print(len(lname))
    datadir = os.path.join(sdir,lname[-1])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    
    
    df = fc.read_log(savepath)
    
    op = opto()
    plt.figure()
    op.plot_plume_simple(meta_data,df)
    
    plt.title(flies[i])
    fname = str.split(flies[i],'\\')
    figsavedir = os.path.join(savedir,fname[0]+fname[1]+fname[2]+'.pdf')
    plt.savefig(figsavedir)
    tt = ug.get_ft_time(df)
    dt = np.mean(np.diff(tt))
    t_int = int(10/dt)
    x = df['ft_posx'].to_numpy()
    y = df['ft_posy'].to_numpy()
    led = df['led1_stpt'].to_numpy()<1
    
    ledblk,ledsize = ug.find_blocks(led)
    ledblk = ledblk[:10];ledsize = ledsize[:10]
    
    blkoff = ledblk+ledsize+1
    
    preon = ledblk-t_int-1
    preoff = ledblk-1
    
    poston = blkoff+1
    postoff = poston+t_int
    
    prey = y[preoff]-y[preon]
    ledy = y[blkoff] - y[ledblk]
    posty = y[postoff] - y[poston]
    
    preym = np.mean(prey)
    ledym = np.mean(ledy)
    postym = np.mean(posty)
    data[i,0] = preym
    data[i,1] = ledym
    data[i,2] = postym
    
    
#%% Plot aggregated stats
plt.close('all')
savedir = r'Y:\Data\Optogenetics\Summaries\hDeltaC'
import scikit_posthocs as sp
x = np.arange(0,3)
xgrid = np.tile(x[np.newaxis,:],(len(flies),1))
# plt.figure()
# plt.subplot(1,2,1)
plt.plot(x,data.T,color='k',alpha=.25)
plt.scatter(xgrid[:,0],data[:,0],color=[.7,.7,.7],zorder=2,alpha=1)
plt.scatter(xgrid[:,1],data[:,1],color=[.7,1,.7],zorder=2,alpha=1)
plt.scatter(xgrid[:,2],data[:,2],color=[.7,.7,.7],zorder=2,alpha=1)
dmean = np.mean(data,axis=0)
plt.plot(x,dmean,color='k',linewidth=3)
plt.scatter(x[[0,2]],dmean[[0,2]],color='k',s=100,zorder=10)
plt.scatter(x[1],dmean[1],color='g',zorder=10,s=100)
_,p1 = stats.wilcoxon(data[:,0],data[:,1])
_,p2 = stats.wilcoxon(data[:,1],data[:,2])
_,p3 = stats.wilcoxon(data[:,0],data[:,2])
p1 = p1*3;p2=p2*3;p3=p3*3

#h_stat, p_val = stats.kruskal(data[:,0],data[:,1])


plt.plot([0,.9],[90,90],color='k')
plt.text(.5,92,str(ug.round_to_sig_figs(p1, 2)),horizontalalignment='center')
plt.plot([1.1,2],[90,90],color='k')
plt.text(1.5,92,str(ug.round_to_sig_figs(p2, 2)),horizontalalignment='center')
plt.plot([0,2],[100,100],color='k')
plt.text(1,102,str(ug.round_to_sig_figs(p3, 2)),horizontalalignment='center')
plt.xticks(x,labels=['10 s pre','10 s led','10 s post'])
plt.ylabel('mean y displacement (mm)')
plt.savefig(os.path.join(savedir,'hDeltaC_GtACR1_pulse_inhib_aggregate.png'))
plt.savefig(os.path.join(savedir,'hDeltaC_GtACR1_pulse_inhib_aggregate.pdf'))

plt.figure()
pdata = data[:,1:]-data[:,0][:,np.newaxis]
plt.plot(xgrid[:,:2].T,pdata.T,color='k',zorder=-1,alpha=.25)
plt.scatter(xgrid[:,0],pdata[:,0],color='g')
plt.scatter(xgrid[:,1],pdata[:,1],color='k')
plt.plot(x[:2],np.median(pdata,axis=0),color='k',linewidth=3)    



#%% hDeltaC flies - horizontal jumping plume
rootdir = r'\\hermes.rockefeller.edu\Katie\Odor\Inhibition\PulsePlumePairs\horizontal_plume_pairings\hDeltaC\horizontal_led_alternation_jump'

flies = [
    r'260730\f1\trial1', # little discernable effect
    #r'260730\f2\trial1', pre odour downwind bias exclude
    r'260730\f3\trial1', # hard to parse
    r'260730\f4\trial1', # strong upwind pre air, expected effect
    r'260730\f5\trial1',# down and up pre air, otherwise decent
    #r'260730\f7\trial1', No LED
    r'260730\f8\trial1',# poor pre air, slightly downwind, exlude maybe
    
    r'260731\f1\trial1', # downwind pre air, exclude maybe
    r'260731\f2\trial1',# downwind pre air, exclude maybe
    r'260731\f3\trial1', # smooth walking, fly walks off but not during stim
    #r'260731\f4\trial1', No LED
    r'260731\f5\trial1', # does a decent job
    r'260731\f7\trial1', # hard to parse but food enough behaviour
   
   # r'260804\f1\trial1', # circling exclude
    r'260804\f2\trial1', # downwind pre air, otherwise Ok, maybe exclude
   # r'260804\f3\trial1', # No LEDs
    
    r'260805\f1\trial1', # downwind pre air okish behaviour, maybe exclude
    r'260805\f2\trial1', # downwind pre air, lots of entries, maybe exclude
    r'260805\f4\trial1', # downwind pre air exlude, otherwise okish
    #r'260805\f6\trial1', # went through plume
    
    #r'260910\f1\trial1', # No LEDs
   # r'260910\f2\trial1', # No LED
    
    #r'260910\f3\trial1', tracking errors
    #r'260910\f4\trial1', # No LED
    #r'260910\f5\trial1', # No LED
    #r'260910\f6\trial1', No LED
   # r'260910\f7\trial1',## return to this one not plotting
    
    
    r'260911\f1\trial1', #  decent data, left with LED
    #r'260911\f2\trial1', # No LED
    #r'260911\f4\trial1', # downwind pre air through plume
    #r'260911\f5\trial1', # No LED
    
    #r'260917\f1\trial1', # No LED
    r'260917\f2\trial1', # little to no phenotype
    r'260917\f3\trial1', # okish walker
    
    r'260918\f1\trial1', # downwind pre air, v good tracker, maybe exclude
    r'260918\f2\trial1', # downwind pre air exclude
   # r'260918\f3\trial1', # No LED
   # r'260918\f4\trial1', # No LED
   # r'260918\f5\trial1',# No LED
    
    ]

plt.close('all')
meta_data = {'stim_type':  'alternation_jump',
              'act_inhib':'inhib',            
              'ledOnx':0,
              'ledOffx':0,
              'ledOny':0,
              'ledOffy':0,
              'PlumeWidth': float(50),
              'PlumeAngle': 90,
              
              }



for i in range(len(flies)):
    sdir = os.path.join(rootdir, flies[i])
    lname = np.array( os.listdir(sdir))
    
    lname = ug.get_folder(sdir)   
    print(len(lname))
    datadir = os.path.join(sdir,lname[-1])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    plt.figure()
    op.plot_returns_in_sequence_horz(meta_data,df)
    #plt.subplot(1,2,1)
    #op.plot_plume_simple(meta_data,df)
   #plt.subplot(1,2,2)
    df = op.apply_cutoffs(df, meta_data,max_from_plume=400,max_downwind=np.inf,max_time=np.inf)
    plt.figure()
    op.plot_returns_in_sequence_horz(meta_data,df)
    plt.title(flies[i])
    plt.ylim([-100,400])
    
    # collect fourth and fifth entry
    
    data = op.get_return_metrics(meta_data,df,ent_duration=.1)
    first_led = np.where(data['led']==1)[0][0]
    my1 = data['max_y'][first_led]
    my2 = data['max_y'][first_led-1]
    
    plt.figure(101)
    plt.scatter(my2,my1,color='k')
plt.plot([0,400],[0,400],color='k',linestyle='--')
#%% hDeltaC control animals horizontal jumping plume
# I think these are w- X GtACR1 raised on SY food




#%% hDeltaJ flies - vertical jumping plume
rootdir = r'\\hermes.rockefeller.edu\Katie\Odor\Inhibition\PulsePlumePairs\vertical_plume_pairings\hDeltaJ\VerticalPlumeLEDAlt'
flies = [r'260625\f1\trial1',
         r'260625\f2\trial1', # LED on after cross over, struggles to return
         r'260625\f3\trial1', # one led stim
        # r'260625\f4\trial1', # downwind tracker exclude
         r'260625\f5\trial1', # v long returns after cross over
        # r'260625\f6\trial1', did not make jumps
         
         #r'260626\f1\trial1', circling
         
         r'260707\f1\trial1', # shorter returns with LED on
        # r'260707\f2\trial1', circling
         #r'260707\f3\trial1', circling
         r'260707\f4\trial1', # decent tracker, pre-odour bias maintained maybe
         
        # r'260709\f1\trial1', No led
        # r'260709\f2\trial1', # downwind tracker exclude
         r'260709\f3\trial1', # Walk off after final LED stim
         r'260709\f4\trial1', # walk off after final LED stim
        # r'260709\f5\trial1', no jumps
         r'260709\f6\trial1', # not great walking but made an LED stim
         
         r'260716\f1\trial1', # walk off with LED on
         #r'260716\f2\trial1', did not make LEDs
         #r'260716\f3\trial1', downwind tracking
         r'260716\f4\trial1', # decent tracking, walks off after plume cross over and stim on.
        # r'260716\f6\trial1', did not make LEDs
         
         r'260717\f1\trial1',
         #r'260717\f2\trial1', circling
         r'260717\f3\trial1', # exclude as cut off before tracking
         r'260717\f4\trial1', # Decent tracking, straigher walking with LED on 
         ]
    

plt.close('all')
meta_data = {'stim_type':  'alternation_jump',
              'act_inhib':'inhib',            
              'ledOnx':0,
              'ledOffx':0,
              'ledOny':0,
              'ledOffy':0,
              'PlumeWidth': float(50),
              'PlumeAngle': 0,
              
              }
for i in range(len(flies)):
    sdir = os.path.join(rootdir, flies[i])
    lname = np.array( os.listdir(sdir))
    
    lname = ug.get_folder(sdir)   
    print(len(lname))
    datadir = os.path.join(sdir,lname[-1])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    plt.figure()
   
    op.plot_plume_simple(meta_data,df)
    op.plot_returns_in_sequence_vert(meta_data,df)
    plt.title(flies[i])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    