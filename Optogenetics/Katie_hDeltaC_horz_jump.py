# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 11:38:30 2026

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


#%% hDeltaC horizontal jumping plume - GtACR1; hDeltaC split
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
rootdir = r'\\hermes.rockefeller.edu\Katie\Odor\PulsePlumePairs\horizontal_plume_pairings\hDeltaC\horizontal_led_alternation_jump'

flies = [
r'260730\f1',
r'260730\f2',
r'260730\f3',
r'260730\f4',
r'260730\f5',
#r'260730\f6',
#r'260730\f7', # did not make it
r'260730\f8',

r'260731\f1',
r'260731\f2',
r'260731\f3',
#r'260731\f4', did not make it 
r'260731\f5',
#r'260731\f6',
r'260731\f7',

r'260804\f1',
r'260804\f2',
#r'260804\f3', did not make it

r'260805\f1',
r'260805\f2',
r'260805\f4',
r'260805\f6',
        ]
for i in range(len(flies)):
    sdir = os.path.join(rootdir, flies[i],'trial1')
    lname = np.array( os.listdir(sdir))
    
    lname = ug.get_folder(sdir)   
    print(lname[-1])
    datadir = os.path.join(sdir,lname[-1])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    plt.figure()
    op.plot_plume_simple(meta_data,df)
    plt.title(flies[i])



#%% hDeltaC horizontal jumping plume - GtACR1; hDeltaC split - SY food controls

rootdir = r'\\hermes.rockefeller.edu\Katie\Odor\PulsePlumePairs\horizontal_plume_pairings\SYhorizontal_control\horizontal_led_alternation_jump'


flies = [
#r'260729\f4', # bug plume always on
r'260729\f5',

r'260730\f1',
r'260730\f3',
r'260730\f5',
#r'260730\f7',# did not make it

r'260731\f1',
r'260731\f3',
r'260731\f5',
r'260731\f7',

r'260811\f2',
r'260811\f3',
r'260811\f4',

r'260812\f1',
r'260812\f2',
r'260812\f3',
r'260812\f4',
r'260812\f5',
r'260812\f6',

r'260826\f1',
r'260826\f3',
r'260826\f4',
        ]
for i in range(len(flies)):
    sdir = os.path.join(rootdir, flies[i],'trial1')
    lname = np.array( os.listdir(sdir))
    
    lname = ug.get_folder(sdir)   
    print(lname[-1])
    datadir = os.path.join(sdir,lname[-1])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    df = fc.read_log(savepath)
    op = opto()
    plt.figure()
    op.plot_plume_simple(meta_data,df)
    plt.title(flies[i])



#%% Plots
# Load and clean data
cutoffs = {'max_y': 400,
           'thru_plume': 0,
           'max_time': 15*60}

groups = {'test':
          {
              'flies': [r'260730\f1',
          r'260730\f2',
          r'260730\f3',
          r'260730\f4',
          r'260730\f5',
          #r'260730\f6',
          #r'260730\f7', # did not make it
          r'260730\f8',

          r'260731\f1',
          r'260731\f2',
          r'260731\f3',
          #r'260731\f4', did not make it 
          r'260731\f5',
          #r'260731\f6',
          r'260731\f7',

          r'260804\f1',
          r'260804\f2',
          #r'260804\f3', did not make it

          r'260805\f1',
          r'260805\f2',
          r'260805\f4',
          r'260805\f6'],
              
              'rootdir':
                  r'\\hermes.rockefeller.edu\Katie\Odor\PulsePlumePairs\horizontal_plume_pairings\hDeltaC\horizontal_led_alternation_jump'        
              
           }
          
          ,
          'SY':
              {
                  'flies': [
                      #r'260729\f4', # bug plume always on
                  r'260729\f5',

                  r'260730\f1',
                  r'260730\f3',
                  r'260730\f5',
                  #r'260730\f7',# did not make it

                  r'260731\f1',
                  r'260731\f3',
                  r'260731\f5',
                  r'260731\f7',

                  r'260811\f2',
                  r'260811\f3',
                  r'260811\f4',

                  r'260812\f1',
                  r'260812\f2',
                  r'260812\f3',
                  r'260812\f4',
                  r'260812\f5',
                  r'260812\f6',

                  r'260826\f1',
                  r'260826\f3',
                  r'260826\f4',
                          ],
                  'rootdir':
                      r'\\hermes.rockefeller.edu\Katie\Odor\PulsePlumePairs\horizontal_plume_pairings\SYhorizontal_control\horizontal_led_alternation_jump'
                  }
              }

all_flies = {}
for g in groups:
    flies = groups[g]['flies']
    rootdir = groups[g]['rootdir']
    all_flies[g] = {}
    goodflies = np.array([],dtype='int')
    for i,f in enumerate(flies):
        all_flies[g][i] ={}
        sdir = os.path.join(rootdir, f,'trial1')
        lname = np.array( os.listdir(sdir))
        
        lname = ug.get_folder(sdir)   
        print(lname[-1])
        datadir = os.path.join(sdir,lname[-1])
        files = os.listdir(datadir)
        savepath = os.path.join(datadir,files[0])
        df = fc.read_log(savepath)
        all_flies[g][i]['df'] = df
        led = df['led1_stpt'].to_numpy()  
        # Apply cutoffs
        t = ug.get_ft_time(df)
        ins = df.instrip.to_numpy()
        st = np.where(ins)[0][0]
        texp = t-t[st]
        x = df.ft_posx.to_numpy();y = df.ft_posy.to_numpy()
        x,y = ug.fictrac_repair(x, y)
        jump = df.jump.to_numpy()
        x =x-x[st];y = y-y[st]
        yjump_rel = y-jump
        tcut = texp<cutoffs['max_time']
        ycut = yjump_rel<cutoffs['max_y']
        
        ycut[:st] = True
        ynotthru = yjump_rel>(-25-cutoffs['thru_plume'])
        ynotthru[:st] = True
        
        plt.plot(t,tcut,color='k')
        plt.plot(t,ycut,color='r')
        plt.plot(t,ynotthru,color='b')
        
        expindex = ((tcut& ycut)&ynotthru)
        
        all_flies[g][i]['expindex'] = expindex
        all_flies[g][i]['x'] = x
        all_flies[g][i]['y'] = y
        all_flies[g][i]['t'] = t
        all_flies[g][i]['ins'] = ins
        all_flies[g][i]['start'] = st
        
        date = int(f.split('\\')[0])
        all_flies[g][i]['date'] = date
        
        lu = np.unique(led[expindex])
        print(lu)
        if len(lu)>1:
            goodflies =np.append(goodflies,i)
            
    all_flies[g]['goodflies'] = goodflies
        
#%% Plot data   
# 1. trajectories side by side
plt.close('all')
offset = 0

groupnames = ['test','SY']
colours = ['r','k']
cnt = 0
for ig,g in enumerate(groupnames):
    data = all_flies[g]
    goodflies = all_flies[g]['goodflies']
    for d in goodflies:
        dx = data[d]['expindex']
        st = data[d]['start']
        df = data[d]['df']
        x = data[d]['x'][dx]
        y = data[d]['y'][dx]
        x = x[st:];y = y[st:]
        plt.figure(1)
        plt.plot(x+offset,y,color=colours[ig])
        xrange = max(x)-min(x)
        offset = offset+xrange+50
        jumps = df['jump'].to_numpy()[dx]
        jnum = np.max(-jumps)/20
        plt.figure(2)
        plt.scatter(data[d]['date'],jnum,color=colours[ig])
        cnt +=1
    
    
    
    offset = offset+500
plt.figure(1)
g = plt.gca().set_aspect('equal')



#%%
# 2. Stim versus no stim
for ig,g in enumerate(groupnames):
    data = all_flies[g]
    goodflies = all_flies[g]['goodflies']
    plt.figure()
    for d in goodflies:
        dx = data[d]['expindex']
        st = data[d]['start']
        df = data[d]['df']
        x = data[d]['x'][dx]
        y = data[d]['y'][dx]
        x = x[st:];y = y[st:]
        ins = data[d]['ins'][dx]
        ins = ins[st:]
        jump = df['jump'].to_numpy()[dx]
        jump = jump[st:]
        yjmp = y-jump
        
        blk,blks = ug.find_blocks(ins==0,mergeblocks=False,merg_threshold=5)
        led  = df['led1_stpt'].to_numpy()[dx]
        led = led[st:]
        ledon = led<1
        for ib,b in enumerate(blk):
            dx = np.arange(b,blks[ib]+b)
            ymx = np.max(yjmp[dx])
            tled = ledon[b+1]
            if tled==0:
                plt.scatter(ib,ymx,color='k')
            else:
                plt.scatter(ib,ymx,color='g')
        
        
    plt.xlim([-1,15])
    
            


# 3. 




















































