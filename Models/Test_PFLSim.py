# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:20:59 2024

@author: dowel

Idea is to simulate PFL_3 activity


"""
#%%
import numpy as np
from analysis_funs.CX_analysis_col import CX_a
import matplotlib.pyplot as plt 
from analysis_funs.utilities import funcs as fn
from Utils.utils_general import utils_general as ug
import os
#%% 
datadirs = [
    "Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f1\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f2\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240502\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240514\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\241104\\f1\\Trial5",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\241106\\f1\\Trial2"]
datadir =datadirs[3]
d = datadir.split("\\")
name = d[-3] + '_' + d[-2] + '_' + d[-1]
cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)

top_dir = 'D:\\ConnectomeData\\FlywireWholeBrain\\'
PFL3_meta = ug.load_pick(os.path.join(top_dir,'PFL3_meta_data.pkl')) 
pb_angles = np.linspace(-np.pi,np.pi,16)
#%%

# edits: account for PFL3 neurons not mapping all bridge phases equally, can put in with
# real anatomy data

# Need to see how close turn data to PFL3 in LAL data actually looks.

# Think about whether the animal is gating FC2 or PFL3 output
def ELU(M):
    r_array = np.zeros(M.shape)
    r_array[M<=0] = np.exp(M[M<=0])-1
    return r_array
    
def PFL3_function(heading,goal,offset=0):
    x = np.linspace(-np.pi,np.pi,16)+offset
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
def PFL3_function_anat(PFL3_meta,heading,goal,gweights=[1,1,1,1,1,1],full_output=False,offset = 0):
    # Does the PFL3 comparison using anatomical data.
    
    #To do: check output versus idealised PFL3 neurons. It could be that the relative
    # FSB to PB tuning is a little off.
    
    pb_angles = np.linspace(0,2*np.pi,16)+offset
    pb_angles = np.append(pb_angles[np.arange(1,16,2)], pb_angles[np.arange(0,16,2)])
    fsb_angles = np.linspace(0,2*np.pi,12)+offset
    
    xblock = np.ones((len(heading),len(pb_angles)))
    xblock = xblock*pb_angles.T
    
    xblock_g = np.ones((len(heading),len(fsb_angles)))
    xblock_g = xblock_g*fsb_angles
    
    heading = np.tile(heading,[16,1]).T
    # Modified for summing different goals
    if len(goal.shape)>1:
        sd = goal.shape[1]
        for s in sd:
            tg = np.tile(goal[:,s],[12,1]).T
            if s==0:
                gsignal =np.cos(tg+xblock_g)*gweights[s]
            else:
                gsignal = np.cos(tg+xblock_g)*gweights[s]+gsignal
    else:
        goal = np.tile(goal,[12,1]).T
        gsignal = np.cos(goal+xblock_g)
    hsignal = np.cos(heading+xblock)
    
    
    PFL_L = PFL3_meta['LAL']<0 
    PFL_R = PFL3_meta['LAL']>0
    
    Lfsb = PFL3_meta['fsb_column'][PFL_L]
    Rfsb = PFL3_meta['fsb_column'][PFL_R]
    
    Lpb = PFL3_meta['pbglom'][PFL_L]
    Rpb = PFL3_meta['pbglom'][PFL_R]
    
    
    Lout = np.exp(gsignal[:,Lfsb]+hsignal[:,Lpb])
    Rout = np.exp(gsignal[:,Rfsb]+hsignal[:,Rpb])
    L_Lal = np.sum(Lout,axis=1)
    R_Lal = np.sum(Rout,axis=1)
    if full_output:
        return Lout,Rout
    return R_Lal,L_Lal
    
def PFL2_function(heading,goal):
    x = np.linspace(-np.pi,np.pi,16)
    xblock = np.ones((len(heading),len(x)))
    xblock = xblock*x.T
    goal = np.tile(goal,[16,1]).T
    heading = np.tile(heading,[16,1]).T
    
    gsignal = np.cos(goal+xblock)
    hsignal_L = np.cos(heading+xblock+np.pi)
    hsignal_R = np.cos(heading+xblock-np.pi)
    
    PFL3_L = np.exp(gsignal+hsignal_L)-1
    PFL3_R = np.exp(gsignal+hsignal_R)-1
    
    R_Lal = np.sum(PFL3_L,axis=1)/25.58 # janky and scales 1 c.a. 1
    L_Lal = np.sum(PFL3_R,axis= 1)/25.58
    return R_Lal,L_Lal

def PFL3_fun_anyl(heading, goal):
    hL = fn.wrap(heading-np.pi/4)
    hR = fn.wrap(heading+np.pi/4)
    gdiff_L = fn.wrap(hL-goal)
    gdiff_R = fn.wrap(hR-goal)
    L_Lal = np.cos(gdiff_L)
    R_Lal = np.cos(gdiff_R)
    return R_Lal,L_Lal

def PFL3_n_2(heading,goal,w=0.75):
    R1,L1 = PFL3_function(heading,goal)
    R2,L2 = PFL2_function(heading,goal)
    
    dna03L = np.exp(L1+L2)-1
    dna03R = np.exp(R1+R2)-1
    dna03L = dna03L/np.max(dna03L)
    dna03R = dna03R/np.max(dna03R)
    dna02L = np.exp(L1+dna03L*w)
    dna02R =np.exp( R1+dna03R*w)
    return dna02R,dna02L
#%% 
def output_trajectory(fvel,heading,goal,PFL3_meta,gain=0.1):
    
    #R,L = PFL3_function(heading,goal) #Simulated wedges - some turn signal for small heading values
   # R,L = PFL3_fun_anyl(heading,goal) #Analytical solution - projection onto offset axes
    #R,L = PFL3_n_2(heading,goal)
    R,L = PFL3_function_anat(PFL3_meta,heading,goal)
    angvel = gain*(R-L)
    infhead = np.cumsum(angvel)
    ang = np.arctan2(np.sin(infhead),np.cos(infhead))
    dx = fvel*np.sin(ang)
    dy = fvel*np.cos(ang)
    dx[0] =dx[0]+np.sin(heading[0])
    dy[0] = dy[0]+np.cos(heading[0])
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    
    return x,y
global PFL3_meta
def output_trajectory_fit(in_array,gain):
    fvel = in_array[0,:]
    heading = in_array[1,:]
    goal = in_array[2,:]
    R,L = PFL3_n_2(heading,goal)
    #R,L = PFL3_function_anat(PFL3_meta,heading,goal)
    angvel = gain*(R-L)
    infhead = np.cumsum(angvel)
    ang = np.arctan2(np.sin(infhead),np.cos(infhead))
    dx = fvel*np.sin(ang)
    dy = fvel*np.cos(ang)
    dx[0] =dx[0]+np.sin(heading[0])
    dy[0] = dy[0]+np.cos(heading[0])
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    d = x**2+y**2
    out_array = np.append(x,y)
    
    return out_array
xs = np.where(cxa.ft2['instrip']==1)[0][0]+10
xo = cxa.ft2['ft_posx'].to_numpy()
yo = cxa.ft2['ft_posy'].to_numpy()
xo = xo-xo[xs]
yo = yo-yo[xs]


yd = np.diff(yo)
xd = np.diff(xo)
v = np.append(0,np.sqrt(xd**2+yd**2))
#disp = np.sqrt(xo**2+yo**2)
#v = np.append(0,np.diff(disp))

plt.figure()
#th = -cxa.pdat['offset_eb_phase'].to_numpy()


th =cxa.ft2['ft_heading'].to_numpy()
#th = cxa.pdat['offset_fsb_upper_phase'].to_numpy()+cxa.pdat['offset_eb_phase'].to_numpy()
#v =cxa.ft2['net_motion'].to_numpy()
tg  = np.append(th[1:],th[-1:])
#cxa.pdat['offset_fsb_upper_phase']
x,y = output_trajectory(v[xs:],th[xs:],tg[xs:],PFL3_meta,gain=0.03)
plt.plot(x,y)

plt.plot(xo[xs:],yo[xs:],color='k')
#%%
from scipy.optimize import curve_fit
in_array = np.zeros((3,len(th)))
in_array[0,:] = v
in_array[1,:] = th
in_array[2,:] = tg
out_array = np.append(xo[xs:],yo[xs:])
popt,pcov = curve_fit(output_trajectory_fit,in_array[:,xs:],out_array,)
print(popt)
#%%
plt.close('all')
w = 0.75
th = np.zeros((100))
plt.subplot(2,2,1)
tadd = np.linspace(-np.pi,np.pi,100)
R1,L1 = PFL3_function(th,th+tadd)
plt.plot(tadd,tadd,color='b')
plt.plot(tadd,R1-L1,color='r')
plt.plot(tadd,(R1-L1)/tadd,color='g')
plt.plot(tadd,th)

plt.subplot(2,2,2)
R,L = PFL3_function_anat(PFL3_meta,th,th+tadd)
Rout,Lout = PFL3_function_anat(PFL3_meta,th,th+tadd,full_output=True)
plt.plot(tadd,tadd,color='b')
plt.plot(tadd,R-L,color='r')
plt.plot(tadd,(R-L)/tadd,color='g')
plt.plot(tadd,th)

plt.subplot(2,2,3)
R2,L2 = PFL2_function(th,th+tadd)
plt.plot(tadd,tadd,color='b')
plt.plot(tadd,R2+L2,color='r')
#plt.plot((R+L)/tadd,color='g')
plt.plot(tadd,th)

plt.subplot(2,2,4)
dna03L = np.exp(L1+L2)-1
dna03R = np.exp(R1+R2)-1
dna03L = dna03L/np.max(dna03L)
dna03R = dna03R/np.max(dna03R)
dna02L = np.exp(L1+dna03L*w)
dna02R =np.exp( R1+dna03R*w)
plt.plot(tadd,dna02R,color='r')
plt.plot(tadd,dna02L,color='b')
plt.plot(tadd,dna03R-dna03L,linestyle='--',color='k')
plt.plot(tadd,dna02R-dna02L,color='k')
#%%
R,L = PFL3_function_anat(PFL3_meta,th,th+tadd)
Rout,Lout = PFL3_function_anat(PFL3_meta,th,th+tadd,full_output=True)
#plt.plot(tadd,tadd,color='b')
plt.figure()
#plt.plot(tadd,(R-L)/tadd,color='g')
prange = [-np.pi/2,-np.pi/4,0,np.pi/4,np.pi/2]
for i,ib in enumerate(prange):
    R,L = PFL3_function_anat(PFL3_meta,th,th+tadd,offset=ib)
    plt.plot(tadd,R-L,color=[i/len(prange),0,0])
    #plt.plot(tadd,th)
plt.ylabel('R_LAL - L_LAL')
plt.xlabel('Goal relative to heading (rad)')
plt.legend(np.array(prange)/np.pi)
Rout,Lout = PFL3_function_anat(PFL3_meta,th,th+tadd,offset=ib,full_output=True)
#%%
plt.close('all')
R,L = PFL3_function(cxa.pdat['offset_eb_phase'],cxa.pdat['offset_fsb_upper_phase'])
x = cxa.ft2['ft_posx']
y = cxa.ft2['ft_posy']
#plt.plot(R,color='r')
#plt.plot(L,color='b')
PFL_turn = (R-L)/(R+L)
plt.plot(PFL_turn,color='k')
angvel = cxa.ft2['ang_velocity']
heading = cxa.ft2['ft_heading']
angvel = angvel/np.max(np.abs(angvel))
cheading = np.cumsum(heading)
#angvel = np.diff(heading)
#angvel = np.append(0,angvel)
#plt.plot(cheading)
ins = cxa.ft2['instrip']
plt.plot(angvel,color='r')
plt.plot(ins,color='b')
plt.figure()
plt.plot(np.pi*(R-L)/(R+L),color='k')
plt.plot(cxa.pdat['offset_fsb_upper_phase'])
plt.plot(cxa.pdat['offset_eb_phase'])
#%%
ldx = PFL3_meta['LAL']<0
x = PFL3_meta['pbglom'][ldx]
y = PFL3_meta['fsb_column'][ldx]
plt.scatter(x,y,color='b')
plt.scatter(PFL3_meta['pbglom'][~ldx],PFL3_meta['fsb_column'][~ldx],color='r')
plt.xlabel('PB glom number')
plt.ylabel('FSB column number')
#%%
x = np.arange(0,10*np.pi,0.01)
plt.plot(np.sin(x))
plt.plot(np.sin(x-0.25*np.pi))
plt.plot(np.sin(x-0.6*np.pi))
plt.plot(np.sin(x-0.25*np.pi)+np.sin(x)+np.sin(x-0.6*np.pi))
#%%
th = cxa.pdat['offset_eb_phase'].to_numpy()

plt.plot(th)
th = cxa.ft2['ft_heading'].to_numpy()
plt.plot(th)