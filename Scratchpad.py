# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:27:58 2024

@author: dowel
"""



from analysis_funs.regression import fci_regmodel

import numpy as np
import pandas as pd
import analysis_funs.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt 
from analysis_funs.utilities import imaging as im
from skimage import io, data, registration, filters, measure
from scipy import signal as sg
from scipy import stats
from analysis_funs.CX_imaging import CX
from analysis_funs.CX_analysis_col import CX_a
from analysis_funs.utilities import funcs as fn
from Utilities.utils_general import utils_general as ug
from scipy.stats import circmean, circstd

plt.rcParams['pdf.fonttype'] = 42 

#%%
pdiff = ug.circ_subtract(cxa.pdat['phase_fsb_upper'],cxa.pdat['phase_eb'])
plt.hist(pdiff,bins=60)
w = cxa.pdat['wedges_fsb_upper']
plt.plot(np.mean(w,axis=0))

w = cxa.pdat['wedges_eb']
plt.plot(np.mean(w,axis=0))
#%% 
plt.close('all')
jumps = cxa.get_jumps()
phase = cxa.pdat['phase_fsb_upper']
phase_eb = cxa.pdat['phase_eb']
for j in jumps:
    dx = np.arange(j[1],j[2])
    plt.figure()
    plt.hist(phase[dx],bins=30)
    plt.figure(101)
    plt.scatter(phase_eb[dx],phase[dx],color='k',s=5)
plt.figure(201)
plt.scatter(phase_eb,phase,color='k',s=5,alpha=0.1)
plt.plot([0,np.pi],[-np.pi,0],color='r')
plt.plot([-np.pi,0],[0,np.pi],color='r')
plt.plot([-np.pi,np.pi],[-np.pi,np.pi],color='r')

#%% phase saccades
plt.close('all')
tp = ug.savgol_circ(cxa.pdat['phase_fsb_upper'],20,3)
tp2 = ug.savgol_circ(cxa.pdat['phase_eb'],20,3)
tpw = np.unwrap(tp)
step = np.ones(20)
step[:10] = -1
pc = np.convolve(tpw,step,mode='same')/20
plt.plot(pc)
x = np.arange(0,len(tp))
peaks,heights = sg.find_peaks(np.abs(pc),height=0.5)
plt.scatter(x,tp,color='r',s=5)
plt.scatter(x,tp2,color='k',s=5)

plt.scatter(x,ug.circ_subtract(tp,np.pi),color=[1,0.5,0.5],s=5)
plt.plot(cxa.ft2['instrip']*3,color='g')
plt.scatter(x[peaks],pc[peaks],color='b')
#%% load data
datadir = r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250714\f1\Trial2"
cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
savedir = "Y:\Data\FCI\FCI_summaries\hDeltaC"
#%% Make movie of phase progression over time


from matplotlib.animation import ImageMagickFileWriter, ImageMagickWriter,FFMpegWriter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Example data (replace with your own)

pe = cxa.pdat['phase_eb']                               # x values
pf = cxa.pdat['phase_fsb_upper']  
phase_eb = cxa.pdat['offset_eb_phase'].to_numpy()
phase_fsb = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
pva = ug.get_pvas(cxa.pdat['wedges_fsb_upper'])
pva = pva/np.max(pva)
pe = ug.savgol_circ(pe,20,3)
pf = ug.savgol_circ(pf,20,3)
xmn = circmean(pe,high=np.pi,low=-np.pi)
pe = ug.circ_subtract(pe,xmn)         #stubtract circular mean to keep data in centre               # y values
pf = ug.circ_subtract(pf,xmn)


x = cxa.ft2['ft_posx'].to_numpy()
y = cxa.ft2['ft_posy'].to_numpy()
x,y = cxa.fictrac_repair(x,y)
ins = cxa.ft2['instrip'].to_numpy()    # example ins (0 or 1)
t = np.linspace(0, 20, len(x))      

jumps = cxa.ft2['jump'].to_numpy()
instrip = cxa.ft2['instrip'].to_numpy()
stripdiff = np.diff(instrip)
stripon = np.where(instrip>0)[0][0]
xs = np.where(instrip==1)[0][0]
strts = np.where(stripdiff>0)[0]
stps = np.where(stripdiff<0)[0]
x = x-x[xs]
y = y-y[xs]           # time points
# Set up the figure and axis
#fig, ax = plt.subplots()


fig, axs = plt.subplots(figsize=(15,8))#,ncols=3,width_ratios=[0.4,0.3,0.3])
axs.set_xticks([])
axs.set_yticks([])
axs.get_yaxis().set_visible(False)
axs.get_xaxis().set_visible(False)
axs.axis("off")

#Phase
ax = plt.subplot2grid((1,2), (0, 0), colspan=1)


# Trajectory
ax2 = plt.subplot2grid((1, 2), (0, 1))
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_aspect('equal')


scat = ax.scatter([], [],color="black", s=50 )
scat2, = ax.plot([],[],color='b')
scat3, = ax.plot([],[],color='b')
scat4 = ax.scatter([],[],color=[0.5,0,0],s = 100)
line, = ax2.plot([],[],lw=2,color='k')
line2, = ax2.plot([],[],lw=3,color=[0.2,0.2,1])
line3, = ax2.plot([],[],lw=3,color=[0.2,0.2,0.2])

ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)
ax.set_xlabel("EB phase")
ax.set_ylabel("FSB phase")
ax.set_title("Phase over time")
ax.plot([0,np.pi],[-np.pi,0],color='r')
ax.plot([-np.pi,0],[0,np.pi],color='r')
ax.plot([-np.pi,np.pi],[-np.pi,np.pi],color='r')


ins = cxa.ft2['instrip'].to_numpy()
jumps = cxa.ft2['jump'].to_numpy()
tt = cxa.pv2['relative_time'].to_numpy()
inplume = ins>0
st  = np.where(ins)[0][0]
x = x-x[st-1]
y = y-y[st-1]
  



jumps = jumps-np.mod(jumps,3)
jd = np.diff(jumps)
jn = np.where(np.abs(jd)>0)[0]+1
print(jumps[jn])
jkeep = np.where(np.diff(jn)>1)[0]

xrange = np.max(x)-np.min(x)
yrange = np.max(y)-np.min(y)

mrange = np.max([xrange,yrange])+100
y_med = yrange/2
x_med = xrange/2
ylims = [y_med-mrange/2, y_med+mrange/2]

xlims = [x_med-mrange/2, x_med+mrange/2]
yj = y[jn]
yj = np.append(yj,y[-1])
tj = 0
x1 = 0+5+tj
x2 = 0-5+tj
y1 = 0
y2 = yj[0]
xvec = np.array([x1,x2,x2,x1])
yvec = [y1,y1,y2,y2]

cents = [-630,-420,-210, 0,210,420,630]
ax2.fill(xvec,yvec,color=[0.7,0.7,0.7])
for c in cents:
    ax2.fill(xvec+c,yvec,color=[0.7,0.7,0.7])
    
for i,j in enumerate(jn):
    
    tj = jumps[j]
    x1 = 0+5+tj
    x2 = 0-5+tj
    y1 = yj[i]
    y2 = yj[i+1]
    xvec = np.array([x1,x2,x2,x1])
    yvec = [y1,y1,y2,y2]
    for c in cents:
        ax2.fill(xvec+c,yvec,color=[0.7,0.7,0.7])


xa = 5*np.sin(phase_fsb)+x
ya = 5*np.cos(phase_fsb)+y
xa2 = 5*np.sin(phase_eb)+x
ya2 = 5*np.cos(phase_eb)+y
# Update function for animation
lastpoint = np.array([np.nan,np.nan])
def update(frame):
    start = max(0, frame - 20)
    scat2.set_data([-np.pi,np.pi],[pf[frame-1],pf[frame-1]])
    scat3.set_data([pe[frame-1],pe[frame-1]],[-np.pi,np.pi])
    scat.set_offsets(np.c_[pe[start:frame], pf[start:frame]])
    line2.set_data([x[frame-1],xa[frame-1] ], [y[frame-1],ya[frame-1]])
    line3.set_data([x[frame-1],xa2[frame-1] ], [y[frame-1],ya2[frame-1]])
    if frame>100:
        line.set_data(x[frame-100:frame], y[frame-100:frame])
        colours = np.tile([0,0,0,1], (20, 1))
        colours[ins[start:frame]>0,0] = 1
        scat.set_color(colours)
    else:
        line.set_data(x[:frame], y[:frame])
        scat.set_color("black")
    # Check ins value at the most recent frame
    
    if np.sum(np.abs(lastpoint))>0:
        scat4.set_offsets([lastpoint[0],lastpoint[1]])
    if frame>1:
        if ins[frame-1]==1 and ins[frame-2]<1:
            lastpoint[0] = pe[frame-1]
            lastpoint[1] = pf[frame-1]
    
    
        
        
        
    alphas = pva[start:frame]
    
    if len(alphas) > 0:
        #alphas = (alphas - np.min(alphas)) / (np.ptp(alphas) + 1e-9)  # normalize 0–1
        colors = np.tile([1,0,0,1], (len(alphas), 1))
        colors[:, 3] = 0.99 # replace alpha channel
    
        scat.set_sizes((alphas*10)**2)
    ax2.set_xlim(x[frame]-10,x[frame]+10)
    ax2.set_ylim(y[frame]-10, y[frame]+10)
    return scat,scat2,scat3,line,line2,line3,scat4
    
    
# Create animation
anim = animation.FuncAnimation(fig, update, frames=len(t), interval=100, blit=True)
#anim = animation.FuncAnimation(fig, update, frames=np.arange(1010,1500), interval=100, blit=True)
plt.show()


writer = FFMpegWriter(fps=20)

path_to_convert = r'C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\convert.exe'
path_to_magick = r'C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe'
anim.save(os.path.join(savedir,'Phase_prog2' + cxa.name +'.avi'), writer=writer)




#%%
datadirs = [
    # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial1",
    # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial2",
    # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial3",
    # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial4",
    # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial5",
    # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial6",
    # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250605\f1\Trial7"
    
    # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250714\f1\Trial1", # A few jumps
    # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250714\f1\Trial2", # Several jumps: good data, though anisotropy in hDeltaC
    # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250714\f1\Trial6", # ACV pulses
    # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250714\f1\Trial7"
    # r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial1",
r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial2",
r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial3",
r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial4",
r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial5",
r"Y:\Data\FCI\Hedwig\hDeltaC_SS02863\250721\f1\Trial6"
]

for d in datadirs:
    cxa = CX_a(d,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    heading = cxa.ft2['ft_heading']
    phase = cxa.pdat['phase_eb']
    plt.figure()
    plt.scatter(heading,phase,s=5)







#%% Sorting out pre air heading in jump suspend
plt.close('all')
ft2 = cxa.ft2_original
ft = cxa.ft
bumpo = ft['bump'].to_numpy()
ubumps = bumpo[np.abs(bumpo)>0]


x = ft2['ft_posx'].to_numpy()
y = ft2['ft_posy'].to_numpy()

x,y = cxa.fictrac_repair(x,y)
th = ft2['train_heading'].to_numpy()
fh = ft2['fix_heading'].to_numpy()
bp = ft2['bump'].to_numpy()
ins = ft2['instrip'].to_numpy()
heading = ft2['ft_heading'].to_numpy()

add_array = np.zeros(len(th))
fh[fh>0] = 1
dfh = np.where(np.diff(fh)>0)[0]
add_array[dfh] =ubumps
add_array = np.cumsum(add_array)

new_heading = ug.circ_subtract(heading,-add_array)
bstart,bsize = ug.find_blocks(add_array>0)
# rotate heading
xnew = x.copy()
ynew = y.copy()
for i,b in enumerate(bstart):
    #plt.figure()
    theta = add_array[b+1]
    dx = np.arange(b,b+bsize[i],dtype=int)
    tx = xnew[dx]
    ty = ynew[dx]
    tx0 = tx-tx[0]
    ty0 = ty-ty[0]

    xy = np.append(tx0[:,np.newaxis],ty0[:,np.newaxis],axis=1)
    #plt.plot(xy[:,0],xy[:,1],color='k')
    rotmat = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    xy = np.matmul(xy,rotmat)
    #plt.plot(xy[:,0],xy[:,1],color='r')
    xn = xy[:,0]+tx[0]
    yn = xy[:,1]+ty[0]
    xnew[dx] = xn
    ynew[dx] = yn
    xnew[dx[-1]+1:] = xnew[dx[-1]+1:]-(xnew[dx[-1]+1]-xnew[dx[-1]])
    ynew[dx[-1]+1:] = ynew[dx[-1]+1:]-(ynew[dx[-1]+1]-ynew[dx[-1]])
    
#xnew,ynew = cxa.fictrac_repair(xnew,ynew)

g = plt.gca()
g.set_aspect('equal')

inst = np.where(ins>0)[0][0]
tx = xnew[:inst]
ty = ynew[:inst]
tfh = fh[:inst]
theading = heading[:inst]

plt.figure()
plt.plot(tx,ty,color='k')

plt.scatter(tx[tfh>0],ty[tfh>0],color='r')
plt.figure()
plt.plot(theading+add_array[:inst])
plt.plot(tfh)










#%% Ring attractor playground
plt.close('all')
fc2 = np.zeros(8)
hdeltaC = np.zeros(8)
heading = np.zeros(8)
randgoal = np.zeros(8)
goaldir = -np.pi/2
heading = 0 
rg = np.linspace(-np.pi,np.pi,16)
initarray = np.linspace(-np.pi,np.pi,16)
hdeltaC = np.cos(initarray-goaldir)+1
heading = np.cos(initarray-heading)+1
hdeltaJ = np.cos(initarray-np.pi)+1
for i, r in enumerate(rg):
    randgoal = np.cos(initarray-r)+1
    
    #ff_inhib = np.mean(randgoal+hdeltaC+heading)
    fc2 = np.exp(randgoal+heading)/10
    
    plt.subplot(4,4,i+1)
    plt.plot(initarray,fc2,color='r')
    plt.plot(initarray,hdeltaC,color='b')
    plt.plot(initarray,heading,color='k')
    plt.plot(initarray,randgoal,color='m')
    plt.ylim([0,25])

hdjw = np.linspace(0,1,5)
for ij ,j in enumerate(hdjw):
    for i, r in enumerate(rg):
        randgoal = np.cos(initarray-r)+1
        fc2 = np.exp(randgoal+heading+hdeltaJ*j)
        plt.figure(ij+2)
        plt.plot(initarray,fc2,color=[i/16,0.5,0.5])
        # plt.figure(ij*10)
        # plt.plot(initarray,randgoal,color=[i/16,0.5,0.5])
        
hdjw = np.linspace(0,1,5)

for ij ,j in enumerate(hdjw):
    plt.figure(100+ij)
    for i, r in enumerate(rg):
        randgoal = np.cos(initarray-r)+1
        fc2 = np.exp(randgoal+heading+hdeltaJ*j)
        am = np.argmax(fc2)
        amin = np.argmin(fc2)
        plt.scatter(initarray[am],fc2[am],color=[i/16,0.5,0.5])
        plt.scatter(initarray[amin],fc2[amin],color=[i/16,0.5,0.5])
        #plt.plot(initarray,fc2,color=[i/16,0.5,0.5])
        
        
plt.figure(1000)
r = -3*np.pi/4
for ij ,j in enumerate(hdjw):
    
    randgoal = np.cos(initarray-r)+1
    fc2 = np.exp(randgoal+heading+hdeltaJ*j)
   
    plt.plot(initarray,fc2,color=[ij/len(hdjw),0.5,0.5])
#%%
labmeetingdir = r'Y:\Presentations\2025\06_LabMeeting\FC2'
plt.close('all')
goaldir = -np.pi/2
heading = 0
hdjdir = -3*np.pi/4
w_hdc = 1.5
w_hdj = 1.5
initarray = np.linspace(-np.pi,np.pi,16)
hdeltaC = np.cos(initarray-goaldir)+1
heading = np.cos(initarray-heading)+1
hdeltaJ = np.cos(initarray-hdjdir)+1

FC2_1 = np.exp(heading)
FC2_2 = np.exp((heading+hdeltaC)/2)
FC2_3 = np.exp((heading+hdeltaJ+hdeltaC)/3)
# FC2_1 = FC2_1/np.mean(FC2_1)
# FC2_2 = FC2_2/np.mean(FC2_2)
# FC2_3 = FC2_3/np.mean(FC2_3)

plt.figure()
plt.plot(initarray,heading,color=[0.5,0.8,0.5])
plt.xlim([-np.pi,np.pi])
plt.xticks(np.linspace(-np.pi,np.pi,5),labels=[-180,-90,0,90,180])
plt.plot([-np.pi/2,-np.pi/2],[0,2],color='r',linestyle='--')
plt.ylabel('activity')
plt.xlabel('phase (deg)')
plt.savefig(os.path.join(labmeetingdir,'hDeltaIn_heading.png'))
plt.plot(initarray,hdeltaC,color=[0.5,0.5,1])
plt.savefig(os.path.join(labmeetingdir,'hDeltahDeltaC.png'))
plt.plot(initarray,hdeltaJ,color=[0.8,0.2,1])

plt.savefig(os.path.join(labmeetingdir,'hDeltaIn.png'))


plt.figure()
plt.plot(initarray,FC2_1,color=[0.5,0.8,0.5])
plt.plot([-np.pi/2,-np.pi/2],[0,7],color='r',linestyle='--')
plt.xticks(np.linspace(-np.pi,np.pi,5),labels=[-180,-90,0,90,180])
plt.plot([-np.pi/2,-np.pi/2],[0,2],color='r',linestyle='--')
plt.ylabel('activity')
plt.xlabel('phase (deg)')
plt.savefig(os.path.join(labmeetingdir,'FC2out_heading.png'))
plt.plot(initarray,FC2_2,color=[0.5,0.5,1])
plt.savefig(os.path.join(labmeetingdir,'FC2out_hdc.png'))
plt.plot(initarray,FC2_3,color=[0.8,0.2,1])
plt.savefig(os.path.join(labmeetingdir,'FC2out_hdc_hdj.png'))

plt.figure()
hdirs = np.linspace(-np.pi,np.pi,17)
lnorm = len(hdirs)
for ih,h in enumerate(hdirs):
    heading = np.cos(initarray-h)+1
    FC2_3 = np.exp((heading+hdeltaJ+hdeltaC)/3)
    #plt.plot(initarray,FC2_3,color=[1-ih/4,0,ih/4])
    
    psin = np.mean(FC2_3*np.sin(initarray))
    pcos = np.mean(FC2_3*np.cos(initarray))
    plt.plot([0,psin],[0,pcos],color=[1-ih/lnorm,0,ih/lnorm])
    
    xs =np.array([0, np.sin(h)])+2
    xc = np.array([0,np.cos(h)])
    plt.plot(xs,xc,color =[1-ih/lnorm,0,ih/lnorm])
gca = plt.gca()
gca.set_aspect('equal')
#plt.plot([-np.pi/2,-np.pi/2],[0,7],color='r',linestyle='--')
#%% Image registraion


datadir =os.path.join(r"F:\2p\LadyBird\prairie\60D05-sytGCa7f\02282025\imaging\tapping_test-M_60D05-GcaMP7f_volume_D3-FD4-1-1-001")
d = datadir.split("\\")
name = d[-3] + '_' + d[-2] + '_' + d[-1]
#% Registration
ex = im.fly(name, datadir)
ex.register_all_images(overwrite=True)
ex.z_projection()
#%
ex.t_projection_mask_slice()

#%% Phase analysis
regions = ['pb']
d = datadir.split("\\")
name = d[-3] + '_' + d[-2] + '_' + d[-1]

cx = CX(name,regions,datadir)
# save preprocessing, consolidates behavioural data
cx.save_preprocessing()
# Process ROIs and saves csv
cx.process_rois()
# Post processing, saves data as h5
cx.crop = False
cx.save_postprocessing()#upsample to 50Hz
pv2, ft, ft2, ix = cx.load_postprocessing()


cxa = CX_a(datadir,regions=regions)    
cxa.save_phases()


#%%
import src.utilities.funcs as fc
file_path = os.path.join(datadir,'data','fictrac-20250228_112318.log')
#df = pd.read_table(file_path, delimiter='[,]', engine='python')
df = pd.read_table(file_path, delimiter='[,]', engine='python')

df = fc.read_log(file_path)













#%%
wedges = cxa.pdat['wedges_pb']
wedges_pb = np.zeros_like(wedges)
wedges_pb[:,cxa.logic2anat] = wedges
plt.imshow(wedges_pb, interpolation='None',aspect='auto',cmap='Blues')

from scipy import fftpack
n = 100
axlen = wedges_pb.shape[-1]*n
epg_fft = fftpack.fft(wedges_pb, axlen, -1)
power = np.abs(epg_fft)**2
freq = fftpack.fftfreq(axlen, 1/n)/n
phase = np.angle(epg_fft)
midpoint = int(freq.size/2)
freq = freq[1:midpoint]
period = (1./freq)
power = power[:, 1:midpoint]
phase = phase[:, 1:midpoint]
ix = np.where(period==8)
phase_8 = phase[:,ix].flatten()












#%% Get odour pulse statistics

import numpy as np
from Utils.utils_general import utils_general as ug
import src.utilities.funcs as fc
import os
from analysis_funs.optogenetics import opto 
import matplotlib.pyplot as plt 
datadirs = [
    "Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f1\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f2\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240502\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240514\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\241104\\f1\\Trial5",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\241106\\f1\\Trial2",
"Y:\Data\\FCI\\Hedwig\\SS70711_FB4X\\241030\\f3\\Trial3",
"Y:\\Data\\FCI\\Hedwig\\SS70711_FB4X\\241031\\f1\\Trial3",
"Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553\\240628\\f1\\Trial2",#Nice
         "Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553\\240917\\f1\\Trial2",#Best for this fly
        "Y:\\Data\\FCI\\Hedwig\\FB5I_SS100553\\240917\\f3\\Trial3",
        "Y:\Data\FCI\\Hedwig\\SS61646_FB4R\\240828\\f3\\Trial1",
        "Y:\Data\FCI\\Hedwig\\SS61646_FB4R\\240910\\f1\\Trial1",
        "Y:\\Data\\FCI\\Hedwig\\FB4P_b_SS60296\\240912\\f2\\Trial3",
        "Y:\Data\FCI\\Hedwig\\FB4P_b_SS60296_sytGC7f\\240809\\f2\\Trial2",
        "Y:\\Data\FCI\\Hedwig\\FB5AB_SS53640\\241205\\f2\\Trial3",
        "Y:\\Data\\FCI\\Hedwig\\hDeltaJ\\240529\\f1\\Trial3",
        "Y:\\Data\\FCI\\Hedwig\\hDeltaA_SS64464\\241121\\f2\\Trial1",
        "Y:\\Data\\FCI\\Hedwig\\hDeltaI_SS60919\\241204\\f1\\Trial2"
        ]
btall = np.array([])
isi_all = np.array([])
savedir = "Y:\\Data\FCI\\ConsolidatedData\\OdourPulseData"
for d in datadirs:
    print(d)
    searchdir = os.path.join(d,'data')
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    
    
    df = fc.read_log(datadir)
    
    op = opto()
    tt= op.get_time(df)
    dt = np.mean(np.diff(tt))
    
    
    ins = df['instrip']
    blk = ug.find_blocks(ins)
    blke = blk[0]+blk[1]
    dblk = blk[0][1:]-blke[:-1]
    
    dblk = dblk*dt
    dblk = dblk[dblk>0.5]
    bt = blk[1]*dt
    bt = bt[bt>0.5]
    btall = np.append(btall,bt)
    isi_all = np.append(isi_all,dblk)
savedict = {'odour_on':btall,'isi':isi_all}
from Utils.utils_general import utils_general as ug
ug.save_pick(savedict,os.path.join(savedir,'odour_pulses.pkl'))

#%%
from scipy.optimize import curve_fit
def modfun(x,a,b,c):
    return a*np.exp(-x*b)+c

bins = np.arange(0.5,20,0.5)
plt.hist(btall,bins=bins)

counts,binedges = np.histogram(btall, bins=bins)
x = bins[1:]-0.25
plt.plot(x,counts)
ft = np.polyfit(np.log(x),counts, 2)
popt,pcov = curve_fit(modfun,x,counts)
yp = modfun(x,popt[0],popt[1],popt[2])
plt.plot(x,yp)

#%%
bins = np.arange(0.5,100,0.5)
plt.hist(isi_all,bins=bins)
counts,binedges = np.histogram(isi_all, bins=bins)
x = bins[1:]-0.25
plt.plot(x,counts)
ft = np.polyfit(np.log(x),counts, 2)
popt,pcov = curve_fit(modfun,x,counts)
yp = modfun(x,popt[0],popt[1],popt[2])
plt.plot(x,yp)



#%%
import numpy as np
files = f[1]
import time
import cv2
t0 = time.time()

for i,file in enumerate(files):
    print(i)
    im = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    if i ==0:
        images = np.zeros((len(files),np.shape(im)[0],np.shape(im)[1]),dtype = 'uint16')
    images[i,:,:] = im
print(f'{time.time()-t0:.2f} s')



#%%




from analysis_funs.CX_imaging import CX
import os
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from scipy.stats import circmean, circstd
from src.utilities import funcs as fn
import pickle
from scipy.optimize import curve_fit
from analysis_funs.CX_phase_modelling import CX_phase_modelling
from analysis_funs.CX_analysis_col import CX_a
#%%
datadir = "Y:\Data\FCI\Hedwig\FC2_maimon2\\240502\\f1\\Trial2"
cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
#
#%%
cxp = CX_phase_modelling(cxa)
phase = cxa.pdat['phase_eb']

x= np.ones((2,len(phase)))
x[0,:] = cxa.ft2['ft_heading']
w1 = 1 
w2 = 2
weights = np.array([w1,w2])

#cxp.phase_function(x,w1,w2)

cxp.fit_phase_function(x,phase)

plt.scatter(x[0,:],phase)
plt.scatter(x[0,:],cxp.phase_function(x,*cxp.popt))
#%%
plt.close('all')
phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
#phase = cxa.pdat['offset_eb_phase']
#phase = cxa.ft2['ft_heading'].to_numpy()
x= np.zeros((3,len(phase)))
x[0,:] = cxa.pdat['offset_eb_phase']
x[1,:] = cxp.plume_memory()

cxp.fit_phase_function(x,phase)
plt.plot(phase)
plt.plot(cxp.phase_function(x,*cxp.results.x))
plt.figure()
plt.scatter(phase,cxp.phase_function(x,*cxp.results.x),s=1)
#%%
plt.close('all')
x= np.zeros((3,len(phase)))
x[0,:] = cxa.pdat['offset_eb_phase']
x[1,:] = cxp.plume_memory()
parts = ['Pre Air','Returns','Jump Returns','In Plume']
cxp.fit_in_parts(x,phase,parts)
#cxp.reg_in_parts(x,phase,['Pre Air','Returns','Jump Returns','In Plume'])
popt_array = cxp.popt_array
popt_array = popt_array/np.expand_dims(np.max(np.abs(popt_array),axis=0),axis=0)
plt.figure()
plt.plot(popt_array)
plt.legend(parts)
#%%
phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
plt.close('all')
phase_eb = cxa.pdat['offset_eb_phase']
#phase_eb = cxa.ft2['ft_heading'].to_numpy()
parts = ['Pre Air','Returns','Jump Returns','In Plume']
dx = cxp.output_time_epochs(cxa.ft2,'Jump Returns')
xm = cxp.plume_memory()

ddiff = np.diff(dx)
e_end = np.where(ddiff>1)[0]
e_end = np.append(e_end,len(ddiff)-1)
estart = np.append(1,e_end[:-1]+1)
endx = dx[e_end]
stdx = dx[estart]
times = cxa.pv2['relative_time'].to_numpy()
x = cxa.ft2['ft_posx'].to_numpy()
y = cxa.ft2['ft_posy'].to_numpy()
x,y = cxa.fictrac_repair(x,y)
amp = cxa.pdat['amp_fsb_upper']
pamp = np.percentile(amp,99)
amp[amp>0.2] = 0.2 
amp = amp/0.2

p_scat = np.zeros((len(endx),2))
for i,e in enumerate(endx):

    tdc = np.arange(stdx[i],e,1,dtype=int)
    t = times[tdc]
    t = t-t[0]
    tx = x[tdc]
    ty = y[tdc]
    tx = tx-tx[0]
    ty = ty-ty[0]
    tphase = phase[tdc]
    d = np.sqrt(tx**2+ty**2)
    
    ttdx = (t-max(t))>-1
    p_scat[i,0] = circmean(tphase[ttdx],high=np.pi,low=-np.pi)
    p_scat[i,1] = xm[tdc[-1]]
    # plt.figure()
    # plt.plot(tx,ty)
    # ax = plt.gca()
    # ax.set_aspect('equal', adjustable='box')
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(tphase, t,color='k')
    #ax.plot(phase[tdc],t,color=[0.2,0.2,0.8])
    ax.scatter(phase[tdc],t,s=amp[tdc]*100,color=[0.2,0.2,0.8],alpha=amp[tdc])
    ax.plot(xm[tdc],t,color='r')
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    #ax.set_ylim([0,60])
    # plt.figure()
    # plt.plot(phase_eb[tdc],phase[tdc],color='k')
    # plt.scatter(phase_eb[tdc],phase[tdc],c=tdc-tdc[0],s=10,zorder=10)
    # plt.plot([-np.pi,np.pi],[-np.pi,np.pi],color='k')
    # plt.scatter(phase_eb[tdc[0]],x[1,tdc[0]],color='r',zorder=11)
plt.figure()
plt.scatter(p_scat[:,1],p_scat[:,0])
plt.xlim([-np.pi,np.pi])
plt.ylim([-np.pi,np.pi])
plt.plot([0,0],[-np.pi,np.pi],color='k')
plt.plot([-np.pi,np.pi],[0,0],color='k')
plt.plot([-np.pi,np.pi],[-np.pi,np.pi],color='k',linestyle='--')
#%% Scatter of last phase before re-entry vs remembered angle
datadirs = [
    "Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f1\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f2\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240502\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240514\\f1\\Trial2"]
savedir= "Y:\Data\FCI\FCI_summaries\FC2_maimon2"
plt.close('all')
plt.figure()
colours = np.array([[166,206,227],
[202,178,214],
[51,160,44],
[251,154,153],
[227,26,28],
[253,191,111],
[255,127,0],
[31,120,180],
[178,223,138]
])/255
for i,datadir in enumerate(datadirs):
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    cxp = CX_phase_modelling(cxa)
    phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
    p_scat = cxp.phase_memory_scatter(phase)
    p_smn = circmean(p_scat,high=np.pi,low=-np.pi,axis=0)
    p_scat = 180*p_scat/np.pi
    p_smn = 180*p_smn/np.pi
    plt.scatter(p_scat[:,1],p_scat[:,0],s=20,color=colours[i,:],zorder=9,alpha=0.5)
    plt.scatter(p_smn[1],p_smn[0],marker='+',s=200,color=colours[i,:],zorder=10)
plt.xlim([-180,180])
plt.ylim([-180,180])
plt.plot([0,0],[-180,180],color='k')
plt.plot([-180,180],[0,0],color='k')
plt.plot([-180,180],[-180,180],color='k',linestyle='--')
plt.plot([-180/2,-180/2],[-180,0],color='r',linestyle='--')
plt.plot([-180,0],[-180/2,-180/2],color='r',linestyle='--')
plt.xlabel('Prior return angle (deg)')
plt.ylabel('FC2 phase 2s return (deg)')
plt.xticks(np.arange(-180,270,90))
plt.yticks(np.arange(-180,270,90))
plt.savefig(os.path.join(savedir,'Scatter_returnVsMemFC2.png'))
#%%
datadirs = [
    "Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f1\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f2\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240502\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240514\\f1\\Trial2"]
plt.close('all')
x_offset = 0
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

for i,datadir in enumerate(datadirs):
    d = datadir.split("\\")
    name = d[-3] + '_' + d[-2] + '_' + d[-1]
    cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
    cxp = CX_phase_modelling(cxa)
    phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
    
    if i==0:
        pltmean = cxp.mean_phase_polar(phase,succession=[fig,ax])
        pltmean = np.expand_dims(pltmean,2)
    else:
        pm = cxp.mean_phase_polar(phase,succession=[fig,ax])
        pm = np.expand_dims(pm,2)
        pltmean = np.append(pltmean,pm,2)

    
#%%
from scipy.stats import circmean, circstd

pltmn = circmean(pltmean,high=np.pi, low=-np.pi,axis=2)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
tstandard = np.linspace(0,49,50)
colours = np.array([[0.2,0.2,0.8],[0, 0, 0],[1,0,0]])
for i in range(3):
    for i2 in range(np.shape(pltmean)[2]):
        ax.plot(pltmean[:,i,i2],tstandard,color=colours[i,:],alpha=0.25) 
        

for i in range(3):
    ax.plot(pltmn[:,i],tstandard,color=colours[i,:],linewidth=2) 
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_xticklabels([0,45,90,135,180,-135,-90,-45])
ax.set_title('FC2 Jump Returns')
#%% 
datadir = "Y:\\Data\\FCI\\Hedwig\\hDeltaJ\\240529\\f1\\Trial3"
cxa = CX_a(datadir,regions=['eb','fsb_upper','fsb_lower'],denovo=False)
cxp = CX_phase_modelling(cxa)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
phase = cxa.pdat['offset_fsb_upper_phase'].to_numpy()
pltmean = cxp.mean_phase_polar(phase,succession=[fig,ax])
ax.set_xticklabels([0,45,90,135,180,-135,-90,-45])
ax.set_title('hDeltaJ Jump Returns')
#%% 
et_dir = [-1,-1,1,1,1,-1]
rootdir = 'Y:\\Data\\FCI\\AndyData\\hDeltaC_imaging\\csv'
folders = ['20220517_hdc_split_60d05_sytgcamp7f',
 '20220627_hdc_split_Fly1',
 '20220627_hdc_split_Fly2',
 '20220628_HDC_sytjGCaMP7f_Fly1',
 #'20220628_HDC_sytjGCaMP7f_Fly1_45-004', 45 degree plume
 '20220629_HDC_split_sytjGCaMP7f_Fly1',
 '20220629_HDC_split_sytjGCaMP7f_Fly3']
plt.close('all')
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
for i,f in enumerate(folders):
    datadir = os.path.join(rootdir,f,"et")
    cxa = CX_a(datadir,Andy='hDeltaC')
    
    cxp = CX_phase_modelling(cxa)
    cxp.side = et_dir[i]
    phase = cxa.pdat['offset_fsb_phase']
    if i==0:
        pltmean = cxp.mean_phase_polar(phase,succession=[fig,ax],part='Returns')
        pltmean = np.expand_dims(pltmean,2)
    else:
        pm = cxp.mean_phase_polar(phase,succession=[fig,ax],part='Returns')
        pm = np.expand_dims(pm,2)
        pltmean = np.append(pltmean,pm,2)
#%%
pltmn = circmean(pltmean,high=np.pi, low=-np.pi,axis=2)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
tstandard = np.linspace(0,49,50)
colours = np.array([[0.2,0.2,0.8],[0, 0, 0],[1,0,0]])
for i in range(3):
    for i2 in range(np.shape(pltmean)[2]):
        ax.plot(pltmean[:,i,i2],tstandard,color=colours[i,:],alpha=0.25) 
        

for i in range(3):
    ax.plot(pltmn[:,i],tstandard,color=colours[i,:],linewidth=2) 
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_xticklabels([0,45,90,135,180,-135,-90,-45])
ax.set_title('hDeltaC Returns')
#%% Script to resave tiffs with appropriate file name
datadir = "Y:\Data\FCI\Hedwig\FC2_maimon2\240911\\f2\\Trial1\\data\\TSeries-09112024-1249-009"









