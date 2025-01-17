# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:17:43 2025

@author: dowel

Aim:
    Use Andy's data to classify ET into different types
    



"""

#%%
import numpy as np
import pandas as pd
import src.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt
from Utils.utils_general import utils_general as ug
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import locally_linear_embedding as LLE
import dtw
#%%

"""
Metrics: 
    Distance from plume
    Entries per m
    Plume velocity
    Time to returns
    Tortuosity of returns
    Return regularity

Analysis:
    UMAP embedding
    PCA
    SVM - with good/bad characterisation
    
"""

#%%
from analysis_funs.behaviour_anyl import ET_behaviour as ETB
rootdir = r'Y:\Data\Behaviour\AndyVanilla\constant_plume_AS'

# Get 

files = os.listdir(rootdir)
plt.close('all')
tdict = {}

for i,f in enumerate(files):
    print(f)
    if '.log' in f:
        savepath = os.path.join(rootdir,f)
        ft = fc.read_log(savepath)
        etb = ETB(ft)
        tdat = etb.extract_metrics()
        if i==1:
            data = tdat[np.newaxis,:]
            trjdata = etb.eigen_return()
        else:
            data = np.append(data,tdat[np.newaxis,:],axis=0)
            trjdata = np.append(trjdata,etb.eigen_return(),axis=0)
        x = etb.x
        y = etb.y
        arr = np.append(x[:,np.newaxis],y[:,np.newaxis],axis=1)
        tdict.update({'f'+str(i):arr})
        

tdict.update({'Combined_data':data})
#%%
plt.close('all')
pca = PCA(n_components=800)
trjx = trjdata[:,0:400]
trjy = trjdata[:,400:]
pctraj = trjdata
#pctraj = (trjdata-np.mean(trjdata,axis=0))/np.std(trjdata,axis=0)
pctraj[:,400] = 0# starting point
pctraj[:,0] = 0
pca.fit(pctraj)
comps = pca.components_
compsx = comps[:,0:400]
compsy = comps[:,400:]
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.plot(compsx[i,1:],compsy[i,1:])
    
expl = pca.explained_variance_ratio_
pcs = np.cumsum(expl)
topcs = pcs<0.90
topcomps = comps[topcs,:]
scores = np.matmul(pctraj,topcomps.T)
# plt.figure()
# plt.subplot(2,2,1)
# plt.scatter(scores[:,0],scores[:,1],s=10)
# plt.subplot(2,2,2)
# plt.scatter(scores[:,2],scores[:,1],s=10)
# plt.subplot(2,2,3)
# plt.scatter(scores[:,3],scores[:,4],s=10)
# plt.subplot(2,2,4)

recapit = np.matmul(scores,topcomps)
x = pctraj[:,0:400]
y = pctraj[:,400:]

rcapx = recapit[:,0:400]
rcapy = recapit[:,400:]
plt.figure()
rands = np.random.randint(0,len(pctraj),16)
for i,r in enumerate(rands):
    plt.subplot(4,4,i+1)
    plt.plot(x[r,:],y[r,:],color='k')
    plt.plot(rcapx[r,:],rcapy[r,:],color='r')
    plt.xticks([])
    cc = np.corrcoef(pctraj[r,:],recapit[r,:])[0,1]
    #plt.title(np.round(cc,decimals=3))
    plt.title(np.round(scores[r,:],decimals=2))
    
    
#%% save top components
ug.save_pick(topcomps,os.path.join(rootdir,'AndyDataReturnPCs_98.pkl'))
#%% Umap on scores
plt.close('all')
reducer  = umap.UMAP()
uscores = (scores-np.mean(scores,axis=0))/np.std(scores,axis=0)
embedding = reducer.fit_transform(uscores)

embedding,_ = LLE(uscores,n_neighbors=10, n_components=2)
embedding = embedding-np.min(embedding,axis=0)

rands = np.random.randint(0,len(pctraj),30)


for i,r in enumerate(rands):
    plt.figure()
    plt.subplot(1,2,1)
    plt.scatter(embedding[:,0],embedding[:,1],s=10,color=[0.6,0.6,0.6])
    plt.scatter(embedding[r,0],embedding[r,1],color='r')
    plt.xscale('log')
    plt.yscale('log')
    plt.subplot(1,2,2)
    plt.plot(x[r,:],y[r,:],color='k')
    plt.plot(rcapx[r,:],rcapy[r,:],color='r')
    plt.xticks([])
    cc = np.corrcoef(pctraj[r,:],recapit[r,:])[0,1]
    #plt.title(np.round(cc,decimals=3))
    plt.title(np.round(scores[r,:],decimals=2))
#%%
plt.close('all')
ds = data.shape
varnames = ['Distance from plume ',
            'Var in distance from plume',
'Entries per m ',
'Plume velocity ',
'Time to returns ',
'Variance in time of returns ',
'Number of RDP segments per return',
'Len last entry ',
'Side fidelity ']

for i in range(ds[1]):
    td = data[:,i]
    plt.subplot(3,3,i+1)
    plt.hist(td)
    plt.title(varnames[i])
#%%
cludata= (data-np.mean(data,axis=0))/np.std(data,axis=0)
ds = data.shape
pca = PCA(n_components=ds[1])
pca.fit(cludata)    
proj = np.matmul(cludata,pca.components_.T)
plt.plot(pca.components_[1,:])
plt.plot(pca.components_[2,:])
plt.figure()
plt.scatter(proj[:,0],proj[:,1])
#%%
plt.close('all')
bad = [35,23,20,25,19,14,13,8]
good = [3,7,10,18,21,27,37,36,34,32,28,26,12,6,4,0]
other = [30,29,24,22,17,11,2,1,0,33]
loopy = [16,19,5,15,29,9]
badmean = np.mean(data[bad,:],axis=0)
goodmean = np.mean(data[good,:],axis=0)
loopmean = np.mean(data[loopy,:],axis=0)

bm = badmean/goodmean
lm = loopmean/goodmean
plt.plot(bm,color='r')
plt.plot(lm,color='m')
plt.plot([0,len(bm)],[1,1],color='k',linestyle='--')
plt.xticks(np.arange(0,len(bm)))
# plt.xticks(np.arange(0,len(bm)),labels=['Distance from plume ',
#             'Var in distance from plume',
# 'Entries per m ',
# 'Plume velocity ',
# 'Time to returns ',
# 'Variance in time of returns ',
# '# RDP segments per return',
# 'Len last entry ',
# 'Side fidelity ','leave mse','return mse','summed angle','time inside'],rotation=90)
plt.subplots_adjust(bottom=0.5)
#%%
plt.close('all')

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LD
clf = LD()
dx = good.copy()
dx.extend(bad)
dx.extend(loopy)
labels = np.ones(len(good),dtype=int)
labels = np.append(labels,np.ones(len(bad),dtype=int)*2)
labels = np.append(labels,np.ones(len(loopy),dtype=int)*3)
lda_data = data
#lda_data = lda_data[:,[2,5,9,10,11,12]]

clf.fit(lda_data[dx,:],labels)
clf.get_params()

proj = np.matmul(lda_data,clf.coef_.T)-clf.intercept_
plt.scatter(proj[:,0],proj[:,1])
plt.scatter(proj[good,0],proj[good,1],color='g')
plt.scatter(proj[bad,0],proj[bad,1],color='r')
plt.scatter(proj[loopy,0],proj[loopy,1],color='m')
plt.scatter(proj[other,0],proj[other,1],color=[0.5,0.5,0.5])


# test usefulness of LDA on other data
#%%
plt.close('all')
testdir = r'Y:\Data\Optogenetics\37G12_PFR_a\37G12_PFR_a_inhibition\Test'
flies = [
     "241013\\f1\\Trial1",
    "241013\\f2\\Trial1",
    "241013\\f3\\Trial1",
    "241014\\f1\\Trial1",
    "241014\\f2\\Trial1",
    "241014\\f3\\Trial1",
    "241014\\f4\\Trial1",
    "241014\\f5\\Trial1",
    "241015\\f4\\Trial2",
    "241015\\f5\\Trial2",
    "241015\\f6\\Trial2",
    
    "241023\\f1\\Trial1",#0 thresh
    "241023\\f2\\Trial1",# crap walker
    
    "241015\\f4\\Trial1",# led off
    "241015\\f5\\Trial1",# crap walker
    
    "241107\\f1\\Trial1",# 0 thresh
    "241107\\f3\\Trial1",# led off
    "241107\\f4\\Trial1",# -1000 mm threshold
    
    "241114\\f1\\Trial1",
    "241114\\f3\\Trial1",
    "241114\\f5\\Trial1",
    
    ]
colours =['g','r','m']
titles = ['good','bad','loopy']
for f in flies:
    searchdir = os.path.join(testdir,f)
    indir = os.listdir(searchdir)
    datadir= os.path.join(searchdir,indir[0])
    files = os.listdir(datadir)
    savepath = os.path.join(datadir,files[0])
    ft = fc.read_log(savepath)
    etb = ETB(ft)
    tdat = etb.extract_metrics()
    tdat = tdat[np.newaxis,:]
    pval = clf.predict(tdat)[0]
    plt.figure()
    plt.subplot(2,2,(1,3))
    etb.plot_traj()
    plt.title(titles[pval-1])
    
    tproj = np.matmul(tdat,clf.coef_.T)-clf.intercept_
    
    plt.subplot(2,2,2)
    plt.scatter(proj[:,0],proj[:,1],color='k')
    plt.scatter(proj[good,0],proj[good,1],color='g')
    plt.scatter(proj[bad,0],proj[bad,1],color='r')
    plt.scatter(proj[loopy,0],proj[loopy,1],color='m')
    plt.scatter(tproj[0][0],tproj[0][1],color='b',marker='+')
    
    
    plt.subplot(2,2,4)
    plt.scatter(proj[:,1],proj[:,2],color='k')
    plt.scatter(proj[good,1],proj[good,2],color='g')
    plt.scatter(proj[bad,1],proj[bad,2],color='r')
    plt.scatter(proj[loopy,1],proj[loopy,2],color='m')
    plt.scatter(tproj[0][1],tproj[0][2],color='b',marker='+')
    
#%%
plt.close('all')
for i in range(38):
    xy = tdict['f'+str(i+1)]
    plt.figure()
    
    plt.subplot(2,2,(1,3))
    plt.fill([-25,25,25,-25],[0,0,1000,1000],color=[0.8,0.8,0.8])
    plt.plot(xy[:,0],xy[:,1],color='k')
    plt.gca().set_aspect('equal')
    plt.title(i)
    
    plt.subplot(2,2,2)
    plt.scatter(proj[:,0],proj[:,1],color='k')
    plt.scatter(proj[good,0],proj[good,1],color='g')
    plt.scatter(proj[bad,0],proj[bad,1],color='r')
    plt.scatter(proj[loopy,0],proj[loopy,1],color='m')
    plt.scatter(proj[i,0],proj[i,1],color='b',marker='+')
    
    plt.subplot(2,2,4)
    plt.scatter(proj[:,1],proj[:,2],color='k')
    plt.scatter(proj[good,1],proj[good,2],color='g')
    plt.scatter(proj[bad,1],proj[bad,2],color='r')
    plt.scatter(proj[loopy,1],proj[loopy,2],color='m')
    plt.scatter(proj[i,1],proj[i,2],color='b',marker='+')
    
#%%
plt.close('all')
bad = [35,23,26,20,19,14,13,8,19]
good = [3,7,10,18,21,27,37,36,34,32,28,26,12,6,4]
other = [31,30,29,24,22,17,11,2,1,0,33]
loopy = [16,5,15]
reducer  = umap.UMAP()
embedding = reducer.fit_transform(cludata)
plt.scatter(embedding[:,0],embedding[:,1])
plt.scatter(embedding[good,0],embedding[good,1],color='g')
plt.scatter(embedding[bad,0],embedding[bad,1],color='r')
plt.scatter(embedding[loopy,0],embedding[loopy,1],color='m')
plt.scatter(embedding[other,0],embedding[other,1],color=[0.5,0.5,0.5])

plt.figure()

plt.subplot(1,2,1)
plt.scatter(proj[:,0],proj[:,1])
plt.scatter(proj[good,0],proj[good,1],color='g')
plt.scatter(proj[bad,0],proj[bad,1],color='r')
plt.scatter(proj[loopy,0],proj[loopy,1],color='m')
plt.scatter(proj[other,0],proj[other,1],color=[0.5,0.5,0.5])

plt.subplot(1,2,2)
plt.scatter(proj[:,2],proj[:,3])
plt.scatter(proj[good,2],proj[good,3],color='g')
plt.scatter(proj[bad,2],proj[bad,3],color='r')
plt.scatter(proj[loopy,2],proj[loopy,3],color='m')
plt.scatter(proj[other,2],proj[other,3],color=[0.5,0.5,0.5])
#%%


#%% Scratchpad


#%% dynamic time warping
plt.close('all')
from dtw import *
xmean = np.mean(trjx,axis=0)
xmean = xmean/np.max(xmean)
ymean = np.mean(trjy,axis=0)
ymean = ymean/np.max(ymean)
tt = np.linspace(0,1,len(xmean))
rands = np.random.randint(0,len(pctraj),36)

plt.figure()
for i,r in enumerate(rands):
    plt.subplot(6,6,i+1)

    
    


    xtest = trjx[r,:]
    xtest= xtest-xtest[0]
    ytest = trjy[r,:]
    dxm = np.argmax(xtest)
    xmax = xtest[dxm]
    ymax = ytest[dxm]
    # y = mx
    m1 = ymax/xmax
    
    ypred1 = xtest[:dxm]*m1

    # y = mx +c
    c = ytest[dxm]
    m2 = (ytest[-1]-ymax)/(xtest[-1]-xmax)
    ypred2 = (xtest[dxm:]-xtest[dxm])*m2+c
    
    plt.plot(xtest,ytest,color='r')
    plt.plot(xtest[:dxm],ypred1,color='k')
    plt.plot(xtest[dxm:],ypred2,color='k')
    
    mse_leave = np.mean((ypred1-ytest[:dxm])**2)
    mse_return = np.mean((ypred2-ytest[dxm:])**2)
    
    mse =np.mean(np.append((ypred1-ytest[:dxm])**2,(ypred2-ytest[dxm:])**2))
    plt.xticks([])
    plt.yticks([])
    plt.title(' MSE return: ' + str(np.round(mse_return,decimals=3)))#'MSE leave: ' +str(np.round(mse_leave,decimals=3)) +
    # xtest = xtest/np.max(xtest)

    # ytest = trjy[r,:]
    # ytest = ytest/np.max(ytest)
    
    
    
    
    # plt.figure()
    # plt.subplot(2,2,1)
    # plt.plot(xtest,color='r')
    # plt.plot(xmean,color='k')
    
    # plt.subplot(2,2,2)
    # plt.plot(ytest,color='r')
    # plt.plot(ymean,color='k')
    
    # plt.subplot(2,2,3)
    # plt.plot(xtest,ytest,color='r')
    # plt.plot(xmean,ymean,color='k')
# alignment = dtw(xtest,xmean,keep_internals=True)


# plt.plot(xtest,color='b')
# plt.plot(xmean,color='k')
# plt.plot(xmean[alignment.index2])
# plt.plot(xmean[alignment.index1])

#%%
## A noisy sine wave as query
idx = np.linspace(0,6.28,num=100)
query = np.sin(idx) + np.random.uniform(size=100)/10.0

## A cosine is for template; sin and cos are offset by 25 samples
template = np.cos(idx)

## Find the best match with the canonical recursion formula
from dtw import *
alignment = dtw(query, template, keep_internals=True)

## Display the warping curve, i.e. the alignment curve
alignment.plot(type="threeway")

## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
dtw(query, template, keep_internals=True, 
    step_pattern=rabinerJuangStepPattern(6, "c"))\
    .plot(type="twoway",offset=-2)

## See the recursion relation, as formula and diagram
print(rabinerJuangStepPattern(6,"c"))
rabinerJuangStepPattern(6,"c").plot()
#%%
plt.plot(query)
#plt.plot(template,color='k')
plt.plot(template[alignment.index2s],color='r')

#%%

etb = ETB(ft)
etb.extract_metrics()
#%%
plt.close('all')
from rdp import rdp
arr = np.append(x[:,np.newaxis],y[:,np.newaxis],axis=1)


for i,en in enumerate(ex_en):
    if en[1]==-1:
        break
    plt.figure()
    plt.plot(arr[:,0],arr[:,1])
    dx = np.arange(en[0],en[1])
    tarr = arr[dx,:]
    m = rdp(tarr,epsilon=10,return_mask=True)
    o = tarr[m,:]
    plt.plot(o[:,0],o[:,1])
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.title(np.sum(m)-1)




