#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 22:15:00 2020

@author: suraj
"""

import numpy as np
from numpy.random import seed

from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt 
import time as tm
import matplotlib.ticker as ticker

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from scipy.stats import norm 
from keras import backend as kb
from sklearn.preprocessing import MinMaxScaler
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.model_selection import train_test_split
from keras.regularizers import l2

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
def coeff_determination(y_true, y_pred):
    SS_res =  kb.sum(kb.square(y_true-y_pred ))
    SS_tot = kb.sum(kb.square( y_true - kb.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + kb.epsilon()) )

def rhs_dx_dt(ne,u,ys,fr,h,c,dt):
    v = np.zeros(ne+3)
    v[2:ne+2] = u
    v[1] = v[ne+1]
    v[0] = v[ne]
    v[ne+2] = v[2]
    
    r = np.zeros(ne)
    
#    for i in range(2,ne+2):
#        r[i-2] = v[i-1]*(v[i+1] - v[i-2]) - v[i] + fr
#    ys = np.sum(y,axis=1)
    r = -v[1:ne+1]*(v[0:ne] - v[3:ne+3]) - v[2:ne+2] + fr - (h*c/b)*ys
    
    return r*dt
    
def rk4uc(ne,J,u,y,fr,h,c,b,dt):
    k1x = rhs_dx_dt(ne,u,y,fr,h,c,dt)
    k2x = rhs_dx_dt(ne,u+0.5*k1x,y,fr,h,c,dt)
    k3x = rhs_dx_dt(ne,u+0.5*k2x,y,fr,h,c,dt)
    k4x = rhs_dx_dt(ne,u+0.5*k3x,y,fr,h,c,dt)
    
    un = u + (k1x + 2.0*(k2x + k3x) + k4x)/6.0
    
    return un
       
#%% Main program:
ne = 36
J = 10
fr = 10.0
c = 10.0
b = 10.0
h = 1.0

fact = 0.1
std = 1.0

dt = 0.001
tmax = 20.0
tinit = 5.0
ns = int(tinit/dt)
nt = int(tmax/dt)

ttrain = 10.0
ntrain = int(ttrain/dt)

nf = 1         # frequency of observation
nb = int(ntrain/nf) # number of observation time
oib = [nf*k for k in range(nb+1)]

u = np.zeros(ne)
utrue = np.zeros((ne,nt+1))
uinit = np.zeros((ne,ns+1))
ysuminit = np.zeros((ne,ns+1))
ysum = np.zeros((ne,nt+1))

ti = np.linspace(-tinit,0,ns+1)
t = np.linspace(0,tmax,nt+1)
tobs = np.linspace(0,tmax,nb+1)
x = np.linspace(1,ne,ne)

X,T = np.meshgrid(x,t,indexing='ij')
Xi,Ti = np.meshgrid(x,ti,indexing='ij')

ttrain = 10.0
ntrain = int(ttrain/dt)

#%%    
data = np.load('data_cyclic.npz')
utrue = data['utrue']
ysum = data['ysum']
#uobsfull = data['uobs']

mean = 0.0
sd2 = 1.0e0 # added noise (variance)
sd1 = np.sqrt(sd2) # added noise (standard deviation)

uobsfull = utrue[:,:] + np.random.normal(mean,sd1,[ne,nt+1])

features_train = utrue[:,oib].T
labels_train = ysum[:,oib].T

sc_input = MinMaxScaler(feature_range=(-1,1))
sc_input = sc_input.fit(features_train)
features_train_sc = sc_input.transform(features_train)

sc_output = MinMaxScaler(feature_range=(-1,1))
sc_output = sc_output.fit(labels_train)
labels_train_sc = sc_output.transform(labels_train)

#%%
#-----------------------------------------------------------------------------#
# generate erroneous soltions trajectory
#-----------------------------------------------------------------------------#
saved_model = load_model('cnn_best_model.hd5', 
                         custom_objects={'coeff_determination': coeff_determination})

tstart = 10.0
tend = 20.0
nts = int(tstart/dt) 
ntest = int((tend-tstart)/dt)

#%%
nt = ntest
nf = 10         # frequency of observation
nb = int(nt/nf) # number of observation time
oib = [nf*k for k in range(nb+1)]

uobs = uobsfull[:,int(nts):][:,oib]

uw = np.zeros((ne,nt+1))
k = 0
mean = 0.0

si2 = 1.0e-2
si1 = np.sqrt(si2)

u = utrue[:,k+nts] #+ np.random.normal(mean,si1,ne)
uw[:,k] = u
ys = ysum[:,k]

for k in range(1,nt+1):
    u[:] = uw[:,k-1]
    un = rk4uc(ne,J,u,ys,fr,h,c,b,dt)
    uw[:,k] = un
    un_sc = sc_input.transform(np.reshape(un,[1,-1]))
    xtest_sc = np.reshape(un_sc,[1,-1,1])
    ypred_sc = saved_model.predict(xtest_sc) 
    ys = sc_output.inverse_transform(np.reshape(ypred_sc,[1,-1])).flatten()
#    u = np.copy(un)

t = np.linspace(tstart,tend,nt+1)
x = np.linspace(1,ne,ne)

X,T = np.meshgrid(x,t,indexing='ij')

vmin = -12
vmax = 12
fig, ax = plt.subplots(3,1,figsize=(6,7.5))
cs = ax[0].contourf(T,X,utrue[:,nts:],40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[0],ticks=np.linspace(vmin, vmax, 6))

cs = ax[1].contourf(T,X,uw,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[1],ticks=np.linspace(vmin, vmax, 6))

diff = utrue[:,nts:] - uw
cs = ax[2].contourf(T,X,diff,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[2],ticks=np.linspace(vmin, vmax, 6))

print(np.linalg.norm(diff))

fig.tight_layout()
plt.show()
fig.savefig('cnn_'+str(tstart)+'_'+str(tend)+'.png',dpi=300)

np.savez('data_'+str(0)+'.npz',T=T,X=X,utrue=utrue,uw=uw)

#%%
#-----------------------------------------------------------------------------#    
#-----------------------------------------------------------------------------#
# EnKF model
# number of observation vector
me = 18
freq = int(ne/me)
oin = [freq*i-1 for i in range(1,me+1)]
roin = np.int32(np.linspace(0,me-1,me))
print(oin)

dh = np.zeros((me,ne))
dh[roin,oin] = 1.0

H = np.zeros((me,ne))
H[roin,oin] = 1.0

#%%
# number of ensemble 
npe = 30
cn = 1.0/np.sqrt(npe-1)
lambd = 1.00

z = np.zeros((me,nb+1))
#zf = np.zeros((me,npe,nb+1))
DhX = np.zeros((me,npe))
DhXm = np.zeros(me)

ua = np.zeros((ne,nt+1)) # mean analyssi solution (to store)
uf = np.zeros(ne)        # mean forecast
sc = np.zeros((ne,npe))   # square-root of the covariance matrix
Af = np.zeros((ne,npe))   # Af data
ue = np.zeros((ne,npe,nt+1)) # all ensambles
yse = np.zeros((ne,npe,nt+1)) # all ensambles
ph = np.zeros((ne,me))

km = np.zeros((ne,me))
kmd = np.zeros((ne,npe))

cc = np.zeros((me,me))
ci = np.zeros((me,me))

for k in range(nb+1):
    z[:,k] = uobs[oin,k]
#    for n in range(npe):
#        zf[:,n,k] = z[:,k] + np.random.normal(mean,sd1,me)

#%%
# initial ensemble
k = 0
se2 = 0.0 #np.sqrt(sd2)
se1 = np.sqrt(se2)

for n in range(npe):
    ue[:,n,k] = uw[:,k] + np.random.normal(mean,si1,ne)       
    yse[:,n,k] = ysum[:,k]
    
ua[:,k] = np.sum(ue[:,:,k],axis=1)
ua[:,k] = ua[:,k]/npe

kobs = 1

# RK4 scheme
for k in range(1,nt+1):
    
    # forecast afor all ensemble fields
    for n in range(npe):
        u[:] = ue[:,n,k-1]
        ys[:] = yse[:,n,k-1]
        un = rk4uc(ne,J,u,ys,fr,h,c,b,dt)
        ue[:,n,k] = un[:] #+ np.random.normal(mean,se1,ne)
    
        un_sc = sc_input.transform(np.reshape(un,[1,-1]))
        xtest_sc = np.reshape(un_sc,[1,-1,1])
        ypred_sc = saved_model.predict(xtest_sc) 
        yse[:,n,k] = sc_output.inverse_transform(np.reshape(ypred_sc,[1,-1])).flatten()
      
    # mean analysis for plotting
    ua[:,k] = np.sum(ue[:,:,k],axis=1)
    ua[:,k] = ua[:,k]/npe

    if k % 1000 == 0:
        print(k)
        
    if k == oib[kobs]:

        # compute mean of the forecast fields
        uf[:] = np.sum(ue[:,:,k],axis=1)   
        uf[:] = uf[:]/npe
        
        # compute Af dat
        for n in range(npe):
            Af[:,n] = ue[:,n,k] - uf[:]
        
        da = dh @ Af
        
        cc = da @ da.T/(npe-1)  
        
        for i in range(me):
            cc[i,i] = cc[i,i] + sd2 
        
        ci = np.linalg.pinv(cc)
        
        km = Af @ da.T @ ci/(npe-1)
        
#        pf = Af @ Af.T
#        pf[:,:] = pf[:,:]/(npe-1)
#        
#        dp = dh @ pf
#        cc = dp @ dh.T     
#
#        for i in range(me):
#            cc[i,i] = cc[i,i] + sd2     
#        
#        ph = pf @ dh.T
#        
#        ci = np.linalg.pinv(cc) # ci: inverse of cc matrix
#        
#        km = ph @ ci # compute Kalman gain
        
        # analysis update    
        kmd = km @ (z[:,kobs] - uf[oin])
        ua[:,k] = uf[:] + kmd[:]
        
        # ensemble correction
        ha = dh @ Af
        
        ue[:,:,k] = Af[:,:] - 0.5*(km @ dh @ Af) + ua[:,k].reshape(-1,1)
        
        #multiplicative inflation (optional): set lambda=1.0 for no inflation
        ue[:,:,k] = ua[:,k].reshape(-1,1) + lambd*(ue[:,:,k] - ua[:,k].reshape(-1,1))
        
        kobs = kobs+1

np.savez('data_'+str(me)+'_'+str(lambd)+'.npz',t=t,tobs=tobs,T=T,X=X,utrue=utrue,uobs=uobs,uw=uw,ua=ua,oin=oin)
    
#%%
fig, ax = plt.subplots(3,1,sharex=True,figsize=(6,5))

n = [9,14,34]
for i in range(3):
    ax[i].plot(t,utrue[n[i],nts:],'k-')
    ax[i].plot(t,uw[n[i],:],'b--')
    ax[i].plot(t,ua[n[i],:],'g-.')
#    if i == 0:
#        ax[i].plot(tobs,uobs[n[i],:],'ro',fillstyle='none', markersize=6,markeredgewidth=2)

    ax[i].set_xlim([np.min(t),np.max(t)])
    ax[i].set_ylabel(r'$u_{'+str(n[i]+1)+'}$')

ax[i].set_xlabel(r'$t$')
line_labels = ['Observation','True','Wrong','EnKF']
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=-0.2, ncol=4, labelspacing=0.)
fig.tight_layout()
plt.show() 
fig.savefig('m_'+str(me)+'_'+str(lambd)+'.png')

#%%
vmin = -12
vmax = 12
fig, ax = plt.subplots(3,1,figsize=(6,7.5))

cs = ax[0].contourf(T,X,utrue[:,nts:],40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[0],ticks=np.linspace(vmin, vmax, 6))
ax[0].set_title('True')

cs = ax[1].contourf(T,X,ua,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(ua)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[1],ticks=np.linspace(vmin, vmax, 6))
ax[1].set_title('EnKF')

diff = ua - utrue[:,nts:]
cs = ax[2].contourf(T,X,diff,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(ua)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[2],ticks=np.linspace(vmin, vmax, 6))
ax[2].set_title('Difference')

fig.tight_layout()
plt.show() 
fig.savefig('f_'+str(me)+'_'+str(lambd)+ '.png',dpi=300)   

print(np.linalg.norm(diff))

l2norm = np.array([me,nf,sd2,lambd,np.linalg.norm(diff)])
f=open('l2norm.dat','ab')
np.savetxt(f,l2norm.reshape([1,-1]))    
f.close()
 
























































