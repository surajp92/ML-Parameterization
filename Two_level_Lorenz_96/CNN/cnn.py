#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:34:23 2020

@author: suraj
"""
import tensorflow as tf
#tf.compat.v1.enable_eager_execution()

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.integrate import simps
import pyfftw

#from keras.utils import plot_model
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
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



from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
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

def grad1(matrix): 
    dx = 1.0
    u_x = np.gradient(matrix,dx,axis=0)
    u_xx = np.gradient(u_x,dx,axis=0)
    return u_xx

def artificial_diffusion(y_true, y_pred):
    u_xx = tf.py_function(grad1,[y_pred],tf.float32)
    
#    xs = tf.ones_like(y_pred)
#    u_x = tf.gradients(y_pred, xs)
#    u_xx = tf.gradients(u_x, xs)
    
    nu = 0.1

    ad_loss = -nu * u_xx    
    
    return kb.mean(ad_loss)
    
#%% main parameters:
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
nt = int(tmax/dt)

t = np.linspace(0,tmax,nt+1)
x = np.linspace(1,ne,ne)
X,T = np.meshgrid(x,t,indexing='ij')

ttrain = 10.0
ntrain = int(ttrain/dt)

nf = 1         # frequency of training samples
nb = int(ntrain/nf) # number of training time
oib = [nf*k for k in range(nb+1)]

training = True

#%%
data = np.load('data_cyclic.npz')
utrue = data['utrue']
ysum = data['ysum']

features_train = utrue[:,oib].T
labels_train = ysum[:,oib].T

#%%
sc_input = MinMaxScaler(feature_range=(-1,1))
sc_input = sc_input.fit(features_train)
features_train_sc = sc_input.transform(features_train)

sc_output = MinMaxScaler(feature_range=(-1,1))
sc_output = sc_output.fit(labels_train)
labels_train_sc = sc_output.transform(labels_train)

xtrain_sc = np.reshape(features_train_sc,[-1,ne,1]) 
ytrain_sc = np.reshape(labels_train_sc,[-1,ne,1])

x_train, x_valid, y_train, y_valid = train_test_split(xtrain_sc, ytrain_sc, test_size=0.2 , 
                                                      shuffle= True)

#%%
n_states, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[2]

lr = 0.001
epochs = 400
batch_size = 128

if training == True:
    model = Sequential()
    input_img = Input(shape=(n_states,n_features))
    
    x = Conv1D(128, kernel_size=7, activation='relu', padding='same')(input_img)
#    x = Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
#    x = Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    
    decoded = Conv1D(n_outputs, kernel_size=3, activation='linear', padding='same')(x)
    
    model = Model(input_img, decoded)
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=[coeff_determination])
    
    model.summary()
    
    filepath = "cnn_best_model.hd5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
    #                              save_best_only=True, mode='min')
    # callbacks_list = [checkpoint]
    
    # fit network
    history_callback = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
                                 validation_data=(x_valid, y_valid))
    
    # plot history
    loss = history_callback.history["loss"]
    val_loss = history_callback.history["val_loss"]
    mse = history_callback.history['coeff_determination']
    val_mse = history_callback.history['val_coeff_determination']
    
    plt.figure()
    epochs = range(1, len(loss) + 1)
    plt.semilogy(epochs, loss, 'b', label='Training loss')
    plt.semilogy(epochs, val_loss, 'r', label='Validationloss')
    plt.title('Training and validation loss')
    plt.legend()
    filename = 'loss.png'
    plt.savefig(filename, dpi = 300)
    plt.show()
    
    model.save(filepath)

#%%
features_test = utrue[:,:].T
features_test_sc = sc_input.transform(features_test)
xtest_sc = np.reshape(features_test_sc,[-1,ne,1]) 

saved_model = load_model('cnn_best_model.hd5', 
                         custom_objects={'coeff_determination': coeff_determination})
ypred_sc = saved_model.predict(xtest_sc)


#%%
ypred_sc = np.reshape(ypred_sc,[-1,ne])

ypred = sc_output.inverse_transform(ypred_sc)
ypred = ypred.T #np.reshape(ypred,[ne,-1])

fig, ax = plt.subplots(4,1,sharex=True,figsize=(10,6))

n = [0,7,15,31]
for i in range(4):
#    ax[i].plot(t,utrue[n[i],:],'k-')
    ax[i].plot(t,ysum[n[i],:],'k-')
    ax[i].plot(t,ypred[n[i],:],'r--')
    ax[i].set_xlim([0,tmax])
#    ax[i].set_ylabel(r'$x_{'+str(n[i]+1)+'}$')
    ax[i].set_ylabel(r'$y_{'+str(n[i]+1)+'}$')
    
ax[i].set_xlabel(r'$t$')
line_labels = ['True','ML']
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=-0.2, ncol=4, labelspacing=0.)
fig.tight_layout()
plt.show() 
fig.savefig('ml.png',dpi=300)

vmin = -3
vmax = 3
fig, ax = plt.subplots(3,1,figsize=(6,7.5))
cs = ax[0].contourf(T,X,ysum,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[0],ticks=np.linspace(vmin, vmax, 6))

cs = ax[1].contourf(T,X,ypred,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[1],ticks=np.linspace(vmin, vmax, 6))

diff = ysum - ypred
cs = ax[2].contourf(T,X,diff,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[2],ticks=np.linspace(vmin, vmax, 6))

fig.tight_layout()
plt.show()

#%%      
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

tstart = 10.0
tend = 20.0
nts = int(tstart/dt) 
ntest = int((tend-tstart)/dt)

uml = np.zeros((ne,ntest+1))
ysml = np.zeros((ne,ntest+1))

uml[:,0] = utrue[:,nts]
ysml[:,0] = ysum[:,nts]


# generate true forward solution
for k in range(1,ntest+1):
    u = uml[:,k-1]
    ys = ysml[:,k-1]
    un = rk4uc(ne,J,u,ys,fr,h,c,b,dt)
    uml[:,k] = un
    un_sc = sc_input.transform(np.reshape(un,[1,-1]))
    xtest_sc = np.reshape(un_sc,[1,-1,1])
    ypred_sc = saved_model.predict(xtest_sc) 
    ysml[:,k] = sc_output.inverse_transform(np.reshape(ypred_sc,[1,-1])).flatten()

#%%
t = np.linspace(tstart,tend,ntest+1)
x = np.linspace(1,ne,ne)
X,T = np.meshgrid(x,t,indexing='ij')

vmin = -12
vmax = 12
fig, ax = plt.subplots(3,1,figsize=(6,7.5))
cs = ax[0].contourf(T,X,utrue[:,nts:nts+ntest+1],40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[0],ticks=np.linspace(vmin, vmax, 6))

cs = ax[1].contourf(T,X,uml,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[1],ticks=np.linspace(vmin, vmax, 6))

diff = utrue[:,nts:nts+ntest+1] - uml
cs = ax[2].contourf(T,X,diff,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[2],ticks=np.linspace(vmin, vmax, 6))

print(np.linalg.norm(diff))

fig.tight_layout()
plt.show()
fig.savefig('cnn_'+str(tstart)+'_'+str(tend)+'.png',dpi=300)

np.savez('data_'+str(3)+'.npz',T=T,X=X,utrue=utrue,uml=uml)

#%%
fig, ax = plt.subplots(4,1,sharex=True,figsize=(8,6))
n = [0,7,15,31]
for i in range(4):
#    ax[i].plot(t,utrue[n[i],:],'k-')
    ax[i].plot(t[:10001],utrue[n[i],nts:],'k-')
    ax[i].plot(t[:10001],uml[n[i],:],'r--')
    ax[i].set_xlim([np.min(t[:]),np.max(t[:])])
#    ax[i].set_ylabel(r'$x_{'+str(n[i]+1)+'}$')
    ax[i].set_ylabel(r'$x_{'+str(n[i]+1)+'}$')
    
ax[i].set_xlabel(r'$t$')
line_labels = ['True','ML']
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=-0.2, ncol=4, labelspacing=0.)
fig.tight_layout()
plt.show() 
fig.savefig('modes.png',dpi=300)

#%%
vmin = -3
vmax = 3
fig, ax = plt.subplots(3,1,figsize=(6,7.5))
cs = ax[0].contourf(T,X,ysum,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[0],ticks=np.linspace(vmin, vmax, 6))

cs = ax[1].contourf(T,X,ysml,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[1],ticks=np.linspace(vmin, vmax, 6))

diff = ysum - ysml
cs = ax[2].contourf(T,X,diff,40,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[2],ticks=np.linspace(vmin, vmax, 6))

fig.tight_layout()
plt.show()
#fig.savefig('field.png',dpi=300)












