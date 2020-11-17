#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 14:16:04 2020

@author: suraj
"""

import numpy as np
from numpy.random import seed
seed(1)
import pyfftw
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt 
import time as tm
import matplotlib.ticker as ticker
import os
from numba import jit
import random
from scipy import ndimage

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K

from scipy.ndimage import gaussian_filter

#from utils import *

#%%
file = open("output_version.txt", "a", buffering=1)
import sys
print(sys.version, file=file)
import keras as ke
print(ke.__version__, file=file)
import tensorflow as tf
print(tf.__version__, file=file)
file.close()

#%%
# fast poisson solver using second-order central difference scheme
def fps(nx, ny, dx, dy, f):
    epsilon = 1.0e-6
    aa = -2.0/(dx*dx) - 2.0/(dy*dy)
    bb = 2.0/(dx*dx)
    cc = 2.0/(dy*dy)
    hx = 2.0*np.pi/np.float64(nx)
    hy = 2.0*np.pi/np.float64(ny)
    
    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[:] = hx*np.float64(np.arange(0, nx))

    ky[:] = hy*np.float64(np.arange(0, ny))
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(np.cos(kx), np.cos(ky), indexing='ij')
    
    data = np.empty((nx,ny), dtype='complex128')
    data1 = np.empty((nx,ny), dtype='complex128')
    
    data[:,:] = np.vectorize(complex)(f[2:nx+2,2:ny+2],0.0)

    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
    
    e = fft_object(data)
    #e = pyfftw.interfaces.scipy_fftpack.fft2(data)
    
    e[0,0] = 0.0
    
    data1[:,:] = e[:,:]/(aa + bb*kx[:,:] + cc*ky[:,:])

    ut = np.real(fft_object_inv(data1))
    
    #periodicity
    u = np.empty((nx+5,ny+5)) 
    u[2:nx+2,2:ny+2] = ut
    u[:,ny+2] = u[:,2]
    u[nx+2,:] = u[2,:]
    u[nx+2,ny+2] = u[2,2]
    
    return u

#%%
# set periodic boundary condition for ghost nodes. Index 0 and (n+2) are the ghost boundary locations
def bc(nx,ny,u):
    u[:,0] = u[:,ny]
    u[:,1] = u[:,ny+1]
    u[:,ny+3] = u[:,3]
    u[:,ny+4] = u[:,4]
    
    u[0,:] = u[nx,:]
    u[1,:] = u[nx+1,:]
    u[nx+3,:] = u[3,:]
    u[nx+4,:] = u[4,:]
    
    return u

#%%
def grad_spectral(nx,ny,u):
    
    '''
    compute the gradient of u using spectral differentiation
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    u : solution field 
    
    Output
    ------
    ux : du/dx (size = [nx+1,ny+1])
    uy : du/dy (size = [nx+1,ny+1])
    '''
    
    ux = np.empty((nx+1,ny+1))
    uy = np.empty((nx+1,ny+1))
    
    uf = np.fft.fft2(u[0:nx,0:ny])

    kx = np.fft.fftfreq(nx,1/nx)
    ky = np.fft.fftfreq(ny,1/ny)
    
    kx = kx.reshape(nx,1)
    ky = ky.reshape(1,ny)
    
    uxf = 1.0j*kx*uf
    uyf = 1.0j*ky*uf 
    
    ux[0:nx,0:ny] = np.real(np.fft.ifft2(uxf))
    uy[0:nx,0:ny] = np.real(np.fft.ifft2(uyf))
    
    # periodic bc
    ux[:,ny] = ux[:,0]
    ux[nx,:] = ux[0,:]
    ux[nx,ny] = ux[0,0]
    
    # periodic bc
    uy[:,ny] = uy[:,0]
    uy[nx,:] = uy[0,:]
    uy[nx,ny] = uy[0,0]
    
    return ux,uy

#%%
@jit
def les_filter(nx,ny,nxc,nyc,u):
    
    '''
    coarsen the solution field keeping the size of the data same
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    nxc,nyc : number of grid points in x and y direction on coarse grid
    u : solution field on fine grid
    
    Output
    ------
    uc : coarsened solution field [nx+1, ny+1]
    '''
    
    uf = np.fft.fft2(u[0:nx,0:ny])
        
    uf[int(nxc/2):int(nx-nxc/2),:] = 0.0
    uf[:,int(nyc/2):int(ny-nyc/2)] = 0.0 
    utc = np.real(np.fft.ifft2(uf))
    
    uc = np.zeros((nx+1,ny+1))
    uc[0:nx,0:ny] = utc
    
    # periodic bc
    uc[:,ny] = uc[:,0]
    uc[nx,:] = uc[0,:]
    uc[nx,ny] = uc[0,0]
    
    return uc

#%%  
def dyn_smag(nx,ny,kappa,sc,wc):
    '''
    compute the eddy viscosity using Germanos dynamics procedure with Lilys 
    least square approximation
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    kapppa : sub-filter grid filter ratio
    wc : vorticity on LES grid
    sc : streamfunction on LES grid
    
    Output
    ------
    ev : (cs*delta)**2*|S| (size = [nx+1,ny+1])
    '''
    
    nxc = int(nx/kappa) 
    nyc = int(ny/kappa)
    
    scc = les_filter(nx,ny,nxc,nyc,sc[2:nx+3,2:ny+3])
    wcc = les_filter(nx,ny,nxc,nyc,wc[2:nx+3,2:ny+3])
    
    scx,scy = grad_spectral(nx,ny,sc[2:nx+3,2:ny+3])
    wcx,wcy = grad_spectral(nx,ny,wc[2:nx+3,2:ny+3])
    
    wcxx,wcxy = grad_spectral(nx,ny,wcx)
    wcyx,wcyy = grad_spectral(nx,ny,wcy)
    
    scxx,scxy = grad_spectral(nx,ny,scx)
    scyx,scyy = grad_spectral(nx,ny,scy)
    
    dac = np.sqrt(4.0*scxy**2 + (scxx - scyy)**2) # |\bar(s)|
    dacc = les_filter(nx,ny,nxc,nyc,dac)        # |\tilde{\bar{s}}| = \tilde{|\bar(s)|}
    
    sccx,sccy = grad_spectral(nx,ny,scc)
    wccx,wccy = grad_spectral(nx,ny,wcc)
    
    wccxx,wccxy = grad_spectral(nx,ny,wccx)
    wccyx,wccyy = grad_spectral(nx,ny,wccy)
    
    scy_wcx = scy*wcx
    scx_wcy = scx*wcy
    
    scy_wcx_c = les_filter(nx,ny,nxc,nyc,scy_wcx)
    scx_wcy_c = les_filter(nx,ny,nxc,nyc,scx_wcy)
    
    h = (sccy*wccx - sccx*wccy) - (scy_wcx_c - scx_wcy_c)
    
    t = dac*(wcxx + wcyy)
    tc = les_filter(nx,ny,nxc,nyc,t)
    
    m = kappa**2*dacc*(wccxx + wccyy) - tc
    
    hm = h*m
    mm = m*m
    
    CS2 = (np.sum(0.5*(hm + abs(hm)))/np.sum(mm))
    
    ev = CS2*dac
    
    return ev

#%%
def stat_smag(nx,ny,dx,dy,s,cs):
        
        
    dsdxy = (1.0/(4.0*dx*dy))*(s[1:nx+2,1:ny+2] + s[3:nx+4,3:ny+4] \
                                             -s[3:nx+4,1:ny+2] - s[1:nx+2,3:ny+4])
    
    dsdxx = (1.0/(dx*dx))*(s[3:nx+4,2:ny+3] - 2.0*s[2:nx+3,2:ny+3] \
                                         +s[1:nx+2,2:ny+3])
    
    dsdyy = (1.0/(dy*dy))*(s[2:nx+3,3:ny+4] - 2.0*s[2:nx+3,2:ny+3] \
                                         +s[2:nx+3,1:ny+2])
    
    ev = cs*cs*dx*dy*np.sqrt(4.0*dsdxy*dsdxy + (dsdxx-dsdyy)*(dsdxx-dsdyy))
    
    return ev    

#%%
def cnn_closure(nx,ny,w,s,max_min,model,ifeat):
    wx,wy = grad_spectral(nx,ny,w[2:nx+3,2:ny+3])
    wxx,wxy = grad_spectral(nx,ny,wx)
    wyx,wyy = grad_spectral(nx,ny,wy)
    
    sx,sy = grad_spectral(nx,ny,s[2:nx+3,2:ny+3])
    sxx,sxy = grad_spectral(nx,ny,sx)
    syx,syy = grad_spectral(nx,ny,sy)
    
    kernel_w = np.sqrt(wx**2 + wy**2)
    kernel_s = np.sqrt(4.0*sxy**2 + (sxx - syy)**2)
    
    wc = (2.0*w[2:nx+3,2:ny+3] - (max_min[0,0] + max_min[0,1]))/(max_min[0,0] - max_min[0,1])
    sc = (2.0*s[2:nx+3,2:ny+3] - (max_min[1,0] + max_min[1,1]))/(max_min[1,0] - max_min[1,1])
    kwc = (2.0*kernel_w - (max_min[2,0] + max_min[2,1]))/(max_min[2,0] - max_min[2,1])
    ksc = (2.0*kernel_s - (max_min[3,0] + max_min[3,1]))/(max_min[3,0] - max_min[3,1])
    
    wcx = (2.0*wx - (max_min[4,0] + max_min[4,1]))/(max_min[4,0] - max_min[4,1])
    wcy = (2.0*wy - (max_min[5,0] + max_min[5,1]))/(max_min[5,0] - max_min[5,1])
    wcxx = (2.0*wxx - (max_min[6,0] + max_min[6,1]))/(max_min[6,0] - max_min[6,1])
    wcyy = (2.0*wyy - (max_min[7,0] + max_min[7,1]))/(max_min[7,0] - max_min[7,1])
    wcxy = (2.0*wxy - (max_min[8,0] + max_min[8,1]))/(max_min[8,0] - max_min[8,1])
    
    scx = (2.0*sx - (max_min[9,0] + max_min[9,1]))/(max_min[9,0] - max_min[9,1])
    scy = (2.0*sy - (max_min[10,0] + max_min[10,1]))/(max_min[10,0] - max_min[10,1])
    scxx = (2.0*sxx - (max_min[11,0] + max_min[11,1]))/(max_min[11,0] - max_min[11,1])
    scyy = (2.0*syy - (max_min[12,0] + max_min[12,1]))/(max_min[12,0] - max_min[12,1])
    scxy = (2.0*sxy - (max_min[13,0] + max_min[13,1]))/(max_min[13,0] - max_min[13,1])

    if ifeat == 1:
        x_test = np.zeros((1,nx+1,ny+1,2))
        x_test[0,:,:,0] = wc
        x_test[0,:,:,1] = sc
    if ifeat == 2:
        x_test = np.zeros((1,nx+1,ny+1,4))
        x_test[0,:,:,0] = wc
        x_test[0,:,:,1] = sc
        x_test[0,:,:,2] = kwc
        x_test[0,:,:,3] = ksc
    if ifeat == 3:
        x_test = np.zeros((1,nx+1,ny+1,12))
        x_test[0,:,:,0] = wc
        x_test[0,:,:,1] = sc
        x_test[0,:,:,2] = wcx
        x_test[0,:,:,3] = wcy
        x_test[0,:,:,4] = wcxx
        x_test[0,:,:,5] = wcyy
        x_test[0,:,:,6] = wcxy
        x_test[0,:,:,7] = scx
        x_test[0,:,:,8] = scy
        x_test[0,:,:,9] = scxx
        x_test[0,:,:,10] = scyy
        x_test[0,:,:,11] = scxy
    
    y_pred_sc = model.predict(x_test[:,:,:,:])
    y_pred = 0.5*(y_pred_sc[0,:,:,0]*(max_min[14,0] - max_min[14,1]) + (max_min[14,0] + max_min[14,1]))
    
    return y_pred

#%% 
# compute rhs using arakawa scheme
# computed at all physical domain points (1:nx+1,1:ny+1; all boundary points included)
# no ghost points
def rhs_arakawa(nx,ny,dx,dy,re,w,s,ifm,kappa,max_min,model,ifeat):
    aa = 1.0/(dx*dx)
    bb = 1.0/(dy*dy)
    gg = 1.0/(4.0*dx*dy)
    hh = 1.0/3.0
    
    f = np.zeros((nx+5,ny+5))
    
    #Arakawa    
    j1 = gg*( (w[3:nx+4,2:ny+3]-w[1:nx+2,2:ny+3])*(s[2:nx+3,3:ny+4]-s[2:nx+3,1:ny+2]) \
             -(w[2:nx+3,3:ny+4]-w[2:nx+3,1:ny+2])*(s[3:nx+4,2:ny+3]-s[1:nx+2,2:ny+3]))

    j2 = gg*( w[3:nx+4,2:ny+3]*(s[3:nx+4,3:ny+4]-s[3:nx+4,1:ny+2]) \
            - w[1:nx+2,2:ny+3]*(s[1:nx+2,3:ny+4]-s[1:nx+2,1:ny+2]) \
            - w[2:nx+3,3:ny+4]*(s[3:nx+4,3:ny+4]-s[1:nx+2,3:ny+4]) \
            + w[2:nx+3,1:ny+2]*(s[3:nx+4,1:ny+2]-s[1:nx+2,1:ny+2]))
    
    j3 = gg*( w[3:nx+4,3:ny+4]*(s[2:nx+3,3:ny+4]-s[3:nx+4,2:ny+3]) \
            - w[1:nx+2,1:ny+2]*(s[1:nx+2,2:ny+3]-s[2:nx+3,1:ny+2]) \
            - w[1:nx+2,3:ny+4]*(s[2:nx+3,3:ny+4]-s[1:nx+2,2:ny+3]) \
            + w[3:nx+4,1:ny+2]*(s[3:nx+4,2:ny+3]-s[2:nx+3,1:ny+2]) )

    jac = (j1+j2+j3)*hh
    
    lap = aa*(w[3:nx+4,2:ny+3]-2.0*w[2:nx+3,2:ny+3]+w[1:nx+2,2:ny+3]) \
        + bb*(w[2:nx+3,3:ny+4]-2.0*w[2:nx+3,2:ny+3]+w[2:nx+3,1:ny+2])
    
    if ifm == 0:
        f[2:nx+3,2:ny+3] = -jac + lap/re 
        f[:,ny+2] = f[:,2]
        f[nx+2,:] = f[2,:]
        f[nx+2,ny+2] = f[2,2] 
        
    elif ifm == 1:
        ev = dyn_smag(nx,ny,kappa,s,w)
        f[2:nx+3,2:ny+3] = -jac + lap/re + ev*lap
    
    elif ifm == 2:
        kconvolve = np.array([[1,1,1],[1,1,1],[1,1,1]])
        
        pi_source = cnn_closure(nx,ny,w,s,max_min,model,ifeat)
        nue = pi_source/lap
        
        nue_p = np.where(nue > 0, nue, 0.0)
        
        nue_loc_avg = ndimage.convolve(nue_p, kconvolve, mode='mirror')#, cval=0.0)
        nue_loc_avg = nue_loc_avg/9.0
        
        nue_loc_avg_use = np.where(nue_p < nue_loc_avg, nue_p, 0.0)
        
        pi_source = np.where(nue_loc_avg_use > 0.0, pi_source[:,:],0.0)
        
        f[2:nx+3,2:ny+3] = -jac + lap/re + pi_source
                        
    return f

#%%
# set initial condition for decay of turbulence problem
def ic_decay(nx,ny,dx,dy):
    #w = np.empty((nx+3,ny+3))
    
    epsilon = 1.0e-6
    
    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[0:int(nx/2)] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(0,int(nx/2)))
    kx[int(nx/2):nx] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(-int(nx/2),0))

    ky[0:ny] = kx[0:ny]
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    
    ksi = 2.0*np.pi*np.random.random_sample((int(nx/2+1), int(ny/2+1)))
    eta = 2.0*np.pi*np.random.random_sample((int(nx/2+1), int(ny/2+1)))
    
    phase = np.zeros((nx,ny), dtype='complex128')
    wf =  np.empty((nx,ny), dtype='complex128')
    
    phase[1:int(nx/2),1:int(ny/2)] = np.vectorize(complex)(np.cos(ksi[1:int(nx/2),1:int(ny/2)] +
                                    eta[1:int(nx/2),1:int(ny/2)]), np.sin(ksi[1:int(nx/2),1:int(ny/2)] +
                                    eta[1:int(nx/2),1:int(ny/2)]))

    phase[nx-1:int(nx/2):-1,1:int(ny/2)] = np.vectorize(complex)(np.cos(-ksi[1:int(nx/2),1:int(ny/2)] +
                                            eta[1:int(nx/2),1:int(ny/2)]), np.sin(-ksi[1:int(nx/2),1:int(ny/2)] +
                                            eta[1:int(nx/2),1:int(ny/2)]))

    phase[1:int(nx/2),ny-1:int(ny/2):-1] = np.vectorize(complex)(np.cos(ksi[1:int(nx/2),1:int(ny/2)] -
                                           eta[1:int(nx/2),1:int(ny/2)]), np.sin(ksi[1:int(nx/2),1:int(ny/2)] -
                                           eta[1:int(nx/2),1:int(ny/2)]))

    phase[nx-1:int(nx/2):-1,ny-1:int(ny/2):-1] = np.vectorize(complex)(np.cos(-ksi[1:int(nx/2),1:int(ny/2)] -
                                                 eta[1:int(nx/2),1:int(ny/2)]), np.sin(-ksi[1:int(nx/2),1:int(ny/2)] -
                                                eta[1:int(nx/2),1:int(ny/2)]))

    k0 = 10.0
    c = 4.0/(3.0*np.sqrt(np.pi)*(k0**5))           
    
    kk = np.sqrt(kx[:,:]**2 + ky[:,:]**2)
    es = c*(kk**4)*np.exp(-(kk/k0)**2)
    wf[:,:] = np.sqrt((kk*es/np.pi)) * phase[:,:]*(nx*ny)
            
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
    ut = np.real(fft_object_inv(wf)) 
    
    #w = np.zeros((nx+3,ny+3))
    
    #periodicity
    w = np.zeros((nx+5,ny+5)) 
    w[2:nx+2,2:ny+2] = ut
    w[:,ny+2] = w[:,2]
    w[nx+2,:] = w[2,:]
    w[nx+2,ny+2] = w[2,2] 
    
    w = bc(nx,ny,w)    
    
    return w

#%%
# compute the energy spectrum numerically
def energy_spectrum(nx,ny,w):
    epsilon = 1.0e-6

    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[0:int(nx/2)] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(0,int(nx/2)))
    kx[int(nx/2):nx] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(-int(nx/2),0))

    ky[0:ny] = kx[0:ny]
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')

    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    wf = fft_object(w[2:nx+2,2:ny+2]) 
    
    es =  np.empty((nx,ny))
    
    kk = np.sqrt(kx[:,:]**2 + ky[:,:]**2)
    es[:,:] = np.pi*((np.abs(wf[:,:])/(nx*ny))**2)/kk
    
    n = int(np.sqrt(nx*nx + ny*ny)/2.0)-1
    
    en = np.zeros(n+1)
    
    for k in range(1,n+1):
        en[k] = 0.0
        ic = 0
        ii,jj = np.where((kk[1:,1:]>(k-0.5)) & (kk[1:,1:]<(k+0.5)))
        ic = ii.size
        ii = ii+1
        jj = jj+1
        en[k] = np.sum(es[ii,jj])
#        for i in range(1,nx):
#            for j in range(1,ny):          
#                kk1 = np.sqrt(kx[i,j]**2 + ky[i,j]**2)
#                if ( kk1>(k-0.5) and kk1<(k+0.5) ):
#                    ic = ic+1
#                    en[k] = en[k] + es[i,j]
                    
        en[k] = en[k]/ic
        
    return en, n

#%%
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
        
    
#%% 
# read input file
l1 = []
with open('input.txt') as f:
    for l in f:
        l1.append((l.strip()).split("\t"))

nd = np.int64(l1[0][0])
nt = np.int64(l1[1][0])
re = np.float64(l1[2][0])
dt = np.float64(l1[3][0])
ns = np.int64(l1[4][0])
isolver = np.int64(l1[5][0])
isc = np.int64(l1[6][0])
ifm = np.int64(l1[7][0])
ipr = np.int64(l1[8][0])
ndc = np.int64(l1[9][0])
ichkp = np.int64(l1[10][0])
istart = np.int64(l1[11][0])
kappa = np.int64(l1[12][0])
pCU3 = np.float64(l1[13][0])
n_dns = np.int64(l1[14][0])

freq = int(nt/ns)

ifeat = 2
model = load_model('./nn_history/CNN_model_1_'+ str(8000.0) + '_' + str(512) + '_' + str(64) + '_' + str(ifeat)+'.hd5',
                   custom_objects={'coeff_determination': coeff_determination})
max_min = np.load('./nn_history/scaling.npy')



#%% 
# assign parameters
nx = nd
ny = nd

nxc = ndc
nyc = ndc

nx_dns = n_dns
ny_dns = n_dns

pi = np.pi
lx = 2.0*pi
ly = 2.0*pi

dx = lx/np.float64(nx)
dy = ly/np.float64(ny)

dxc = lx/np.float64(nxc)
dyc = ly/np.float64(nyc)

ifile = 0
time = 0.0

x = np.linspace(0.0,2.0*np.pi,nx+1)
y = np.linspace(0.0,2.0*np.pi,ny+1)

x, y = np.meshgrid(x, y, indexing='ij')

#%% 
# DA parameters
npe = 120

npx = 32
npy = 32

ne = (nx+1)*(ny+1)
me = npx*npy

mean = 0.0
sd2 = 2.0e0 # added noise (variance)
sd1 = np.sqrt(sd2) # added noise (standard deviation)

si2 = 1.0e0
si1 = np.sqrt(si2)

shift = 0.5

#xp = np.arange(0,npx)
#yp = np.arange(0,npy)

xp = np.arange(0,npx)*int(nx/npx) + int(shift*nx/npx)
yp = np.arange(0,npy)*int(ny/npy) + int(shift*ny/npy)

xprobe, yprobe = np.meshgrid(xp,yp)

oin = []
for i in xp:
    for j in yp:
        n = i*(nx+1) + j
        oin.append(n)
#        print(n)

roin = np.int32(np.linspace(0,npx*npy-1,npx*npy))
dh = np.zeros((me,ne))
dh[roin,oin] = 1.0

nf = 10         # frequency of observation
nb = int(nt/nf) # number of observation time
oib = [nf*k for k in range(nb+1)]

ns_start = 2000
ns = int(nt/nf)
z = np.zeros((me,ns+1))
zf = np.zeros((me,npe,ns+1))
DhX = np.zeros((me,npe))
DhXm = np.zeros(me)

sc = np.zeros((ne,npe))   # square-root of the covariance matrix
cn = 1.0/np.sqrt(npe-1)

data = np.load('../w_fdns_8000.0_512_64.npz')
wobs = data['wac_obs']

oib_obs = [ns_start + nf*k for k in range(ns+1)]
wobs = wobs[:,oib_obs]

for k in range(ns+1):
    wo = np.reshape(wobs[:,k],[nx+5,ny+5])
    wobs_tr = np.reshape(wo[2:nx+3,2:ny+3], [ne])
    z[:,k] = wobs_tr[oin] #+ np.random.normal(mean,sd1,[me])
    for n in range(npe):
        zf[:,n,k] = z[:,k] + np.random.normal(mean,sd1,me)
    
kobs = 1

#%% 
# allocate the vorticity and streamfunction arrays
we = np.zeros(((nx+5)*(ny+5),npe,nt+1))
wet = np.zeros(((nx+1)*(ny+1),npe,nt+1)) 

wen = np.zeros((nx+5,ny+5,npe)) 
sen = np.zeros((nx+5,ny+5,npe))

ten = np.zeros((nx+5,ny+5,npe))
ren = np.zeros((nx+5,ny+5,npe))

kstart = random.sample(range(nt), npe)

k = 0
w0 = np.reshape(wobs[:,k],[nx+5,ny+5]) #ic_decay(nx,ny,dx,dy)

for n in range(npe):
#    w0 = ic_decay(nx,ny,dx,dy)
    #w0 = np.reshape(wobs[:,kstart[n]],[nx+5,ny+5])
    
    wen[2:nx+2,2:ny+2,n] = w0[2:nx+2,2:ny+2] + np.random.normal(mean,si1,[nx,ny])
    wen[:,ny+2,n] = wen[:,2,n]
    wen[nx+2,:,n] = wen[2,:,n]
    wen[nx+2,ny+2,n] = wen[2,2,n]
    
    wen[:,:,n] = bc(nx,ny,wen[:,:,n])

    sen[:,:,n] = fps(nx, ny, dx, dy, -wen[:,:,n])
    sen[:,:,n] = bc(nx,ny,sen[:,:,n])
    
    we[:,n,k] = np.reshape(w0,[(nx+5)*(ny+5)])    

wa = np.zeros(((nx+5)*(ny+5),nt+1))
wat = np.zeros(((nx+1)*(ny+1),nt+1))

wa[:,k] = np.sum(we[:,:,k],axis=1)
wa[:,k] = wa[:,k]/npe

wf = np.zeros(ne)
Af = np.zeros((ne,npe))   # Af data
w_ = np.zeros((nx+5,ny+5))


#%%
file = open("output.txt", "a", buffering=1)

def rhs(nx,ny,dx,dy,re,pCU3,w,s,ifm,isolver,max_min,model,ifeat):
    if isolver == 1:
        return rhs_arakawa(nx,ny,dx,dy,re,w,s,ifm,kappa,max_min,model,ifeat)


# time integration using third-order Runge Kutta method
aa = 1.0/3.0
bb = 2.0/3.0
clock_time_init = tm.time()

se2 = 0.0 #np.sqrt(sd2)
se1 = np.sqrt(se2)
lambd = 1.25

for k in range(1,nt+1):
    time = time + dt
    for n in range(npe):
        ren[:,:,n] = rhs(nx,ny,dx,dy,re,pCU3,wen[:,:,n],sen[:,:,n],ifm,isolver,max_min,model,ifeat)
        
        #stage-1
        ten[2:nx+3,2:ny+3,n] = wen[2:nx+3,2:ny+3,n] + dt*ren[2:nx+3,2:ny+3,n]
        
        ten[:,:,n] = bc(nx,ny,ten[:,:,n])
        
        sen[:,:,n] = fps(nx, ny, dx, dy, -ten[:,:,n])
        sen[:,:,n] = bc(nx,ny,sen[:,:,n])
        
        ren[:,:,n] = rhs(nx,ny,dx,dy,re,pCU3,ten[:,:,n],sen[:,:,n],ifm,isolver,max_min,model,ifeat)
    
        #stage-2
        ten[2:nx+3,2:ny+3,n] = 0.75*wen[2:nx+3,2:ny+3,n] + 0.25*ten[2:nx+3,2:ny+3,n] + 0.25*dt*ren[2:nx+3,2:ny+3,n]
        
        ten[:,:,n] = bc(nx,ny,ten[:,:,n])
        
        sen[:,:,n] = fps(nx, ny, dx, dy, -ten[:,:,n])
        sen[:,:,n] = bc(nx,ny,sen[:,:,n])
        
        ren[:,:,n] = rhs(nx,ny,dx,dy,re,pCU3,ten[:,:,n],sen[:,:,n],ifm,isolver,max_min,model,ifeat)
    
        #stage-3
        wen[2:nx+3,2:ny+3,n] = aa*wen[2:nx+3,2:ny+3,n] + bb*ten[2:nx+3,2:ny+3,n] + bb*dt*ren[2:nx+3,2:ny+3,n] + np.random.normal(mean,se1,[nx+1,ny+1]) 
        
        wen[:,:,n] = bc(nx,ny,wen[:,:,n])
        
        sen[:,:,n] = fps(nx, ny, dx, dy, -wen[:,:,n])
        sen[:,:,n] = bc(nx,ny,sen[:,:,n])
        
        we[:,n,k] = np.reshape(wen[:,:,n],[(nx+5)*(ny+5)])
        wet[:,n,k] = np.reshape(wen[2:nx+3,2:ny+3,n],[(nx+1)*(ny+1)])
    
    wa[:,k] = np.sum(we[:,:,k],axis=1)
    wa[:,k] = wa[:,k]/npe
    
    wat[:,k] = np.sum(wet[:,:,k],axis=1)
    wat[:,k] = wat[:,k]/npe
    
    if k == oib[kobs]:
        print('%.2f %.2f %.2f %.2f %.2f %.2f %.2f' %(k, np.max(wat[:,k]), np.min(wat[:,k]), np.max(wet[:,:,k]), np.min(wet[:,:,k]), np.max(z[:,kobs]), np.min(z[:,kobs])))
        print('%.2f %.2f %.2f %.2f %.2f %.2f %.2f' %(k, np.max(wat[:,k]), np.min(wat[:,k]), np.max(wet[:,:,k]), np.min(wet[:,:,k]), np.max(z[:,kobs]), np.min(z[:,kobs])), file=file)
        # compute mean of the forecast fields
        wf[:] = np.sum(wet[:,:,k],axis=1)   
        wf[:] = wf[:]/npe
        
        # compute square-root of the covariance matrix
        for n in range(npe):
            sc[:,n] = cn*(wet[:,n,k] - wf[:])
        
        # compute DhXm data
        DhXm[:] = np.sum(wet[oin,:,k],axis=1)    
        DhXm[:] = DhXm[:]/npe
        
        # compute DhM data
        for n in range(npe):
            DhX[:,n] = cn*(wet[oin,n,k] - DhXm[:])
            
        # R = sd2*I, observation m+atrix
        cc = DhX @ DhX.T
        
        for i in range(me):
            cc[i,i] = cc[i,i] + sd2
        
        ph = sc @ DhX.T
                    
        ci = np.linalg.pinv(cc) # ci: inverse of cc matrix
        
        km = ph @ ci
        
        # analysis update    
        kmd = km @ (zf[:,:,kobs] - wet[oin,:,k])
        wet[:,:,k] = wet[:,:,k] + kmd[:,:]
        
        wat[:,k] = np.sum(wet[:,:,k],axis=1)
        wat[:,k] = wat[:,k]/npe
        
        #multiplicative inflation (optional): set lambda=1.0 for no inflation
        wet[:,:,k] = wat[:,k].reshape(-1,1) + lambd*(wet[:,:,k] - wat[:,k].reshape(-1,1))
        
        for n in range(npe):
            w_[2:nx+3,2:ny+3] = np.reshape(wet[:,n,k],[nx+1,ny+1])
            w_ = bc(nx,ny,w_)
            wen[:,:,n] = w_
            we[:,n,k] = np.reshape(wen[:,:,n],[(nx+5)*(ny+5)])
        
        wa[:,k] = np.sum(we[:,:,k],axis=1)
        wa[:,k] = wa[:,k]/npe
        
        kobs = kobs + 1

total_clock_time = tm.time() - clock_time_init
print('Total clock time=', total_clock_time,file=file)
np.save('cpu_time.npy',total_clock_time)

nobs = 400
oib_save = [10*k for k in range(nobs+1)]
we = we[:,:,oib_save]
np.savez('wda_'+ str(re) + '_' + str(nx) + '_' + str(ifm) + '_' + str(npx) + '.npz', ws = we)

#%%
if (ipr == 4):

    fig, axs = plt.subplots(1,2,figsize=(12,5))
    
    for ne in range(npe):
        w0 = np.reshape(we[:,ne,0], [nx+5,ny+5])
        wh = np.reshape(we[:,ne,int(nobs/2)], [nx+5,ny+5])
        wn = np.reshape(we[:,ne,-1], [nx+5,ny+5])
        en0, n = energy_spectrum(nx,ny,w0)    
        enh, n = energy_spectrum(nx,ny,wh) 
        ent, n = energy_spectrum(nx,ny,wn)
        k = np.linspace(1,n,n)
            
        #axs[0].loglog(k,en0[1:],'r', ls = '-', lw = 2, label='$t = 0.0$')
        
        print(np.max(ent))
        axs[0].loglog(k,enh[1:],'k', ls = '-', lw = 2, alpha = 0.2,label='$t = '+str(0.5*dt*nt)+'$')
        #axs[0].loglog(k,ent[1:], 'b', lw = 2, alpha = 0.5, label = '$t = '+str(dt*nt)+'$')
        
        axs[1].loglog(k,ent[1:], 'k', lw = 2, alpha = 0.2, label = '$t = '+str(dt*nt)+'$')
    
    w0 = np.reshape(wa[:,0], [nx+5,ny+5])
    wh = np.reshape(wa[:,int(nt/2)], [nx+5,ny+5])
    wn = np.reshape(wa[:,-1], [nx+5,ny+5])
    en0, n = energy_spectrum(nx,ny,w0)    
    enh, n = energy_spectrum(nx,ny,wh) 
    ent, n = energy_spectrum(nx,ny,wn)
    k = np.linspace(1,n,n)
    
    axs[0].loglog(k,enh[1:],'r', ls = '--', lw = 2, alpha = 1.0, label='$t = '+str(0.5*dt*nt)+'$')
    #axs[0].loglog(k,ent[1:], 'b', ls = '--', lw = 2, alpha = 1.0, label = '$t = '+str(dt*nt)+'$')
    
    axs[1].loglog(k,ent[1:], 'r', ls = '--', lw = 2, alpha = 1.0, label = '$t = '+str(dt*nt)+'$')
    
    line = 100*k**(-3.0)
    axs[0].loglog(k,line, 'k--', lw = 2, )
    axs[1].loglog(k,line, 'k--', lw = 2, )
    
    axs[0].text(0.75, 0.75, '$k^{-3}$', transform=axs[0].transAxes, fontsize=16, fontweight='bold', va='top')
    axs[1].text(0.75, 0.75, '$k^{-3}$', transform=axs[1].transAxes, fontsize=16, fontweight='bold', va='top')
    
    
    axs[0].set_xlabel('$K$')
    axs[0].set_ylabel('$E(K)$')
    #axs[0].legend(loc=3)
    axs[0].set_ylim(1e-8,1e0)
    axs[0].set_title('$t$ = ' + str(0.5*nt*dt))
    
    axs[1].set_xlabel('$K$')
    axs[1].set_ylabel('$E(K)$')
    #axs[0].legend(loc=3)
    axs[1].set_ylim(1e-8,1e0)
    axs[1].set_title('$t$ = ' + str(1.0*nt*dt))
    
    plt.show()
    fig.savefig('eda_'+ str(re) + '_' + str(nx) + '_' + str(ifm) + '_' + str(npx) + 'lambda=2.png', 
                bbox_inches = 'tight', pad_inches = 0, dpi = 300)

#%%    
file.close()
