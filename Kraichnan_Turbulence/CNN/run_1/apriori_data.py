#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:00:38 2020

@author: suraj
"""

import numpy as np
from numpy.random import seed
seed(1)
import pyfftw
from scipy import integrate
from scipy.integrate import simps
from scipy import linalg
import matplotlib.pyplot as plt 
import time as tm
import matplotlib.ticker as ticker
import os
from numba import jit
from utils import *

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%% 
# assign parameters
nx = 512
ny = 512

nxc = 64
nyc = 64

pi = np.pi
lx = 2.0*pi
ly = 2.0*pi

dx = lx/np.float64(nx)
dy = ly/np.float64(ny)

dxc = lx/np.float64(nxc)
dyc = ly/np.float64(nyc)

re = 8000
ns = 600
k0 = 10.0

dt = 1e-3
nt = 6000

ifile = 0
time = 0.0

x = np.linspace(0.0,2.0*np.pi,nx+1)
y = np.linspace(0.0,2.0*np.pi,ny+1)
x, y = np.meshgrid(x, y, indexing='ij')

xc = np.linspace(0.0,2.0*np.pi,nxc+1)
yc = np.linspace(0.0,2.0*np.pi,nyc+1)
xc, yc = np.meshgrid(xc, yc, indexing='ij')

t = np.linspace(0,nt*dt,ns+1)

apriori_data = True

#%%
if apriori_data:
    folder_in = './results/data_'+str(int(re))+'_'+str(nx)+'_'+str(ny)
    folder_out = './results/data_'+str(int(re))+'_fdns_'+str(nx)+'_'+str(nxc)
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    
    wc = np.zeros((nxc+5,nyc+5))
    sc = np.zeros((nxc+5,nyc+5))
    w = np.zeros((nx+5,ny+5))
    s = np.zeros((nx+5,ny+5))
    
    ne = (nxc+5)*(nyc+5)
    wac_obs = np.zeros((ne,ns+1))
    
    nu_avg = np.zeros((ns+1))
    
    for n in range(ns+1):
        print(n)
    
        file_in= folder_in+'/results_'+str(n)+'.npz'
        data = np.load(file_in)
        
        w = data['w']
        s = np.zeros((nx+5,ny+5))
        s[2:nx+3,2:ny+3] = fpsd(nx, ny, dx, dy, w[2:nx+3,2:ny+3])
        s = bc(nx,ny,s)
        
        wc[2:nxc+3,2:nyc+3] = coarsen(nx,ny,nxc,nyc,w[2:nx+3,2:ny+3]) # change to [2:nx+3,2:ny+3] afterwards
        wc = bc(nxc,nyc,wc)
        
        wac_obs[:,n] = np.reshape(wc,[ne])
        
        sc[2:nxc+3,2:nyc+3] = fpsd(nxc,nyc,dxc,dyc,wc[2:nxc+3,2:nyc+3])
        sc = bc(nxc,nyc,sc)
        
        jac_f = jacobian(nx,ny,dx,dy,re,w,s)
        jac_ff = coarsen(nx,ny,nxc,nyc,jac_f)
        jac_c = jacobian(nxc,nyc,dxc,dyc,re,wc,sc)
        
        pi_source = jac_c - jac_ff
        
        wcx,wcy = grad_spectral(nxc,nyc,wc[2:nx+3,2:ny+3])
        wcxx,wcxy = grad_spectral(nxc,nyc,wcx)
        wcyx,wcyy = grad_spectral(nxc,nyc,wcy)
        
        scx,scy = grad_spectral(nxc,nyc,sc[2:nx+3,2:ny+3])
        scxx,scxy = grad_spectral(nxc,nyc,scx)
        scyx,scyy = grad_spectral(nxc,nyc,scy)
        
        kernel_w = np.sqrt(wcx**2 + wcy**2)
        kernel_s = np.sqrt(4.0*scxy**2 + (scxx - scyy)**2)
        
        lap = wcxx + wcyy
        
        nu_e = pi_source/lap
        
        file_out = folder_out+'/results_'+str(n)+'.npz'
        np.savez(file_out,wc=wc[2:nxc+3,2:nyc+3],sc=sc[2:nxc+3,2:nyc+3],
                 nue=nu_e,lap=lap,pi=pi_source,ks=kernel_s,kw=kernel_w,
                 wcx=wcx,wcy=wcy,wcxx=wcxx,wcyy=wcyy,wcxy=wcxy,
                 scx=scx,scy=scy,scxx=scxx,scyy=scyy,scxy=scxy)
        
        nu_avg[n] = np.mean(nu_e)
    
    en_dns, n_dns = energy_spectrumd(nx,ny,dx,dy,w[2:nx+3,2:ny+3])
    k_dns = np.linspace(1,n_dns,n_dns)
    en_fdns, n_fdns = energy_spectrumd(nxc,nyc,dxc,dyc,wc[2:nxc+3,2:nyc+3])
    k_fdns = np.linspace(1,n_fdns,n_fdns)
    
    fig, ax = plt.subplots(1,1,figsize=(6,6))

    ax.loglog(k_dns,en_dns[1:], 'b',  lw = 2, label = 'DNS - '+str(nx))
    ax.loglog(k_fdns,en_fdns[1:], 'r',  lw = 2, label = 'fDNS - '+str(nxc))
        
    ax.legend()    
    plt.show()
    filename = folder_out + '/energy_spectrum_' + str(nx)+'_'+str(nxc) + '.pdf'
    fig.savefig(filename)

re = 8000.0
np.savez('w_fdns_'+ str(re) + '_' + str(nx) + '_' + str(nxc) + '_dns_snapshots.npz', wac_obs = wac_obs)

#%%
fig, ax = plt.subplots(1,1,figsize=(6,5))

ax.loglog(k_dns,en_dns[1:], 'b',  lw = 2, label = 'DNS - '+str(nx))
ax.loglog(k_fdns,en_fdns[1:], 'r',  lw = 2, label = 'fDNS - '+str(nxc))
ax.set_ylim(1e-10,1e0)
    
ax.legend()    
plt.show()
filename = folder_out + '/energy_spectrum_' + str(nx)+'_'+str(nxc) + '.pdf'
fig.savefig(filename)