#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 19:17:22 2020

@author: suraj
"""

import numpy as np
from numpy.random import seed
seed(1)
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt 
import time as tm
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
data = np.load('data_0.npz')
ua_0 = data['uw']

data = np.load('data_9.npz')
T = data['T']
X = data['X']
utrue_9 = data['utrue']
uw_9 = data['uw']
ua_9 = data['ua']

data = np.load('data_18.npz')
T = data['T']
X = data['X']
utrue_18 = data['utrue']
uw_18 = data['uw']
ua_18 = data['ua']


nts = 10000
utrue = utrue_9[:,nts:]
diff_0 = utrue - ua_0
diff_9 = utrue - ua_9
diff_18 = utrue - ua_18

#%%
vmin = -12
vmax = 12

field = [utrue,ua_0,ua_9,ua_18,diff_0,diff_9,diff_18]
#label = ['True','True','True','EnKF','EnKF','EnKF','Error','Error','Error']
title = ['(a) True', '(b) ANN-5', '(c) DEnKF ($m=9$)', '(d) DEnKF ($m=18$)', '(e) Error (ANN-5)',
         '(f) Error (DEnKF ($m=9$))','(g) Error (DEnKF ($m=18$))']
ylabel = [r'$X$',r'$X$',r'$X$',r'$X$',r'$\epsilon$',r'$\epsilon$',r'$\epsilon$']

fig,ax = plt.subplots(figsize=(12,10),sharex=True,constrained_layout=True)

AX = gridspec.GridSpec(4,4)
AX.update(wspace = 0.5, hspace = 0.5)
axs1  = plt.subplot(AX[0,1:3])
axs2 = plt.subplot(AX[1,0:2])
axs3 = plt.subplot(AX[1,2:])
axs4 = plt.subplot(AX[2,0:2])
axs5 = plt.subplot(AX[2,2:])
axs6 = plt.subplot(AX[3,0:2])
axs7 = plt.subplot(AX[3,2:])

#axs1 = plt.subplot2grid((4, 4), (0, 1), colspan=2)
#axs2 = plt.subplot2grid((4, 4), (1, 0), colspan=2)
#axs3 = plt.subplot2grid((4, 4), (1, 2), colspan=2)
#axs4 = plt.subplot2grid((4, 4), (2, 0), colspan=2)
#axs5 = plt.subplot2grid((4, 4), (2, 2), colspan=2)
#axs6 = plt.subplot2grid((4, 4), (3, 0), colspan=2)
#axs7 = plt.subplot2grid((4, 4), (3, 2), colspan=2)

axs = [axs1, axs2, axs4, axs6, axs3, axs5, axs7]

for i in range(7):
    cs = axs[i].contourf(T,X,field[i],40,cmap='jet',vmin=vmin,vmax=vmax,zorder=-9)
    axs[i].set_rasterization_zorder(-1)
    axs[i].set_ylabel(ylabel[i],size=20)
    for c in cs.collections:
        c.set_edgecolor("face")
    axs[i].set_title(title[i],size=16)

axs[3].set_xlabel(r'$t$',size=18)
axs[6].set_xlabel(r'$t$',size=18)

m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
#fig.colorbar(m,ax=axs[0],ticks=np.linspace(vmin, vmax, 6))

#fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.25, 0.03, 0.5, 0.025])
fig.colorbar(m, cax=cbar_ax,orientation='horizontal')

fig.tight_layout()
plt.show()  
fig.savefig('field_plot_ann_da.pdf',bbox_inches='tight',dpi=300)
fig.savefig('field_plot_ann_da.eps',bbox_inches='tight')
fig.savefig('field_plot_ann_da.png',bbox_inches='tight',dpi=300)

