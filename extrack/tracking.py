#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:23:30 2022

@author: francois
"""
import numpy as np

GPU_computing = False

if GPU_computing :
    import cupy as cp
    from cupy import asnumpy
else :
    import numpy as cp
    def asnumpy(x):
        return np.array(x)

from scipy import linalg
import scipy
from lmfit import minimize, Parameters

from itertools import product
from numba import njit, typed, prange, jit

from extrack.refined_localization import position_refinement

from time import time
from itertools import product

from time import time
'''
Maximum likelihood to determine transition rates :
We compute the probability of observing the tracks knowing the parameters :
For this, we assum a given set of consecutive states ruling the displacement terms,
we express the probability of having a given set of tracks and real positions.
Next, we integrate reccursively over the possible real positions to find the probability
of the track knowing the consecutive states. This recurrance can be performed as the
integral of the product of normal laws is a constant time a normal law.
In the end of the reccurance process we have a constant time normal law of which 
all the terms are known and then a value of the probability.
We finally use the conditional probability principle over the possible set of states
to compute the probability of having the tracks. 
'''

do_jit = True
args = {'nopython': True, 'cache': True}
#args = {'forceobj': True}


def ds_froms_states(cur_d2s, d2s, cur_Bs, cur_nb_Bs, intermediate_d2s):
    for i in range(cur_nb_Bs):
        cur_d2 = d2s[cur_Bs[i,0]]
        for j in range(1,cur_Bs.shape[1]):            
            cur_d2_plus1 = d2s[cur_Bs[i,j]]
            intermediate_d2s[j-1] = (cur_d2 + cur_d2_plus1)*0.5# assuming a transition at the middle of the substeps
            cur_d2 = cur_d2_plus1
    # we can average the variances of displacements per step to get the actual std of displacements per step
        cur_d2s[i,0] = np.mean(intermediate_d2s)

if do_jit:
    ds_froms_states = jit(ds_froms_states, **args)

'''
%timeit cur_d2s = ds_froms_states(cur_d2s, d2s, cur_Bs, cur_nb_Bs, intermediate_d2s)

@jit(**args)
def f(cur_d2s, d2s, cur_Bs, cur_nb_Bs, intermediate_d2s):
    cur_d2s = ds_froms_states(cur_d2s, d2s, cur_Bs, cur_nb_Bs, intermediate_d2s)

alternative :
@jit(**args)
def ds_froms_states(cur_d2s, d2s, cur_Bs, cur_nb_Bs, intermediate_d2s):
    for i in range(cur_nb_Bs):
        cur_d2 = d2s[cur_Bs[i,0]]
        for j in range(1,cur_Bs.shape[1]):            
            cur_d2_plus1 = d2s[cur_Bs[i,j]]
            intermediate_d2s[j-1] = (cur_d2 + cur_d2_plus1)*0.5  # assuming a transition at the middle of the substeps
    # we can average the variances of displacements per step to get the actual std of displacements per step
        cur_d2s[i,0] = np.mean(intermediate_d2s)
    return cur_d2s[:cur_nb_Bs]
'''


#Ci, l2 = Cs[:,nb_locs-current_step], LocErr2[:,min(LocErr_index,nb_locs-current_step)]
"""
@jit(**args)
def log_integrale_dif(Ci, l2, cur_d2s, m_arr, s2_arr, LP, cur_nb_Bs, nb_dims):
    '''
    integral of the 3 exponetional terms (localization error, diffusion, previous term)
    the integral over r1 of f_l(r1-c1)f_d(r1-r0)f_Ks(r1-m_arr) equals :
    np.exp(-((l**2+Ks**2)*r0**2+(-2*m_arr*l**2-2*Ks**2*c1)*r0+m_arr**2*l**2+(m_arr**2-2*c1*m_arr+c1**2)*d**2+Ks**2*c1**2)/((2*d**2+2*Ks**2)*l**2+2*Ks**2*d**2))/(2*np.pi*Ks*d*l*np.sqrt((d**2+Ks**2)*l**2+Ks**2*d**2))
    which can be turned into the form Constant*fKs(r0 - newm_arr) where fKs is a normal law of std newKs
    the idea is to create a function of integral of integral of integral etc
    dim 0 : tracks
    dim 1 : possible sequences of states
    dim 2 : x,y (z)
    '''
    l2_plus_s2_arr = l2+s2_arr[:cur_nb_Bs]
    if s2_arr.shape[1] == 1:
        LP[:cur_nb_Bs] += m_arr.shape[1] * - 0.5*cp.log(2*np.pi*(l2_plus_s2_arr[:,0]))
        for dim in range(nb_dims):
            LP[:cur_nb_Bs] -= ((Ci[:, dim]-m_arr[:cur_nb_Bs, dim]))**2/(2*l2_plus_s2_arr[:,0])
    #else:
    #    LP[:cur_nb_Bs] += np.sum(-0.5*cp.log(2*np.pi*(l2_plus_s2_arr)), 1) - cp.sum(((Ci-m_arr))**2/(2*l2_plus_s2_arr),axis = 1)
    m_arr = (m_arr[:cur_nb_Bs]*l2 + Ci*s2_arr[:cur_nb_Bs])/(l2+s2_arr[:cur_nb_Bs])
    s2_arr = ((cur_d2s[:cur_nb_Bs]*l2 + cur_d2s[:cur_nb_Bs]*s2_arr[:cur_nb_Bs] + l2*s2_arr[:cur_nb_Bs])/l2_plus_s2_arr)
"""

def log_integrale_dif(Ci, l2, cur_d2s, m_arr, s2_arr, LP, cur_nb_Bs):
    '''
    integral of the 3 exponetional terms (localization error, diffusion, previous term)
    the integral over r1 of f_l(r1-c1)f_d(r1-r0)f_Ks(r1-m_arr) equals :
    np.exp(-((l**2+Ks**2)*r0**2+(-2*m_arr*l**2-2*Ks**2*c1)*r0+m_arr**2*l**2+(m_arr**2-2*c1*m_arr+c1**2)*d**2+Ks**2*c1**2)/((2*d**2+2*Ks**2)*l**2+2*Ks**2*d**2))/(2*np.pi*Ks*d*l*np.sqrt((d**2+Ks**2)*l**2+Ks**2*d**2))
    which can be turned into the form Constant*fKs(r0 - newm_arr) where fKs is a normal law of std newKs
    the idea is to create a function of integral of integral of integral etc
    dim 0 : possible sequences of states
    dim 1 : x,y (z)
    '''
    l2_plus_s2_arr = l2 + s2_arr[:cur_nb_Bs]
    
    if s2_arr.shape[1] == 1:
        LP[:cur_nb_Bs] += m_arr.shape[1] * - 0.5*cp.log(2*np.pi*(l2_plus_s2_arr[:,0])) - cp.sum(((Ci-m_arr[:cur_nb_Bs]))**2/(2*l2_plus_s2_arr),axis = 1)
    else:
        LP[:cur_nb_Bs] += np.sum(-0.5*cp.log(2*np.pi*(l2_plus_s2_arr)), 1)             - cp.sum(((Ci-m_arr[:cur_nb_Bs]))**2/(2*l2_plus_s2_arr),axis = 1)
    
    m_arr[:cur_nb_Bs] = (m_arr[:cur_nb_Bs]*l2 + Ci*s2_arr[:cur_nb_Bs])/(l2+s2_arr[:cur_nb_Bs])
    s2_arr[:cur_nb_Bs] = ((cur_d2s[:cur_nb_Bs]*l2 + cur_d2s[:cur_nb_Bs]*s2_arr[:cur_nb_Bs] + l2*s2_arr[:cur_nb_Bs])/l2_plus_s2_arr)

if do_jit:
    log_integrale_dif = jit(log_integrale_dif, **args)

#Ci, l2, cur_d2s = Cs[:, nb_locs-current_step], LocErr2[:, min(LocErr_index, nb_locs-current_step)], cur_d2s

def first_log_integrale_dif(Ci, l2, cur_d2s, cur_nb_Bs, m_arr, s2_arr):
    '''
    convolution of 2 normal laws = normal law (mean = sum of means and variance = sum of variances)
    ''' 
    for i in range(cur_nb_Bs):
        s2_arr[i] = l2[0]+cur_d2s[i]
        m_arr[i] = Ci
    
    return m_arr, s2_arr

if do_jit:
    first_log_integrale_dif = jit(first_log_integrale_dif, **args)

def last_integrale(LP, Cs, m_arr, s2_arr, LocErr2, nb_locs, current_step, LocErr_index, nb_dims, cur_nb_Bs):
    new_s2_arr = s2_arr[:cur_nb_Bs] + LocErr2[:, min(LocErr_index, nb_locs-current_step)]
    temp_result = -0.5*cp.log(2*np.pi*new_s2_arr) - (Cs[:,0] - m_arr[:cur_nb_Bs])**2/(2*new_s2_arr)
    for k in range(nb_dims):
        LP[:cur_nb_Bs] += temp_result[:,k]        

if do_jit:
    last_integrale = jit(last_integrale, **args)

#anomalous_rectification(cur_Bs_cat, cur_Bs, gammas, betas, s2_arr, m"_arr, previous_mus, next_Cs, cur_d2s, LP, l2, cur_nb_Bs, current_step)
"""
def anomalous_rectification(cur_Bs_cat, cur_Bs, gammas, betas, s2_arr, m_arr, previous_mus, next_Cs, cur_d2s, LP, l2, cur_nb_Bs, current_step):
    norm_disp = np.zeros(m_arr.shape[-1])
    
    for k in range(cur_nb_Bs):
        state_args = np.argmax(cur_Bs_cat[k, -current_step:], axis = 1) # 
        current_state = int(state_args[0])
        current_gamma = gammas[current_state]
        correct_persistent = False
        if current_gamma != 0:
            l = np.argmin(state_args == current_state)
            if l == 0:
                l = state_args.shape[0]
            l += -2
            
            if l >= 1:
                
                weight_factors = betas[current_state] * np.arange(l)
                weight_factors = weight_factors - weight_factors[-1]
                weigths = np.exp(weight_factors)
                weigths = weigths / np.sum(weigths)
                
                cur_previous_mus = previous_mus[:l+1,k]
                
                disps = cur_previous_mus[-l:] - cur_previous_mus[-l-1:-1]
                for i in range(norm_disp.shape[0]):
                    norm_disp[i] = np.sum(disps[:,i] * weigths)
                norm_disp = norm_disp / np.sum(norm_disp**2)**0.5
                m_arr[k] += current_gamma * norm_disp
                if current_gamma > 0:
                    s2_arr[k] += 2*l2[0]/(l+1)**2  # 2*l2[0]/(l+1)**2 increase the localization error to consider error on the direction of the motion
            
            elif l == 0 and current_gamma > 0:
                correct_persistent = True
                #s2_arr[k] += 2*l2[0] + current_gamma**2  # 2*l2[0] increase the localization error to consider error on the direction of the motion
                s2_arr[k] += current_gamma**2
            
            
            else:
                previous_gamma = gammas[int(state_args[1])]
                if l < 0 and previous_gamma > 0:
                    correct_persistent = True
            
            if correct_persistent:
                '''
                in this case, the pdf is actually circular around the previous position, however we can't take circular pdf into account.
                Moreover, the transition may have occured at any time, Therefore the pdf follows a uniform disk of radius gamma instead 
                of a gaussian pdf of std l2_plus_s2_arr. Thus we need to correct for that.
                '''
                
                disps2 = next_Cs[0] - previous_mus[current_step-2,k]
                norm = np.sum(disps2**2)**0.5
                max_disp = current_gamma + np.mean(3*l2**0.5) # threshold to apply the correction
                
                if norm < max_disp and current_gamma > np.sum(l2)**0.5:
                    l2_plus_s2_arr = s2_arr[k] + l2[0]
                    
                    if s2_arr.shape[1] == 1:
                        LP[:cur_nb_Bs] += m_arr.shape[1] * 0.5*np.log(2*np.pi*l2_plus_s2_arr[0]) + m_arr.shape[1] * -0.5 * np.log(np.pi*current_gamma**2) # 
                    else:
                        LP[:cur_nb_Bs] += np.sum(-0.5*np.log(2*np.pi*(l2_plus_s2_arr))) + m_arr.shape[1] * -0.5 * np.log(np.pi*current_gamma**2) 
                    
                    m_arr[k] = next_Cs
                    s2_arr[k] += 2*l2[0]/(l+1)**2
                
                else: 
                    # if the next position is too far away, we dont apply the threshold
                    
                    norm_disp2 = disps2 / norm
                    m_arr[k] += current_gamma * norm_disp2
                    s2_arr[k] += 2*l2[0]/(l+1)**2

#"""

"""
def anomalous_rectification(cur_Bs_cat, cur_Bs, gammas, betas, s2_arr, m_arr, previous_mus, next_Cs, cur_d2s, LP, l2, cur_nb_Bs, current_step):
    norm_disp = np.zeros(m_arr.shape[-1])
    for k in range(cur_nb_Bs):
        state_args = np.argmax(cur_Bs_cat[k, -current_step:], axis = 1) # 
        current_state = int(state_args[0])
        current_gamma = gammas[current_state]
        if current_gamma != 0:
            l = np.argmin(state_args == current_state)
            if l == 0:
                l = state_args.shape[0]
            l += -2
            
            if l >= 1:
                
                weight_factors = betas[current_state] * np.arange(l)
                weight_factors = weight_factors - weight_factors[-1]
                weigths = np.exp(weight_factors)
                weigths = weigths / np.sum(weigths)
                
                cur_previous_mus = previous_mus[:l+1,k]
                
                disps = cur_previous_mus[-l:] - cur_previous_mus[-l-1:-1]
                for i in range(norm_disp.shape[0]):
                    norm_disp[i] = np.sum(disps[:,i] * weigths)
                norm_disp = norm_disp / np.sum(norm_disp**2)**0.5
                m_arr[k] += current_gamma * norm_disp
                if current_gamma > 0:
                    s2_arr[k] += 4*l2[0]/(l+1)**2  # 2*l2[0]/(l+1)**2 increase the localization error to consider error on the direction of the motion
            if l == 0 and current_gamma > 0:
                '''
                in this case, the pdf is actually circular around the previous position, however we can't take circular pdf into account.
                Moreover, the transition may have occured at any time, Therefore the pdf follows a uniform disk of radius gamma instead 
                of a gaussian pdf of std l2_plus_s2_arr. Thus we need to correct for that.
                '''
                '''
                s2_arr[k] += 2*l2[0] + current_gamma**2  # 2*l2[0] increase the localization error to consider error on the direction of the motion
                #s2_arr[k] += current_gamma**2
                
                disps2 = next_Cs[0] - previous_mus[current_step-2,k]
                norm = np.sum(disps2**2)**0.5
                max_disp = current_gamma + np.mean(3*l2**0.5) # threshold to apply the correction
                
                if norm < max_disp and current_gamma > np.sum(l2)**0.5:
                    l2_plus_s2_arr = s2_arr[k] + l2[0]
                    
                    if s2_arr.shape[1] == 1:
                        LP[:cur_nb_Bs] += m_arr.shape[1] * 0.5*np.log(2*np.pi*l2_plus_s2_arr[0]) + m_arr.shape[1] * -0.5 * np.log(np.pi*current_gamma**2) # 
                    else:
                        LP[:cur_nb_Bs] += np.sum(-0.5*np.log(2*np.pi*(l2_plus_s2_arr))) + m_arr.shape[1] * -0.5 * np.log(np.pi*current_gamma**2) 
                    
                    m_arr[k] = next_Cs
                    #m_arr[k] += current_gamma * disps2 / norm
            if l < 1 :
                # case where a transition is happening
                s2_arr[k] = l2[0] + current_gamma**2 # we reset the std term to the localization error + the current gamma (alternative to ), 

"""

def anomalous_rectification(cur_Bs_cat, cur_Bs, gammas, betas, s2_arr, m_arr, previous_mus, next_Cs, disp, precomputed_disps, cur_d2s, LP, l2, cur_nb_Bs, current_step):
    norm_disp = np.zeros(m_arr.shape[-1])
    delta_L = m_arr.shape[1] * 0.5 * np.log(2*np.pi*np.mean(l2[0])) / 500
    for k in range(cur_nb_Bs):
        state_args = np.argmax(cur_Bs_cat[k, -current_step:], axis = 1) # 
        current_state = int(state_args[0])
        current_gamma = gammas[current_state]
        if current_gamma != 0:
            l = np.argmin(state_args == current_state)
            if l == 0:
                l = state_args.shape[0]
            l += -2
            
            if l >= 1:
                if precomputed_disps:
                    sum_disp = np.sum(disp**2)**0.5
                    if sum_disp != 0:
                        norm_disp = disp / np.sum(disp**2)**0.5
                    else:
                        norm_disp = np.zeros(disp.shape)
                else:
                    weight_factors = betas[current_state] * np.arange(l)
                    weight_factors = weight_factors - weight_factors[-1]
                    weigths = np.exp(weight_factors)
                    weigths = weigths / np.sum(weigths)
                    
                    cur_previous_mus = previous_mus[:l+1,k]
                    
                    disps = cur_previous_mus[-l:] - cur_previous_mus[-l-1:-1]
                    for i in range(norm_disp.shape[0]):
                        norm_disp[i] = np.sum(disps[:,i] * weigths)
                    norm_disp = norm_disp / np.sum(norm_disp**2)**0.5
                
                m_arr[k] += current_gamma * norm_disp
                if current_gamma > 0:
                    s2_arr[k] += 2*l2[0]/(l+1)**2 #+ l2[0]  # 2*l2[0]/(l+1)**2 increase the localization error to consider error on the direction of the motion
                    LP[k] += delta_L
            if l < 1: # and current_gamma**2 > l2:
                '''
                in this case, the pdf is actually circular around the previous position, however we can't take circular pdf into account.
                Moreover, the transition may have occured at any time, Therefore the pdf follows a uniform disk of radius gamma instead 
                of a gaussian pdf of std l2_plus_s2_arr. Thus we need to correct for that.
                '''
                #s2_arr[k] += 2*l2[0] + current_gamma**2  # 2*l2[0] increase the localization error to consider error on the direction of the motion
                # case where a transition is happening
                #s2_arr[k] = 2*l2[0] + current_gamma**2 # we reset the std term to the localization error + the current gamma (alternative to ), 
                #s2_arr[k] = l2[0] + current_gamma**2 # we reset the std term to the localization error + the current gamma (alternative to ), 
                s2_arr[k] += current_gamma**2 # we reset the std term to the localization error + the current gamma (alternative to ), 

#"""

'''
track_len = 10
positions = np.arange(track_len)[None,:]*0.1 + np.random.normal(0,0.02, (10000, track_len))

positions = np.random.normal(0,0.02, (10000, track_len))

np.std((positions[:,1] - positions[:,0])/0.3)
 
 
 
 2*l2/(0.02 * (l+1))**2


2**0.5

l = 0
stds = []
for k in range(1, track_len):
    stds.append(np.std((positions[:,k] - positions[:,0]))/l)

arr = np.zeros((9, 10000))

plt.figure()

for l in range(1, track_len):
    for j, track in enumerate(positions):
        a, b = np.polyfit(np.arange(l+1), track[:l+1], 1)
        arr[l-1, j] = a

'''

'''
previous_mus[current_step-1-l:current_step-1,k]

previous_mus[current_step-2, k]

d = cur_d2s[k]**0.5
 s2_arr[k]**0.5

(2*np.pi)**(-3/2)/(current_gamma * std) /  (1/(2*np.pi*std**2))

(2*np.pi)**(-1/2)*std / current_gamma

cur_Bs_cat[k]


np.log((2*np.pi)**(-3/2)/(current_gamma * std)) - np.log((1/(2*np.pi*std**2)))


Cs = np.array([[[0.01,0],
                [0.02,0],
                [0.03,0],
                [0.1,0],
                [0.2,0],
                [0.3,0]]])[:,::-1]

nb_locs = Cs.shape[1]

LocErr2 = np.array([[[0.0004]]])
'''
if do_jit:
    anomalous_rectification = jit(anomalous_rectification, **args)

# Cs, LocErr2, ds, Fs, TrMat, nb_locs, nb_dims, nb_states, gammas, betas, pBL,isBL, cell_dims, nb_substeps, frame_len, min_len, threshold, max_nb_states, cartesian_prod, cartesian_prod_cat = args_prod[0]
do_preds = 0
#%timeit LP, cur_Bs_cat, preds = P_Cs_inter_bound_stats_th(Cs, LocErr, ds, Fs, TrMat, nb_locs, nb_dims, nb_states, cartesian_prod, cartesian_prod_cat, gammas, betas, pBL,isBL, cell_dims, nb_substeps, frame_len, min_len, threshold, max_nb_states)
#%timeit P_Cs_inter_bound_stats_th(Cs, LocErr2, ds, Fs, TrMat, nb_locs, nb_dims, nb_states, cartesian_prod, cartesian_prod_cat, gammas, betas, pBL, isBL, cell_dims, nb_substeps, frame_len, do_preds, min_len, threshold, max_nb_states)
# Cs,  nb_locs, isBL = (all_tracks[k], all_nb_locs[k], isBLs[k])

def P_Cs_inter_bound_stats_th(Cs, LocErr2, ds, Fs, TrMat, nb_locs, nb_dims, nb_states, cartesian_prod, cartesian_prod_cat, disps, precomputed_disps = False, gammas = [-0.02,0.02], betas = [0.5,0.5], pBL=0.1, isBL = 1, cell_dims = [0.5], nb_substeps=2, frame_len = 6, do_preds = 1, min_len = 3, threshold = 0.05, max_nb_states = 30):
    '''
    compute the product of the integrals over Ri as previousily described
    work in log space to avoid overflow and underflow
    
    Cs : dim 0 = track ID, dim 1 : states, dim 2 : peaks postions through time,
    
    dim 3 : x, y position
    
    we process by steps, at each step we account for 1 more localization, we compute
    the canstant (LC), the mean (m_arr) and std (Ks) of of the normal distribution 
    resulting from integration.
    
    each step is made of substeps if nb_substeps > 1, and we increase the matrix
    of possible Bs : cur_Bs accordingly
    
    to be able to process long tracks with good accuracy, for each track we fuse m_arr and Ks
    of sequences of states equal exept for the state 'frame_len' steps ago.
    '''
    #threshold = 0.2
    
    #Cs, LocErr, ds, Fs, TrMat, nb_locs, nb_dims, nb_states, gammas, betas, pBL,isBL, cell_dims, nb_substeps, frame_len, min_len, threshold, max_nb_states, cartesian_prod, cartesian_prod_cat = args_prod[0]
    #max_nb_states = 120
    #ds = np.array([0, 0.25])
    
    current_step = 1
    
    if LocErr2.shape[1] == 1: # combined to min(LocErr_index, nb_locs-current_step) it will select the right index
        LocErr_index = -1
        vary_LocErr_index = False
    elif LocErr2.shape[1] == nb_locs:
        vary_LocErr_index = True
        LocErr_index = nb_locs-current_step
    else:
        raise ValueError("Localization error is not specified correctly, see the following link for more details: ")
    
    TrMat = TrMat.T # could be put outside the function to gain a bit of time
    
    cur_nb_Bs = nb_states**(nb_substeps + 1)
    cur_Bs = np.zeros((max_nb_states, nb_substeps + 1), dtype=np.int8)-1
    LP = np.zeros((max_nb_states), dtype=np.float64)
    m_arr = np.zeros((max_nb_states, nb_dims), dtype=np.float64)-1
    s2_arr = np.zeros((max_nb_states, LocErr2.shape[2]), dtype=np.float64)-1
    previous_mus = np.zeros((nb_locs, max_nb_states, nb_dims), dtype=np.float64)-1
    cur_d2s = np.zeros((max_nb_states,1), dtype=np.float64)-1
    
    cur_Bs[:cur_nb_Bs] = get_all_Bs(nb_substeps + 1, nb_states) # get initial sequences of states
    
    cur_Bs_cat = np.zeros((max_nb_states, nb_locs, nb_states), dtype=np.float64)-1
    cur_Bs_cat[:cur_nb_Bs, -2:] = to_Categorical(cur_Bs[:cur_nb_Bs], nb_states)
    
    # compute the vector of diffusion stds knowing the current states
    
    anomalous = np.any(gammas != 0)
    #beta = np.log(2)
    #np.exp(-np.arange(10)*np.log(2))
    
    get_Ts_from_Bs(LP, cur_Bs[:cur_nb_Bs], TrMat, cur_nb_Bs) # update the probability of the sequences of states to consider state transitions
    LP[:cur_nb_Bs] += cp.log(Fs[cur_Bs[:cur_nb_Bs, -1]]) # update the probability to consider the initial state
    
    # current log proba of seeing the track
    d2s = ds**2
    intermediate_d2s = np.empty(nb_substeps)
    # we can average the variances of displacements per step to get the actual std of displacements per step
    ds_froms_states(cur_d2s, d2s, cur_Bs, cur_nb_Bs, intermediate_d2s)
    
    Lp_stay, p_stay, sub_Bs = compute_P_leave(cur_Bs[:cur_nb_Bs], nb_substeps, nb_states, d2s, cell_dims, pBL, cur_nb_Bs) # proba for the track to survive = both stay in the FOV and not bleach
    
    # inject the first position to get the associated m_arr and Ks :
    m_arr, s2_arr = first_log_integrale_dif(Cs[:, nb_locs-current_step], LocErr2[:,LocErr_index], cur_d2s,  cur_nb_Bs, m_arr, s2_arr)
    
    previous_mus[0,] = m_arr
    
    current_step += 1
    
    cur_nb_Bs, current_step = reccurence_loop(Cs, disps, precomputed_disps, min_len, current_step, nb_locs, anomalous, m_arr, s2_arr, cur_Bs_cat, cur_Bs, gammas, betas, previous_mus, nb_states, nb_substeps, cartesian_prod, cartesian_prod_cat, LP, LocErr2, LocErr_index,Lp_stay,sub_Bs, threshold, max_nb_states, nb_dims, do_preds, frame_len, TrMat, d2s, cur_d2s, intermediate_d2s, cur_nb_Bs, vary_LocErr_index)
    
    next_Cs = Cs[:,nb_locs-current_step]
    l2 = LocErr2[:,LocErr_index]
    
    disp = disps[-1]
    
    if anomalous:
        ### 5.48 µs
        anomalous_rectification(cur_Bs_cat, cur_Bs, gammas, betas, s2_arr, m_arr, previous_mus, next_Cs, disp, precomputed_disps, cur_d2s, LP, l2, cur_nb_Bs, current_step)
    
    if isBL:
        
        cur_nb_Bs = update_states_Leaving_FOV(cur_Bs, nb_states, nb_substeps, cartesian_prod, LP, m_arr, s2_arr, cur_nb_Bs, current_step) #update_states(cur_Bs, cur_Bs_cat, nb_states, nb_substeps, cartesian_prod, cartesian_prod_cat, LP, m_arr, s2_arr, previous_mus, cur_nb_Bs, current_step)
        
        # create a smaller equivalent of previous_mus so we can use the function repeat without repeating previous_mus which is useless
        get_Ts_from_Bs(LP, cur_Bs, TrMat, cur_nb_Bs)
        
        #LL = Lp_stay[np.argmax(np.all(cur_states[:,None] == sub_Bs[:,:,None],-1),1)] # pick the right proba of staying according to the current states
        #end_p_stay = p_stay[np.argmax(np.all(cur_states[:,None:,:-1] == sub_Bs[:,:,None],-1),1)]
        end_p_stay = np.empty(cur_nb_Bs)
        for k in range(cur_nb_Bs):
            end_p_stay[k] = p_stay[cur_Bs[k,0]]
        
        LP[:cur_nb_Bs] += cp.log(pBL + (1-end_p_stay) - pBL * (1-end_p_stay))
    
    last_integrale(LP, Cs, m_arr, s2_arr, LocErr2, nb_locs, current_step, LocErr_index, nb_dims, cur_nb_Bs)
    #print('CumLP', np.log(np.sum(np.exp(LP[:cur_nb_Bs] - np.max(LP[:cur_nb_Bs])))) + np.max(LP[:cur_nb_Bs]))
    
    #LF = cp.log(Fs[cur_Bs[:,:,0].astype(int)]) # Log proba of starting in a given state (fractions)
    #LF = cp.log(0.5)
    # cp.mean(cp.log(Fs[cur_Bs[:,:,:].astype(int)]), 2) # Log proba of starting in a given state (fractions)
    
    if do_preds:
        preds = compute_state_preds(LP[:cur_nb_Bs], cur_Bs_cat[:cur_nb_Bs], nb_locs, nb_states)
    else:
        preds = np.zeros((1, 1))
    # inverse the order so preds[0] corresponds to the first time point
    return LP, cur_Bs_cat, preds, cur_nb_Bs

if do_jit:
    P_Cs_inter_bound_stats_th = jit(P_Cs_inter_bound_stats_th, **args)

def compute_state_preds(LP, cur_Bs_cat, nb_locs, nb_states):
    preds = np.zeros((nb_locs, nb_states))
    pred_LP = LP
    nb_locs_minus1 = nb_locs - 1
    if np.max(LP)>600: # avoid overflow of exponentials, (drawback: mechanically also reduces the weights of longest tracks)
        pred_LP = LP - (np.max(LP)-600)
    
    P = np.exp(pred_LP)
    sum_P = np.sum(P)
    for k in range(nb_locs):
        for state in range(nb_states):
            preds[nb_locs_minus1 - k, state] = np.sum(P*cur_Bs_cat[:, k, state]) / sum_P
    return preds

if do_jit:
    compute_state_preds = jit(compute_state_preds, **args)

'''
the three following functions are used to compute the probablility density function
a Gaussian distribution, required for compute_P_leave (numba compatible)
'''
'''
@jit(**args)
def norm_cdf(xs):
    size = xs.shape[0]
    ys = np.empty(size)
    for i in range(size):
        ys[i] = norm_CDF(xs[i])
    return ys

@jit(**args)
def norm_CDF(x):
    inv_sqrt2 = 0.7071067811865475
    return (1.0 + erf(x * inv_sqrt2)) * 0.5

@jit(**args)
def erf(x):
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    
    # Save the sign of x
    sign = 1
    if x < 0:
        sign = -1
    x = abs(x)
    
    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
    
    return sign*y
'''

def norm_cdf(xs):
    inv_sqrt2 = 0.7071067811865475
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    
    size = xs.shape[0]
    ys = np.empty(size)
    for i in range(size):
        x=xs[i]
        x = x * inv_sqrt2
        
        # Save the sign of x
        sign = 1
        if x < 0:
            sign = -1
        x = abs(x)
        if x < 5:
            # A&S formula 7.1.26
            t = 1.0/(1.0 + p*x)
            y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
        else:
            y = 1
        
        ys[i] = (1.0 + sign*y) * 0.5
    return ys

if do_jit:
    norm_cdf = jit(norm_cdf, **args)


def compute_P_leave(cur_Bs, nb_substeps, nb_states, d2s, cell_dims, pBL, cur_nb_Bs):
    
    sub_Bs = get_all_Bs(nb_substeps + 1, nb_states)[:cur_Bs.shape[0]//nb_states,:nb_substeps] # list of possible current states we can meet to compute the proba of staying in the FOV
    nb_sub_ds = sub_Bs.shape[0]
    sub_ds = np.zeros(nb_sub_ds)
    for k in range(nb_sub_ds):
        sub_ds[k] = cp.mean(d2s[sub_Bs[k]])**0.5 # corresponding list of d
    
    cur_p_stay = np.empty(nb_sub_ds)
    
    presision = 400
    
    p_stay = np.ones(sub_ds.shape[-1])
    for cell_len in cell_dims:
        xs = np.linspace(0+cell_len/(2*presision),cell_len-cell_len/(2*presision),presision)
        for k in range(nb_sub_ds):
            #cur_p_stay[k] = ((cp.mean(scipy.stats.norm.cdf((cell_len-xs)/(sub_ds[k]+1e-200)) - scipy.stats.norm.cdf(-xs/(sub_ds[k]+1e-200)),0))) # proba to stay in the FOV for each of the possible cur Bs
            cur_p_stay[k] = cp.mean(norm_cdf((cell_len-xs)/(sub_ds[k]+1e-200)) - norm_cdf(-xs/(sub_ds[k]+1e-200))) # proba to stay in the FOV for each of the possible cur Bs
        p_stay = p_stay*cur_p_stay
    Lp_stay = cp.log(p_stay * (1-pBL)) # proba for the track to survive = both stay in the FOV and not bleach
    return Lp_stay, p_stay, sub_Bs

if do_jit:
    compute_P_leave = jit(compute_P_leave, **args)
'''
@jit(**args)
def repeat(m_arr, s2_arr, LP, previous_mus, nb_states, nb_substeps, cur_nb_Bs):
    new_m_arr = np.empty((cur_nb_Bs, m_arr.shape[1]))
    new_s2_arr = np.empty((cur_nb_Bs, s2_arr.shape[1]))
    nb_repeats = nb_states**nb_substeps
    new_LP = np.empty((cur_nb_Bs))
    new_previous_mus = np.empty((previous_mus.shape[0], cur_nb_Bs, previous_mus.shape[2]))
    
    for k in range(m_arr.shape[0]):
        new_m_arr[nb_repeats*k:nb_repeats*(k+1)] = m_arr[k:k+1]
        new_s2_arr[nb_repeats*k:nb_repeats*(k+1)] = s2_arr[k:k+1]
        new_LP[nb_repeats*k:nb_repeats*(k+1)] = LP[k:k+1]
        new_previous_mus[:, nb_repeats*k:nb_repeats*(k+1)] = previous_mus[:,k:k+1]
    return new_m_arr, new_s2_arr, new_LP, new_previous_mus
'''
def update_states(cur_Bs, cur_Bs_cat, nb_states, nb_substeps, cartesian_prod, cartesian_prod_cat, LP, m_arr, s2_arr, previous_mus, cur_nb_Bs, current_step):
    nb_repeats = cartesian_prod.shape[0]
    cur_Bs[:cur_nb_Bs, 1] = cur_Bs[:cur_nb_Bs, 0]
    
    for rep in range(1,nb_repeats):
        idx0 = cur_nb_Bs*rep
        idx1 = cur_nb_Bs*(rep+1)
        LP[idx0:idx1] = LP[:idx0]
        m_arr[idx0:idx1] = m_arr[:idx0]
        s2_arr[idx0:idx1] = s2_arr[:idx0]
        previous_mus[:current_step-1,idx0:idx1] = previous_mus[:current_step-1,:idx0]
        for k in range(cur_nb_Bs):            
            cur_Bs_cat[idx0:idx1,-current_step:] = cur_Bs_cat[:cur_nb_Bs,-current_step:]
        cur_Bs[idx0:idx1, 1] = cur_Bs[:cur_nb_Bs, 0]
    for state in range(nb_states):
        for k in range(cur_nb_Bs):
            cur_Bs_cat[k+state*cur_nb_Bs,-current_step-1] = cartesian_prod_cat[state]
            cur_Bs[k+state*cur_nb_Bs, 0] = state
    
    cur_nb_Bs = cur_nb_Bs*nb_repeats
    return cur_nb_Bs

if do_jit:
    update_states = jit(update_states, **args)

def update_states_Leaving_FOV(cur_Bs, nb_states, nb_substeps, cartesian_prod, LP, m_arr, s2_arr, cur_nb_Bs, current_step):
    
    nb_repeats = cartesian_prod.shape[0]
    #new_cur_Bs = np.empty((nb_repeats*cur_nb_Bs, cur_Bs.shape[1]+nb_substeps), dtype=cur_Bs.dtype)
    #new_cur_Bs_cat = np.empty((nb_repeats*cur_nb_Bs, cur_Bs_cat.shape[1]+nb_substeps, nb_states), dtype=cur_Bs_cat.dtype)
    
    cur_Bs[:cur_nb_Bs, 1] = cur_Bs[:cur_nb_Bs, 0]
    
    for rep in range(1,nb_repeats):
        idx0 = cur_nb_Bs*rep
        idx1 = cur_nb_Bs*(rep+1)
        LP[idx0:idx1] = LP[:idx0]
        m_arr[idx0:idx1] = m_arr[:idx0]
        s2_arr[idx0:idx1] = s2_arr[:idx0]
        cur_Bs[idx0:idx1, 1] = cur_Bs[:cur_nb_Bs, 0]
    for state in range(nb_states):
        for k in range(cur_nb_Bs):
            cur_Bs[k+state*cur_nb_Bs, 0] = state
    cur_nb_Bs = cur_nb_Bs*nb_repeats
    return cur_nb_Bs

if do_jit:
    update_states_Leaving_FOV = jit(update_states_Leaving_FOV, **args)

#np.log(np.sum(np.exp(LP[:cur_nb_Bs] - np.max(LP[:cur_nb_Bs])))) + np.max(LP[:cur_nb_Bs])
#cur_Bs_cat[7]
def reccurence_loop(Cs, disps, precomputed_disps, min_len, current_step, nb_locs, anomalous, m_arr, s2_arr, cur_Bs_cat, cur_Bs, gammas, betas, previous_mus, nb_states, nb_substeps, cartesian_prod, cartesian_prod_cat, LP, LocErr2, LocErr_index,Lp_stay,sub_Bs, threshold, max_nb_states, nb_dims, do_preds, frame_len, TrMat, d2s, cur_d2s, intermediate_d2s, cur_nb_Bs, vary_LocErr_index):
    #while current_step <= nb_locs-1:
    
    all_group_IDs = np.empty(max_nb_states)
    nb_repeats = cartesian_prod.shape[0]
    
    target_max_nb_Bs = max_nb_states // nb_repeats
    
    if not precomputed_disps:
        disp = np.array([0., 0.])
    
    debug = 0
    
    for current_step in range(current_step, nb_locs):
        
        next_Cs = Cs[:,nb_locs-current_step]
        l2 = LocErr2[:,LocErr_index]
        if precomputed_disps:
            disp = disps[nb_locs-current_step]
        
        if debug:
            print('current_step:', current_step)
        if anomalous:
            ### 5.48 µs
            anomalous_rectification(cur_Bs_cat, cur_Bs, gammas, betas, s2_arr, m_arr, previous_mus, next_Cs, disp, precomputed_disps, cur_d2s, LP, l2, cur_nb_Bs, current_step)
        
        # update cur_Bs to describe the states at the next step:
        # cur_Bs = get_all_Bs(current_step*nb_substeps+1 - removed_steps, nb_states)[None]
        # cur_Bs = all_Bs[:,:nb_states**(current_step + nb_substeps - removed_steps),:current_step + nb_substeps - removed_steps]
        cur_nb_Bs = update_states(cur_Bs, cur_Bs_cat, nb_states, nb_substeps, cartesian_prod, cartesian_prod_cat, LP, m_arr, s2_arr, previous_mus, cur_nb_Bs, current_step)
        #print('cur_Bs', cur_Bs[:cur_nb_Bs])
        ds_froms_states(cur_d2s, d2s, cur_Bs, cur_nb_Bs, intermediate_d2s) # update cur_d2s according to the current states
        # cur_d2s[:cur_nb_Bs]
        #save_LP = LP.copy()
        
        # cur_Bs[:cur_nb_Bs]
        # cur_Bs_cat[7+8]
        cur_Bs_cat[np.argmax(LP[:cur_nb_Bs])]
        m_arr[np.argmax(LP[:cur_nb_Bs])]
        m_arr[:cur_nb_Bs]
        
        get_Ts_from_Bs(LP, cur_Bs, TrMat, cur_nb_Bs)
        #print('LP+T', np.log(np.sum(np.exp(LP[:cur_nb_Bs] - np.max(LP[:cur_nb_Bs])))) + np.max(LP[:cur_nb_Bs]))
        if debug:
            print('LP + LT', (LP)[:cur_nb_Bs])
        # repeat the previous matrix to account for the states variations due to the new position
        
        #0.004 ms
        '''
        to do :
        '''
        #save_LP = LP.copy()
        if vary_LocErr_index:
            LocErr_index = nb_locs-current_step
        #log_integrale_dif(Cs[:,nb_locs-current_step], LocErr2[:,LocErr_index], cur_d2s, m_arr, s2_arr, LP, cur_nb_Bs)
        log_integrale_dif(next_Cs, l2, cur_d2s, m_arr, s2_arr, LP, cur_nb_Bs)
        #print('m_arr', m_arr[:cur_nb_Bs])
        #print('s2_arr', s2_arr[:cur_nb_Bs])
        if debug:
            print('LP + LC', (LP)[:cur_nb_Bs])
        #print('LP+motion', np.log(np.sum(np.exp(LP[:cur_nb_Bs] - np.max(LP[:cur_nb_Bs])))) + np.max(LP[:cur_nb_Bs]))
        #print('LC', (LP - save_LP)[:cur_nb_Bs])
        #save_LP = LP.copy()
        
        if current_step >= min_len:
            L_leave_FOV(LP, Lp_stay, cur_Bs, cur_nb_Bs, sub_Bs, nb_substeps) # pick the right proba of staying in the field of view according to the current states
        #print('LP+Leave', np.log(np.sum(np.exp(LP[:cur_nb_Bs] - np.max(LP[:cur_nb_Bs])))) + np.max(LP[:cur_nb_Bs]))
        #print('LL', (LP - save_LP)[:cur_nb_Bs])
        #print('LP', LP[:cur_nb_Bs])
        
        if debug:
            
            print('LP + LL', (LP)[:cur_nb_Bs])
            print('cur_nb_Bs:', cur_nb_Bs)
            print('CumLP', np.log(np.sum(np.exp(LP[:cur_nb_Bs] - np.max(LP[:cur_nb_Bs])))) + np.max(LP[:cur_nb_Bs]))
            
            mean_mu = np.sum(np.exp(LP[:cur_nb_Bs])[:,None] * m_arr[:cur_nb_Bs], 0) / np.sum(np.exp(LP[:cur_nb_Bs]))
            
            plt.scatter(mean_mu[0], mean_mu[1], color = cs[kkk], marker = 'x')
            argmax = np.argmax(LP[:cur_nb_Bs])
            plt.scatter(m_arr[argmax, 0], m_arr[argmax, 1], c = 'k', s=10.)
            
            Cs[:,nb_locs-current_step] = mean_mu
        
        '''idea : the position and the state 6 steps ago should not impact too much the 
        probability of the next position so the m_arr and s2_arr of tracks with the same 6 last 
        states must be very similar, we can then fuse the parameters of the pairs of Bs
        which vary only for the last step (7) and sum their probas'''
        
        cur_nb_Bs = fuse_tracks_th(m_arr,
                                    s2_arr,
                                    LP,
                                    cur_Bs,
                                    cur_Bs_cat,
                                    previous_mus,
                                    gammas,
                                    cur_nb_Bs,
                                    all_group_IDs,
                                    target_max_nb_Bs,
                                    nb_repeats,
                                    current_step,
                                    nb_states = nb_states,
                                    nb_dims = nb_dims,
                                    do_preds = do_preds,
                                    threshold = threshold,
                                    frame_len = frame_len) # threshold on values normalized by sigma.
        
        #print('LP+remove', np.log(np.sum(np.exp(LP[:cur_nb_Bs] - np.max(LP[:cur_nb_Bs])))) + np.max(LP[:cur_nb_Bs]))
        previous_mus[current_step-1,:cur_nb_Bs] = m_arr[:cur_nb_Bs]
    
    current_step += 1
    
    #print(cur_nb_Bs)
    return cur_nb_Bs, current_step

#gammas[1]=0.02
#ds[1]=0.01
'''
Cs = np.array([[[98.2926239 ,  0.50474765],
        [98.28677041,  0.52621683],
        [98.27543602,  0.50772219],
        [98.32650934,  0.48000563],
        [98.33619592,  0.55664411],
        [98.35380494,  0.58195414],
        [98.38450814,  0.5534428 ],
        [98.40312247,  0.57989092],
        [98.38714524,  0.53815048],
        [98.41561127,  0.54311691],
        [98.36731562,  0.58772104],
        [98.38525886,  0.6071252 ],
        [98.41314827,  0.61902968],
        [98.40467369,  0.60473141],
        [98.39100318,  0.66504958],
        [98.4312602 ,  0.66535726],
        [98.44087653,  0.72470322],
        [98.46071122,  0.69433139],
        [98.47476727,  0.75291761],
        [98.49197678,  0.73170668]]])[:, ::-1]

Cs = np.array([[[98.28688479,  0.50393597],
        [98.2870943 ,  0.50831824],
        [98.2972603 ,  0.50500459],
        [98.32864383,  0.51374305],
        [98.35596358,  0.56093724],
        [98.37707462,  0.56588958],
        [98.3864386 ,  0.55706981],
        [98.38817034,  0.55948402],
        [98.37935769,  0.54362416],
        [98.38261722,  0.56319362],
        [98.37131677,  0.59105515],
        [98.38560602,  0.60894554],
        [98.39579532,  0.62336082],
        [98.39447821,  0.63781714],
        [98.40308383,  0.67401105],
        [98.43263714,  0.69334474],
        [98.45637954,  0.72311413],
        [98.47194055,  0.7235101 ],
        [98.48206984,  0.7439171 ],
        [98.49197678,  0.73170668]]])

Cs2 = np.array([[[98.47458821,  0.726081  ],
        [98.45874336,  0.72075787],
        [98.44312923,  0.69024965],
        [98.42696951,  0.68636677],
        [98.41287355,  0.64833674],
        [98.3964911 ,  0.63186881],
        [98.40050713,  0.60227185],
        [98.39766819,  0.60087272],
        [98.3846471 ,  0.58348808],
        [98.38426848,  0.56506765],
        [98.39769008,  0.54886276],
        [98.38325003,  0.55335542],
        [98.38060115,  0.56619675],
        [98.35943134,  0.55350117],
        [98.33596195,  0.55307436],
        [98.3152919 ,  0.52312618],
        [98.29981483,  0.50047097],
        [98.28378503,  0.51272451],
        [98.28944739,  0.51639832],
        [98.2926239 ,  0.50474765]]])

Cs = (Cs + Cs2[:,::-1])/2

mus = np.cumsum(disps, 0)
mus = mus - np.mean(disps, 0, keepdims = True) + np.mean(Cs[0], 0, keepdims = True)

plt.figure()
plt.plot(Cs[0,:,0], Cs[0,:,1])
plt.scatter(Cs[0,:,0], Cs[0,:,1])
plt.scatter(Cs[0,0,0], Cs[0,0,1], marker = 'x', s = 40, c='r')
plt.gca().set_aspect('equal', adjustable='datalim')

plt.scatter(mus[:,0], mus[:,1], c='r')

'''


if do_jit:
    reccurence_loop = jit(reccurence_loop, **args)

def cartesian_jit(arrays):
    
    """
    Generate a cartesian product of input arrays.
    
    Parameters
    ----------
    arrays : list or tuple of arrays
        1-D arrays to form the cartesian product of.
    
    
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    
    """
    
    n = 1
    for x in arrays:
        n *= x.size
    out = np.zeros((n, len(arrays)))
    
    
    for i in range(len(arrays)):
        m = int(n / arrays[i].size)
        out[:n, i] = np.repeat(arrays[i], m)
        n //= arrays[i].size
    
    n = arrays[-1].size
    for k in range(len(arrays)-2, -1, -1):
        n *= arrays[k].size
        m = int(n / arrays[k].size)
        for j in range(1, arrays[k].size):
            out[j*m:(j+1)*m,k+1:] = out[0:m,k+1:]
    return out

if do_jit:
    cartesian_jit = jit(cartesian_jit, **args)

def to_Categorical(cur_Bs, nb_states):
    '''
    Turn arrays of sequences of states to categorical
    
    Parameters
    ----------
    cur_Bs : array (2D, axis 0 = sequences of states, axis 1 = time)
        array of sequences of states
    nb_states : Int,
        number of states
    
    Returns
    -------
    cur_Bs_cat : array (2D, axis 0 = sequences of states, axis 1 = time, axis 2 = categorical state)
        categorical array of sequences of states
    '''
    
    cur_Bs_cat = np.zeros((cur_Bs.shape[0], cur_Bs.shape[1], nb_states))
    for i in range(cur_Bs.shape[0]):
        for j in range(cur_Bs.shape[1]):
            cur_Bs_cat[i,j, cur_Bs[i,j]] = 1
    return cur_Bs_cat

if do_jit:
    to_Categorical = jit(to_Categorical, **args)

def L_leave_FOV(LP, Lp_stay, cur_Bs, cur_nb_Bs, sub_Bs, nb_substeps):
    for k in range(cur_nb_Bs):
        idx = -1
        is_idx = True
        while is_idx:
            idx += 1
            i = 0
            while i < nb_substeps:
                if cur_Bs[k, i] == sub_Bs[idx, i]:
                    i += 1
                else:
                    i = nb_substeps+1
            if i == nb_substeps:
                is_idx = False
        LP[k] += Lp_stay[idx]

if do_jit:
    L_leave_FOV = jit(L_leave_FOV, **args)

def group_states(m_arr, s2_arr, cur_Bs, LP, nb_states, threshold, all_group_IDs, cur_nb_Bs):
    s_arr = s2_arr[:cur_nb_Bs]**0.5
    
    LL_threshold = - np.log(1e-15)
    
    max_LP = np.max(LP[:cur_nb_Bs])
    
    LL_fuse = LP[:cur_nb_Bs] < max_LP - LL_threshold
    
    all_group_IDs[:] = -1 # reset the fusing group IDs to -1 for each iteration
    
    for state in range(nb_states):
        args = np.where(LL_fuse * (cur_Bs[:cur_nb_Bs,0]==state))[0]
        if len(args)>1:
            all_group_IDs[args] = cur_nb_Bs + state
    
    Remaining_IDs = np.arange(cur_nb_Bs)[all_group_IDs[:cur_nb_Bs]<0] # we may want to make this a unique array
    
    args = np.empty(Remaining_IDs.shape[0], dtype = np.int16)
    for ID in Remaining_IDs:
        if all_group_IDs[ID] == -1:
            
            cur_m_arr = m_arr[ID]
            cur_s_arr = s_arr[ID]
            cur_B = cur_Bs[ID,0]
            
            #Remaining_state_mask[:] = Remaining_cur_Bs[:,0] == -1
            args_idx = 0
            for ID2 in Remaining_IDs:
                if cur_Bs[ID2,0] == cur_B:
                    close_mu = (np.mean(np.abs(m_arr[ID2] - cur_m_arr)/s_arr[ID2])) < threshold
                    if close_mu:
                        close_sig = (np.mean(np.abs(s_arr[ID2] - cur_s_arr)/s_arr[ID2])) < threshold
                        if close_sig:
                            args[args_idx] = ID2
                            args_idx += 1
            
            for ID2 in args[:args_idx]:
                all_group_IDs[ID2] = ID
    
    final_group_IDs = np.unique(all_group_IDs[:cur_nb_Bs])
    nb_groups = final_group_IDs.shape[0]
    #print(all_group_IDs)
    return all_group_IDs

if do_jit:
    group_states = jit(group_states, **args)

def compute_fusion_means(final_group_IDs, all_group_IDs, nb_groups, cur_Bs, LP, m_arr, s2_arr, previous_mus, cur_Bs_cat, nb_states, current_step):
    for idx, Bs_ID in enumerate(final_group_IDs):
        group = np.where(all_group_IDs == Bs_ID)[0]
        if len(group)>1:
            max_LP = LP[group].max()
            weights = np.exp(LP[group] - max_LP)
            weights = weights / np.sum(weights)
            
            cur_Bs_cat[idx, -current_step-1:] = weights[0] *  cur_Bs_cat[group[0], -current_step-1:]
            m_arr[idx] = weights[0] * m_arr[group[0]]
            s2_arr[idx] = weights[0] * s2_arr[group[0]]
            previous_mus[:current_step-1,idx] = weights[0] * previous_mus[:current_step-1, group[0]]
            for k in range(1,len(weights)):
                cur_Bs_cat[idx, -current_step-1:] += weights[k] *  cur_Bs_cat[group[k], -current_step-1:]
                m_arr[idx] += weights[k] * m_arr[group[k]]
                s2_arr[idx] += weights[k] * s2_arr[group[k]]
                previous_mus[:current_step-1,idx] += weights[k] * previous_mus[:current_step-1, group[k]]
            LP[idx] = np.log(np.sum(np.exp(LP[group]-max_LP))) + max_LP # recompute the log likelihood avoiding under/overflowing
        else:
            cur_Bs_cat[idx] =  cur_Bs_cat[group[0]]
            m_arr[idx] = m_arr[group[0]]
            s2_arr[idx] = s2_arr[group[0]]
            LP[idx] = LP[group[0]]
            previous_mus[:,idx] = previous_mus[:, group[0]]
        cur_Bs[idx] = cur_Bs[group[0]] # by definition the group shares the same last state so we can only use the last state of the first element of the group

if do_jit:
    compute_fusion_means = jit(compute_fusion_means, **args)

"""
@jit(**args)
def second_group_states(all_group_IDs2, LP, m_arr, s2_arr, cur_Bs, nb_states, nb_dims, cur_nb_Bs, target_max_nb_Bs):
    '''
    Franc
    further group the sequences of low proba to keep the number of sequences lower than max_nb_states (cur_nb_Bs needs to be lower than cur_nb_Bs / nb_state**(nb_substeps-1))
    
    need to fix the case where a state only falls in the low_P_IDs
    
    '''
    
    s_arr = s2_arr[:cur_nb_Bs]**0.5
    
    args = np.argsort(LP[:cur_nb_Bs])
    min_LP = LP[args[cur_nb_Bs-target_max_nb_Bs]]
    low_P_IDs = np.where(LP[:cur_nb_Bs] < min_LP)[0]
    high_P_IDs = np.where(LP[:cur_nb_Bs] >= min_LP)[0]
    
    
    all_group_IDs2[:] = -1
    all_group_IDs2[high_P_IDs] = high_P_IDs
    
    for i, low_P_ID in enumerate(low_P_IDs):
        min_dist = np.inf
        cur_B = cur_Bs[low_P_ID, 0]
        pos = m_arr[low_P_ID]
        std = s_arr[low_P_ID]
        for high_P_ID in high_P_IDs:
            if cur_B == cur_Bs[high_P_ID, 0]:
                print(cur_Bs[high_P_ID])
                is_smaller = True
                pos2 = m_arr[high_P_ID]
                cur_dist = 0
                dim = 0
                while dim < nb_dims:
                    cur_dist += np.abs(pos[dim] - pos2[dim])
                    dim += 1
                    if cur_dist > min_dist:
                        dim += nb_dims
                        is_smaller = False
                if is_smaller:
                    std2 = s_arr[high_P_ID]
                    cur_dist += np.sum(np.abs(std - std2))
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_ID = high_P_ID
        all_group_IDs2[low_P_ID] = closest_ID
    
    return all_group_IDs2, target_max_nb_Bs
"""

def second_group_states(all_group_IDs, LP, m_arr, s2_arr, cur_Bs, nb_states, nb_dims, cur_nb_Bs, target_max_nb_Bs):
    '''
    Franc
    further group the sequences of low proba to keep the number of sequences lower than max_nb_states (cur_nb_Bs needs to be lower than cur_nb_Bs / nb_state**(nb_substeps-1))
    
    need to fix the case where a state only falls in the low_P_IDs
    
    '''
    
    s_arr = s2_arr[:cur_nb_Bs]**0.5
    
    args = np.argsort(LP[:cur_nb_Bs])
    min_LP = LP[args[cur_nb_Bs-target_max_nb_Bs]]
    low_P_IDs = np.where(LP[:cur_nb_Bs] < min_LP)[0]
    high_P_IDs = np.where(LP[:cur_nb_Bs] >= min_LP)[0]
    
    all_group_IDs[:] = -1
    all_group_IDs[high_P_IDs] = high_P_IDs
    #all_group_IDs[:target_max_nb_Bs] = high_P_IDs
    
    for i, low_P_ID in enumerate(low_P_IDs):
        min_dist = np.inf
        cur_B = cur_Bs[low_P_ID, 0]
        pos = m_arr[low_P_ID]
        std = s_arr[low_P_ID]
        for high_P_ID in high_P_IDs:
            if cur_B == cur_Bs[high_P_ID, 0]:
                is_smaller = True
                pos2 = m_arr[high_P_ID]
                cur_dist = 0
                dim = 0
                while dim < nb_dims:
                    cur_dist += np.abs(pos[dim] - pos2[dim])
                    dim += 1
                    if cur_dist > min_dist:
                        dim += nb_dims
                        is_smaller = False
                if is_smaller:
                    std2 = s_arr[high_P_ID]
                    cur_dist += np.sum(np.abs(std - std2))
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_ID = high_P_ID
        all_group_IDs[low_P_ID] = closest_ID
    
    return all_group_IDs, target_max_nb_Bs

if do_jit:
    second_group_states = jit(second_group_states, **args)

'''
for ID in high_P_IDs:
    print(cur_Bs[:cur_nb_Bs][all_group_IDs2[:cur_nb_Bs] == ID])

np.where(all_group_IDs2[:cur_nb_Bs] == 3)
cur_Bs[4]

cur_Bs[high_P_IDs]
cur_Bs[low_P_IDs]

'''
def fuse_tracks_th(m_arr, s2_arr, LP, cur_Bs, cur_Bs_cat, previous_mus, gammas, cur_nb_Bs, all_group_IDs, target_max_nb_Bs, nb_repeats, current_step, nb_states = 2, nb_dims = 2, do_preds = 1, threshold = 0.2, frame_len = 6):
    '''
    The probabilities of the pairs of tracks must be added
    I chose to define the updated m_arr and s2_arr as the weighted average (of the variance for s2_arr)
    but other methods may be better
    As I must divid by a sum of exponentials which can be equal to zero because of underflow
    I correct the values in the exponetial to keep the maximal exp value at 0
    '''
    ###### 0.02 ms/iter
    # cut the matrixes so the resulting matrices only vary for their last state
    
    all_group_IDs = group_states(m_arr, s2_arr, cur_Bs, LP, nb_states, threshold, all_group_IDs, cur_nb_Bs)
    #group_states(m_arr, s2_arr, cur_Bs, LP, nb_states, threshold, all_group_IDs, cur_nb_Bs)
    #cur_Bs[:cur_nb_Bs]
    ###### 0.015 ms/iter
    final_group_IDs = np.unique(all_group_IDs[:cur_nb_Bs])
    # the grouped sequences of states need to share the same current state to be equivalent. To take this into account we split the groups according to this property
    # tracks with a unique state also need to not be merged for the anomalous diffusion analysis
    
    # part that is too time consuming, find a way to make it faster
    nb_groups = final_group_IDs.shape[0]
    '''
    while nb_repeats * nb_groups > max_nb_states: # if we have not reduced the number of sequences of states enough to fit the max_nb_states limitation, we increase the threshold until it works
        threshold *= 1.2 
        all_group_IDs = group_states(m_arr, s2_arr, cur_Bs, LP, nb_states, threshold, all_group_IDs, cur_nb_Bs)
        final_group_IDs = np.unique(all_group_IDs[:cur_nb_Bs])
        nb_groups = final_group_IDs.shape[0]
    '''
    
    ###### 0.038 ms/iter
    compute_fusion_means(final_group_IDs, all_group_IDs, nb_groups, cur_Bs, LP, m_arr, s2_arr, previous_mus, cur_Bs_cat, nb_states, current_step)
    cur_nb_Bs = nb_groups
    LP[:cur_nb_Bs]
    
    if cur_nb_Bs > target_max_nb_Bs:
        all_group_IDs2, cur_nb_Bs2 = second_group_states(all_group_IDs.copy(), LP, m_arr, s2_arr, cur_Bs, nb_states, nb_dims, cur_nb_Bs, target_max_nb_Bs)
        final_group_IDs = np.unique(all_group_IDs2[:cur_nb_Bs])
        compute_fusion_means(final_group_IDs, all_group_IDs2, nb_groups, cur_Bs, LP, m_arr, s2_arr, previous_mus, cur_Bs_cat, nb_states, current_step)
        cur_nb_Bs = target_max_nb_Bs
    
    return cur_nb_Bs

if do_jit:
    fuse_tracks_th = jit(fuse_tracks_th, **args)

#cur_Bs[:cur_nb_Bs2]

def get_all_Bs(nb_Cs=2, nb_states=3):
    '''
    produces a matrix of the possible sequences of states
    '''
    Bs_ID = np.arange(nb_states**nb_Cs)
    all_Bs = np.zeros((nb_states**nb_Cs, nb_Cs), dtype = np.int8)
    
    for k in range(all_Bs.shape[1]):
        cur_row = np.mod(Bs_ID,nb_states**(k+1))
        Bs_ID = (Bs_ID - cur_row)
        all_Bs[:,k] = cur_row // nb_states**k
    return all_Bs

if do_jit:
    get_all_Bs = jit(get_all_Bs, **args)

def get_Ts_from_Bs(LP, all_Bs, TrMat, cur_nb_Bs):
    '''
    compute the probability of the sequences of states according to the markov transition model
    '''
    LogTrMat = np.log(TrMat)
    # change from binary base 10 numbers to identify the consecutive states (from ternary if 3 states) 
    for k in range(cur_nb_Bs):
        cur_B = all_Bs[k]
        for j in range(all_Bs.shape[1]-1):
            LP[k] += LogTrMat[cur_B[j], cur_B[j+1]]

if do_jit:
    get_Ts_from_Bs = jit(get_Ts_from_Bs, **args)

def Proba_Cs(Cs, disps, precomputed_disps, LocErr2, ds, Fs, TrMat, nb_locs, nb_dims, nb_states, gammas, betas, pBL, isBL, cell_dims, nb_substeps, frame_len, min_len, threshold, max_nb_states, cartesian_prod, cartesian_prod_cat):
    '''
    inputs the observed localizations and determine the probability of 
    observing these data knowing the localization error, D the diffusion coef,
    pu the proba of unbinding per step and pb the proba of binding per step
    sum the proba of Cs inter Bs (calculated with P_Cs_inter_bound_stats)
    over all Bs to get the proba of Cs (knowing the initial position c0)
    '''
    
    LP_CB, _, _, cur_nb_Bs = P_Cs_inter_bound_stats_th(Cs = Cs,
                              LocErr2 = LocErr2,
                              ds = ds,
                              Fs = Fs,
                              TrMat = TrMat,
                              nb_locs = nb_locs,
                              nb_dims = nb_dims, 
                              nb_states = nb_states,
                              cartesian_prod = cartesian_prod,
                              cartesian_prod_cat = cartesian_prod_cat, 
                              disps = disps,
                              precomputed_disps = precomputed_disps,
                              gammas = gammas, 
                              betas = betas, 
                              pBL = pBL, 
                              isBL = isBL, 
                              cell_dims = cell_dims, 
                              nb_substeps = nb_substeps, 
                              frame_len = frame_len, 
                              do_preds = 0, 
                              min_len = min_len, 
                              threshold = threshold, 
                              max_nb_states = max_nb_states)
    
    #Cs, LocErr2, ds, Fs, TrMat, nb_locs, nb_dims, nb_states, cartesian_prod, cartesian_prod_cat, gammas, betas, pBL, isBL, cell_dims, nb_substeps, frame_len, do_preds, min_len, threshold, max_nb_states
    # calculates P(C) the sum of P(C inter B) for each track
    max_LP = np.max(LP_CB[:cur_nb_Bs])
    
    LP_C = np.log(np.sum(np.exp(LP_CB[:cur_nb_Bs] - max_LP))) + max_LP
    
    return LP_C #scalar

if do_jit:
    Proba_Cs = jit(Proba_Cs, **args)

'''

plt.figure()
plt.plot(Cs[0,:,0], Cs[0,:,1])
plt.scatter(Cs[0,:,0], Cs[0,:,1])
plt.gca().set_aspect('equal', adjustable='datalim')


plt.plot(disps[:,0], disps[:,1])

'''

def prange_fit_LocErr(all_tracks, all_disps, precomputed_disps, nb_tracks, LocErr2, ds, Fs, TrMat, all_nb_locs, nb_dims, nb_states, gammas, betas, pBL, isBLs, cell_dims, nb_substeps, frame_len, min_len, threshold, max_nb_states, cartesian_prod, cartesian_prod_cat):
    LPs = np.empty(nb_tracks)
    #if not precomputed_disps:
    #    disps = all_disps[0]
    #LPs = []
    for k in prange(nb_tracks):
        #LPs[k] = Proba_Cs(all_tracks[k], LocErr2, ds, Fs, TrMat, all_nb_locs[k], nb_dims, nb_states, gammas, betas, pBL, isBLs[k], cell_dims, nb_substeps, frame_len, min_len, threshold, LL_threshold, max_nb_states, cartesian_prod, cartesian_prod_cat)
        Cs = all_tracks[k]
        nb_locs = all_nb_locs[k]
        isBL = isBLs[k]
        if precomputed_disps:
            disps = all_disps[k]
        else:
            disps = Cs[0]
        LPs[k] = Proba_Cs(Cs, disps, precomputed_disps, LocErr2, ds, Fs, TrMat, nb_locs, nb_dims, nb_states, gammas, betas, pBL, isBL, cell_dims, nb_substeps, frame_len, min_len, threshold, max_nb_states, cartesian_prod, cartesian_prod_cat)
        #LPs.append(Proba_Cs(all_tracks[k], LocErr2, ds, Fs, TrMat, all_nb_locs[k], nb_dims, nb_states, gammas, betas, pBL, isBLs[k], cell_dims, nb_substeps, frame_len, min_len, threshold, max_nb_states, cartesian_prod, cartesian_prod_cat)
    return LPs

if do_jit:
    prange_fit_LocErr = jit(prange_fit_LocErr, nopython = True, parallel=True, cache = True, nogil = True)

def prange_input_LocErr(all_tracks, all_disps, precomputed_disps, nb_tracks, input_LocErr2, ds, Fs, TrMat, all_nb_locs, nb_dims, nb_states, gammas, betas, pBL, isBLs, cell_dims, nb_substeps, frame_len, min_len, threshold, max_nb_states, cartesian_prod, cartesian_prod_cat):
    LPs = np.empty(nb_tracks)
    #LPs = []
    for k in prange(nb_tracks):
        #LPs[k] = Proba_Cs(all_tracks[k], LocErr2, ds, Fs, TrMat, all_nb_locs[k], nb_dims, nb_states, gammas, betas, pBL, isBLs[k], cell_dims, nb_substeps, frame_len, min_len, threshold, LL_threshold, max_nb_states, cartesian_prod, cartesian_prod_cat)
        Cs = all_tracks[k]
        disps = all_disps[k]
        nb_locs = all_nb_locs[k]
        isBL = isBLs[k]
        LocErr2 = input_LocErr2[k]
        LPs[k] = Proba_Cs(Cs, disps, precomputed_disps, LocErr2, ds, Fs, TrMat, nb_locs, nb_dims, nb_states, gammas, betas, pBL, isBL, cell_dims, nb_substeps, frame_len, min_len, threshold, max_nb_states, cartesian_prod, cartesian_prod_cat)
        #LPs.append(Proba_Cs(all_tracks[k], LocErr2, ds, Fs, TrMat, all_nb_locs[k], nb_dims, nb_states, gammas, betas, pBL, isBLs[k], cell_dims, nb_substeps, frame_len, min_len, threshold, max_nb_states, cartesian_prod, cartesian_prod_cat)
    return LPs

if do_jit:
    prange_input_LocErr = jit(prange_input_LocErr, nopython = True, parallel=True, cache = True, nogil = True)

'''
%time LP = prange_fit_LocErr(all_tracks, nb_tracks, LocErr2, ds, Fs, TrMat, all_nb_locs, nb_dims, nb_states, gammas, betas, pBL, isBLs, cell_dims, nb_substeps, frame_len, min_len, threshold, max_nb_states, cartesian_prod, cartesian_prod_cat)


prange_fit_LocErr_par = njit(parallel=True, cache = True, nogil = True)(prange_fit_LocErr)

%time LP = prange_fit_LocErr_par(all_tracks, nb_tracks, LocErr2, ds, Fs, TrMat, all_nb_locs, nb_dims, nb_states, gammas, betas, pBL, isBLs, cell_dims, nb_substeps, frame_len, min_len, threshold, max_nb_states, cartesian_prod, cartesian_prod_cat)

%timeit LP = prange_fit_LocErr_par(all_tracks, nb_tracks, LocErr2, ds, Fs, TrMat, all_nb_locs, nb_dims, nb_states, gammas, betas, pBL, isBLs, cell_dims, nb_substeps, frame_len, min_len, threshold, max_nb_states, cartesian_prod, cartesian_prod_cat)
'''

def extract_params(params, dt, nb_states, nb_substeps, input_LocErr2 = None, Matrix_type = 1):
    '''
    turn the parameters which differ deppending on the number of states into lists
    ds (diffusion lengths), Fs (fractions), TrMat (substep transiton matrix)
    '''
    param_names = np.sort(list(params.keys()))
    
    LocErr2 = []
    for param in param_names:
        if param.startswith('LocErr'):
            LocErr2.append(params[param].value**2)
    
    LocErr2 = np.array(LocErr2)[None, None]
    if input_LocErr2 != None:
        LocErr2 = []
        if np.any(np.array(list(params.keys())) == 'slope_LocErr'):
            for l in range(len(input_LocErr2)):
                LocErr2.append(np.clip(input_LocErr2[l] * params['slope_LocErr'].value + params['offset_LocErr'].value, 0.000001, np.inf))
        else:
            LocErr2 = input_LocErr2
    Ds = []
    Fs = []
    for param in param_names:
        if param.startswith('D') and len(param)<3:
            Ds.append(params[param].value)
        elif param.startswith('F'):
            Fs.append(params[param].value)
    Ds = np.array(Ds)
    Fs = np.array(Fs)
    TrMat = np.zeros((len(Ds),len(Ds)))
    for param in params:
        if param == 'pBL':
            pBL = params[param].value
        elif param.startswith('p'):
            i = int(param[1])
            j = int(param[2])
            TrMat[i,j] = params[param].value
    
    TrMat = TrMat/nb_substeps
    
    if Matrix_type == 0:
        TrMat[np.arange(len(Ds)), np.arange(len(Ds))] = 1-np.sum(TrMat,1)
    if Matrix_type == 1: # 1 - exp(-)
        TrMat = 1 - np.exp(-TrMat)
        TrMat[np.arange(len(Ds)), np.arange(len(Ds))] = 1-np.sum(TrMat,1)
    elif Matrix_type == 2:
        TrMat[np.arange(len(Ds)), np.arange(len(Ds))] = -np.sum(TrMat,1)
        TrMat = linalg.expm(TrMat)
    elif Matrix_type == 3:
        TrMat[np.arange(len(Ds)), np.arange(len(Ds))] = 0
        G = np.copy(TrMat)
        TrMat[np.arange(len(Ds)), np.arange(len(Ds))] = 1-np.sum(TrMat,1)
        G[np.arange(len(Ds)), np.arange(len(Ds))] = -np.sum(G,1)
        TrMatG = linalg.expm(G)
        TrMat = np.mean([TrMat, TrMatG], axis = 0)
    elif Matrix_type == 4:
        TrMat[np.arange(len(Ds)), np.arange(len(Ds))] = 0
        G = np.copy(TrMat)
        TrMat[np.arange(len(Ds)), np.arange(len(Ds))] = 1-np.sum(TrMat,1)
        G[np.arange(len(Ds)), np.arange(len(Ds))] = -np.sum(G,1)
        TrMatG = linalg.expm(G)
        TrMat = (TrMat* TrMatG)**0.5  
    
    gammas = np.zeros(nb_states)
    betas = np.zeros(nb_states)
    if 'gamma0' in list(params.keys()):
        try:
            for state in range(nb_states):
                gammas[state] = params['gamma'+str(state)]
                betas[state] = params['beta'+str(state)]
        except:
            raise ValueError('gamma and beta parameters must be either completely absent or present for each parameters, their values can be set fixed to 0 if necessary.')
    
    ds = np.sqrt(2*Ds*dt)
    return LocErr2, ds, Fs, TrMat, gammas, betas, pBL

def generate_params(nb_states = 3,
                    LocErr_type = 1,
                    nb_dims = 3, # only matters if LocErr_type == 2,
                    LocErr_bounds = [0.005, 0.1], # the initial guess on LocErr will be the geometric mean of the boundaries
                    D_max = 10, # maximal diffusion coefficient allowed
                    estimated_LocErr = None,
                    estimated_Ds = None, # D will be arbitrary spaced from 0 to D_max if None, otherwise input 1D array/list of Ds for each state from state 0 to nb_states - 1.
                    estimated_Fs = None, # fractions will be equal if None, otherwise input 1D array/list of fractions for each state from state 0 to nb_states - 1.
                    estimated_transition_rates = 0.1, # transition rate per step. [0.1,0.05,0.03,0.07,0.2,0.2]
                    slope_offsets_estimates = None, # need to specify the list [slop, offset] if LocErr_type = 4,
                    Fractions_bounds = [0.0001, 0.9999],
                    rate_boundaries = [0.00001, 1],
                    gammas = None,
                    vary_gammas = None,
                    gamma_boundaries = [-1,1],
                    betas = None,
                    vary_betas = None,
                    beta_boundaries = [0,2]):
    
    '''
    nb_states: number of states of the model.
    LocErr_type: 1 for a single localization error parameter,
                 2 for a localization error parameter for each dimension,
                 3 for a shared localization error for x and y dims (the 2 first dimensions) and another for z dim.
                 4 for an affine relationship between localization error a peak-wise input specified with input_LocErr (like an estimate of localization error/quality of peak/signal to noise ratio, etc).
                 None for no localization error fits, localization error is then directly assumed from a prior peak-wise estimate of localization error specified in input_LocErr.
    '''
    param_kwargs = []
    if estimated_Ds == None:
        if nb_states == 1:
            param_kwargs.append({'name' : 'D0', 'value' : 0.2 * D_max, 'min' : 0, 'max' : D_max, 'vary' : True})
        else:
            for s in range(nb_states):
                param_kwargs.append({'name' : 'D'+str(s), 'value' : 0.5*s**2 * D_max / (nb_states-1)**2, 'min' : 0, 'max' : D_max, 'vary' : True})
    else:
        for s in range(nb_states):
            param_kwargs.append({'name' : 'D'+str(s), 'value' : estimated_Ds[s], 'min' : 0, 'max' : D_max, 'vary' : True})
    if estimated_LocErr == None:
        if LocErr_type == 1:
            param_kwargs.append({'name' : 'LocErr', 'value' : (LocErr_bounds[0] * LocErr_bounds[1])**0.5, 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})
        elif LocErr_type == 2:
            for d in range(nb_dims):
                param_kwargs.append({'name' : 'LocErr' + str(d), 'value' : (LocErr_bounds[0] * LocErr_bounds[1])**0.5, 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})
        elif LocErr_type == 3:
            param_kwargs.append({'name' : 'LocErr0', 'value' : (LocErr_bounds[0] * LocErr_bounds[1])**0.5, 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})
            param_kwargs.append({'name' : 'LocErr1', 'expr' : 'LocErr0'})
            param_kwargs.append({'name' : 'LocErr2', 'value' : (LocErr_bounds[0] * LocErr_bounds[1])**0.5, 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})
    else:
        if LocErr_type == 1:
            param_kwargs.append({'name' : 'LocErr', 'value' : estimated_LocErr[0], 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})
        elif LocErr_type == 2:
            for d in range(nb_dims):
                param_kwargs.append({'name' : 'LocErr' + str(d), 'value' : estimated_LocErr[d], 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})
        elif LocErr_type == 3:
            param_kwargs.append({'name' : 'LocErr0', 'value' : estimated_LocErr[0], 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})
            param_kwargs.append({'name' : 'LocErr1', 'expr' : 'LocErr0'})
            param_kwargs.append({'name' : 'LocErr2', 'value' : estimated_LocErr[-1], 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})
    
    if LocErr_type == 4:
        param_kwargs.append({'name' : 'slope_LocErr', 'value' : slope_offsets_estimates[0], 'min' : 0, 'max' : 100, 'vary' : True})
        param_kwargs.append({'name' : 'offset_LocErr', 'value' : slope_offsets_estimates[1], 'min' : -100, 'max' : 100, 'vary' : True})
    
    F_expr = '1' 
    if estimated_Fs == None:
        for s in range(nb_states-1):
            param_kwargs.append({'name' : 'F'+str(s), 'value' : 1/nb_states, 'min' : Fractions_bounds[0], 'max' : Fractions_bounds[1], 'vary' : True})
            F_expr +=  ' - F'+str(s)
    else:
        for s in range(nb_states-1):
            param_kwargs.append({'name' : 'F'+str(s), 'value' : estimated_Fs[s], 'min' : Fractions_bounds[0], 'max' : Fractions_bounds[1], 'vary' : True})
            F_expr +=  ' - F'+str(s)
    param_kwargs.append({'name' : 'F'+str(nb_states-1), 'expr' : F_expr})
    
    if vary_gammas == None:
        vary_gammas = []
        for state in range(nb_states):
            vary_gammas.append('True')
    
    if gammas == None:
        gammas = []
        for state in range(nb_states):
            gammas.append(0)
    
    if vary_betas == None:
        vary_betas = []
        for state in range(nb_states):
            vary_betas.append('True')
    
    if betas == None:
        betas = []
        for state in range(nb_states):
            betas.append(0.3)
    
    for state in range(nb_states):
        param_kwargs.append({'name' : 'gamma'+str(state), 'value' : gammas[state], 'min': gamma_boundaries[0], 'max': gamma_boundaries[1], 'vary' : vary_gammas[state]})
        param_kwargs.append({'name' : 'beta'+str(state), 'value' : betas[state], 'min': beta_boundaries[0], 'max': beta_boundaries[1], 'vary' : vary_betas[state]})
    
    if not (type(estimated_transition_rates) == np.ndarray or type(estimated_transition_rates) == list):
        estimated_transition_rates = [estimated_transition_rates] * (nb_states * (nb_states-1))
    idx = 0
    for i in range(nb_states):
        for j in range(nb_states):
            if i != j:
                param_kwargs.append({'name' : 'p'+ str(i) + str(j), 'value' : estimated_transition_rates[idx], 'min' : rate_boundaries[0], 'max' : rate_boundaries[1], 'vary' : True})
                idx += 1
    param_kwargs.append({'name' : 'pBL', 'value' : 0.1, 'min' : rate_boundaries[0], 'max' : 1, 'vary' : True})
    
    params = Parameters()
    [params.add(**param_kwargs[k]) for k in range(len(param_kwargs))]
    
    return params

"""
def generate_params(nb_states = 3,
                    dt = 0.02,
                    LocErr_type = 1,
                    nb_dims = 3, # only matters if LocErr_type == 2,
                    LocErr_bounds = [0.005, 0.1], # the initial guess on LocErr will be the geometric mean of the boundaries
                    D_max = 10, # maximal diffusion coefficient allowed
                    estimated_LocErr = None,
                    estimated_Ds = None, # D will be arbitrary spaced from 0 to D_max if None, otherwise input 1D array/list of Ds for each state from state 0 to nb_states - 1.
                    estimated_Fs = None, # fractions will be equal if None, otherwise input 1D array/list of fractions for each state from state 0 to nb_states - 1.
                    estimated_transition_rates = 0.1, # transition rate per step. [0.1,0.05,0.03,0.07,0.2,0.2]
                    slope_offsets_estimates = None, # need to specify the list [slop, offset] if LocErr_type = 4,
                    Fractions_bounds = [0.0001, 0.9999],
                    rate_boundaries = [0.00001, 1],
                    gammas = None,
                    vary_gammas = None,
                    gamma_boundaries = [-1,1],
                    betas = None,
                    vary_betas = None,
                    beta_boundaries = [0,2]):
    
    '''
    nb_states: number of states of the model.
    LocErr_type: 1 for a single localization error parameter,
                 2 for a localization error parameter for each dimension,
                 3 for a shared localization error for x and y dims (the 2 first dimensions) and another for z dim.
                 4 for an affine relationship between localization error a peak-wise input specified with input_LocErr (like an estimate of localization error/quality of peak/signal to noise ratio, etc).
                 None for no localization error fits, localization error is then directly assumed from a prior peak-wise estimate of localization error specified in input_LocErr.
    '''
    
    if np.any(vary_gammas == None):
        vary_gammas = []
        for state in range(nb_states):
            vary_gammas.append('True')
    
    if np.any(gammas == None):
        gammas = []
        for state in range(nb_states):
            gammas.append(0.)
    
    if np.any(vary_betas == None):
        vary_betas = []
        for state in range(nb_states):
            vary_betas.append('True')
    np.any(True)
    if np.any(betas == None):
        betas = []
        for state in range(nb_states):
            betas.append(0.2)
    
    if np.any(estimated_Ds == None):
        estimated_Ds = []
        for s in range(nb_states):
            estimated_Ds.append(0.5*s**2 * D_max / (nb_states-1)**2)
    
    param_kwargs = []
    param_kwargs.append({'name' : 'dt', 'value': dt, 'min' : 0, 'max' : dt, 'vary' : False})
    
    for s in range(nb_states):
        cur_d_gamma = (2*estimated_Ds[s]*dt)**0.5 + gammas[s]
        if  gammas[s]!=0:
            cur_alpha = gammas[s] / cur_d_gamma
        else:
            cur_alpha = 0.
        # d+gamma could in principle have negative values to model ocilatory behaviors it is not very 
        param_kwargs.append({'name' : 'd_gamma'+str(s), 'value' : cur_d_gamma, 'min' : 0, 'max' : (2*D_max*dt)**0.5, 'vary' : True})
        param_kwargs.append({'name' : 'alpha'+str(s), 'value' : cur_alpha, 'min' : - 1, 'max' : 1, 'vary' : True})
        param_kwargs.append({'name' : 'gamma'+str(s), 'expr' : 'd_gamma'+str(s)+ '*alpha'+str(s) })
        param_kwargs.append({'name' : 'd'+str(s), 'expr' : 'd_gamma'+str(s)+ '*(1-alpha'+str(s) + ')' })
        param_kwargs.append({'name' : 'D'+str(s), 'expr' : 'd'+str(s) + '**2/(2*dt)' })
        param_kwargs.append({'name' : 'beta'+str(s), 'value' : betas[s], 'min': beta_boundaries[0], 'max': beta_boundaries[1], 'vary' : vary_betas[s]})
    
    if estimated_LocErr == None:
        if LocErr_type == 1:
            param_kwargs.append({'name' : 'LocErr', 'value' : (LocErr_bounds[0] * LocErr_bounds[1])**0.5, 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})
        elif LocErr_type == 2:
            for d in range(nb_dims):
                param_kwargs.append({'name' : 'LocErr' + str(d), 'value' : (LocErr_bounds[0] * LocErr_bounds[1])**0.5, 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})
        elif LocErr_type == 3:
            param_kwargs.append({'name' : 'LocErr0', 'value' : (LocErr_bounds[0] * LocErr_bounds[1])**0.5, 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})
            param_kwargs.append({'name' : 'LocErr1', 'expr' : 'LocErr0'})
            param_kwargs.append({'name' : 'LocErr2', 'value' : (LocErr_bounds[0] * LocErr_bounds[1])**0.5, 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})
    else:
        if LocErr_type == 1:
            param_kwargs.append({'name' : 'LocErr', 'value' : estimated_LocErr[0], 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})
        elif LocErr_type == 2:
            for d in range(nb_dims):
                param_kwargs.append({'name' : 'LocErr' + str(d), 'value' : estimated_LocErr[d], 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})
        elif LocErr_type == 3:
            param_kwargs.append({'name' : 'LocErr0', 'value' : estimated_LocErr[0], 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})
            param_kwargs.append({'name' : 'LocErr1', 'expr' : 'LocErr0'})
            param_kwargs.append({'name' : 'LocErr2', 'value' : estimated_LocErr[-1], 'min' : LocErr_bounds[0], 'max' : LocErr_bounds[1], 'vary' : True})
    
    if LocErr_type == 4:
        param_kwargs.append({'name' : 'slope_LocErr', 'value' : slope_offsets_estimates[0], 'min' : 0, 'max' : 100, 'vary' : True})
        param_kwargs.append({'name' : 'offset_LocErr', 'value' : slope_offsets_estimates[1], 'min' : -100, 'max' : 100, 'vary' : True})
    
    F_expr = '1' 
    if estimated_Fs == None:
        for s in range(nb_states-1):
            param_kwargs.append({'name' : 'F'+str(s), 'value' : 1/nb_states, 'min' : Fractions_bounds[0], 'max' : Fractions_bounds[1], 'vary' : True})
            F_expr +=  ' - F'+str(s)
    else:
        for s in range(nb_states-1):
            param_kwargs.append({'name' : 'F'+str(s), 'value' : estimated_Fs[s], 'min' : Fractions_bounds[0], 'max' : Fractions_bounds[1], 'vary' : True})
            F_expr +=  ' - F'+str(s)
    param_kwargs.append({'name' : 'F'+str(nb_states-1), 'expr' : F_expr})
    
    if not (type(estimated_transition_rates) == np.ndarray or type(estimated_transition_rates) == list):
        estimated_transition_rates = [estimated_transition_rates] * (nb_states * (nb_states-1))
    idx = 0
    for i in range(nb_states):
        for j in range(nb_states):
            if i != j:
                param_kwargs.append({'name' : 'p'+ str(i) + str(j), 'value' : estimated_transition_rates[idx], 'min' : rate_boundaries[0], 'max' : rate_boundaries[1], 'vary' : True})
                idx += 1
    param_kwargs.append({'name' : 'pBL', 'value' : 0.1, 'min' : rate_boundaries[0], 'max' : 1, 'vary' : True})
    
    params = Parameters()
    [params.add(**param_kwargs[k]) for k in range(len(param_kwargs))]
    
    return params
"""

def cum_Proba_Cs(params, all_tracks, all_disps, dt, cell_dims, input_LocErr2, nb_states, min_len, cartesian_prod, cartesian_prod_cat, nb_substeps, frame_len, isBLs, all_nb_locs, nb_dims, verbose = 1, Matrix_type = 1, threshold = 0.2, max_nb_states = 40):
    '''
    each probability can be multiplied to get a likelihood of the model knowing
    the parameters LocErr, D0 the diff coefficient of state 0 and F0 fraction of
    state 0, D1 the D coef at state 1, p01 the probability of transition from
    state 0 to 1 and p10 the proba of transition from state 1 to 0.
    here sum the logs(likelihood) to avoid too big numbers
    '''
    #print(params)
    nb_tracks = len(all_tracks)
    LocErr2, ds, Fs, TrMat, gammas, betas, pBL = extract_params(params, dt, nb_states, nb_substeps, input_LocErr2, Matrix_type)
    #gammas[0]=-0.02
    #gammas[1]=0.02
    #ds[1]=0
    
    
    if type(all_disps)==type(None):
        precomputed_disps = False
        all_disps = typed.List([np.array([[0., 0]])])
    else:
        precomputed_disps = True
    
    if np.all(TrMat>0) and np.all(Fs>0) and np.all(ds[1:]-ds[:-1]>=0):
        if input_LocErr2 == None:
            LPs = prange_fit_LocErr(all_tracks, all_disps, precomputed_disps, nb_tracks, LocErr2, ds, Fs, TrMat, all_nb_locs, nb_dims, nb_states, gammas, betas, pBL, isBLs, cell_dims, nb_substeps, frame_len, min_len, threshold, max_nb_states, cartesian_prod, cartesian_prod_cat)
        else:
            LPs = prange_input_LocErr(all_tracks, all_disps, precomputed_disps, nb_tracks, input_LocErr2, ds, Fs, TrMat, all_nb_locs, nb_dims, nb_states, gammas, betas, pBL, isBLs, cell_dims, nb_substeps, frame_len, min_len, threshold, max_nb_states, cartesian_prod, cartesian_prod_cat)
        Cum_P = cp.sum(LPs)
        
        if verbose == 1:
            q = [param + ' = ' + str(np.round(params[param].value, 4)) for param in params]
            print(Cum_P, q)
        else:
            print('.', end='')
        out = - Cum_P # normalize by the number of tracks and number of displacements
    else:
        out = np.inf
        print('x',end='')
        if verbose == 1:
            q = [param + ' = ' + str(np.round(params[param].value, 4)) for param in params]
            print(out, q)
    if np.isnan(out):
        out = np.inf
        print('input parameters give nans, you may want to pick more suitable parameter initial values')
    #print(time() - t0)
    return out


def param_fitting(all_tracks,
                  dt,
                  params = None,
                  nb_states = 2,
                  nb_substeps = 1,
                  frame_len = 6,
                  verbose = 1,
                  Matrix_type = 1,
                  method = 'BFGS',
                  steady_state = False,
                  cell_dims = [1], # list of dimensions limit for the field of view (FOV) of the cell in um, a membrane protein in a typical e-coli cell in tirf would have a cell_dims = [0.5,3], in case of cytosolic protein one should imput the depth of the FOV e.g. [0.3] for tirf or [0.8] for hilo
                  input_LocErr = None, 
                  threshold = 0.1, 
                  max_nb_states = 120):
    
    '''
    vary_params = {'LocErr' : [True, True], 'D0' : True, 'D1' : True, 'F0' : True, 'p01' : True, 'p10' : True, 'pBL' : True}
    estimated_vals =  {'LocErr' : [0.025, 0.03], 'D0' : 1e-20, 'D1' : 0.05, 'F0' : 0.45, 'p01' : 0.05, 'p10' : 0.05, 'pBL' : 0.1}
    min_values = {'LocErr' : [0.007, 0.007], 'D0' : 1e-12, 'D1' : 0.00001, 'F0' : 0.001, 'p01' : 0.001, 'p10' : 0.001, 'pBL' : 0.001}
    max_values = {'LocErr' : [0.6, 0.6], 'D0' : 1, 'D1' : 10, 'F0' : 0.999, 'p01' : 1., 'p10' : 1., 'pBL' : 0.99}
    
    fitting the parameters to the data set
    arguments:
    all_tracks: dict describing the tracks with track length as keys (number of time positions, e.g. '23') of 3D arrays: dim 0 = track, dim 1 = time position, dim 2 = x, y position.
    dt: time in between frames.
    cell_dims: dimension limits (um).
    nb_substeps: number of virtual transition steps in between consecutive 2 positions.
    nb_states: number of states. estimated_vals, min_values, max_values should be changed accordingly to describe all states and transitions.
    frame_len: number of frames for which the probability is perfectly computed. See method of the paper for more details.
    verbose: if 1, print the intermediate values for each iteration of the fit.
    steady_state: True if tracks are considered at steady state (fractions independent of time), this is most likely not true as tracks join and leave the FOV.
    vary_params: dict specifying if each parameters should be changed (True) or not (False).
    estimated_vals: initial values of the fit. (stay constant if parameter fixed by vary_params). estimated_vals must be in between min_values and max_values even if fixed.
    min_values: minimal values for the fit.
    max_values: maximal values for the fit.
    outputs:
    model_fit: lmfit model
    
    in case of 3 states models vary_params, estimated_vals, min_values and max_values can be replaced :
    
    vary_params = {'LocErr' : True, 'D0' : False, 'D1' :  True, 'D2' : True, 'F0' : True, 'F1' : True, 'p01' : True, 'p02' : True, 'p10' : True,'p12' :  True,'p20' :  True, 'p21' : True, 'pBL' : True},
    estimated_vals = {'LocErr' : 0.023, 'D0' : 1e-20, 'D1' : 0.02, 'D2' :  0.1, 'F0' : 0.33,  'F1' : 0.33, 'p01' : 0.1, 'p02' : 0.1, 'p10' :0.1, 'p12' : 0.1, 'p20' :0.1, 'p21' :0.1, 'pBL' : 0.1},
    min_values = {'LocErr' : 0.007, 'D0' : 1e-20, 'D1' : 0.0000001, 'D2' :  0.000001, 'F0' : 0.001,  'F1' : 0.001, 'p01' : 0.001, 'p02' : 0.001, 'p10' :0.001, 'p12' : 0.001, 'p20' :0.001, 'p21' :0.001, 'pBL' : 0.001},
    max_values = {'LocErr' : 0.6, 'D0' : 1e-20, 'D1' : 1, 'D2' :  10, 'F0' : 0.999,  'F1' : 0.999, 'p01' : 1, 'p02' : 1, 'p10' : 1, 'p12' : 1, 'p20' : 1, 'p21' : 1, 'pBL' : 0.99}
    
    in case of 4 states models :
    
    vary_params = {'LocErr' : True, 'D0' : True, 'D1' : True, 'D2' :  True, 'D3' : True, 'F0' : True,  'F1' : True, 'F2' : True, 'p01' : True, 'p02' : True, 'p03' : True, 'p10' : True, 'p12' : True, 'p13' : True, 'p20' :True, 'p21' :True, 'p23' : True, 'p30' :True, 'p31' :True, 'p32' : True, 'pBL' : True}
    estimated_vals = {'LocErr' : 0.023, 'D0' : 1e-20, 'D1' : 0.02, 'D2' :  0.1, 'D3' : 0.5, 'F0' : 0.1,  'F1' : 0.2, 'F2' : 0.3, 'p01' : 0.1, 'p02' : 0.1, 'p03' : 0.1, 'p10' :0.1, 'p12' : 0.1, 'p13' : 0.1, 'p20' :0.1, 'p21' :0.1, 'p23' : 0.1, 'p30' :0.1, 'p31' :0.1, 'p32' : 0.1, 'pBL' : 0.1}
    min_values = {'LocErr' : 0.005, 'D0' : 0, 'D1' : 0, 'D2' :  0.001, 'D3' : 0.001, 'F0' : 0.001,  'F1' : 0.001, 'F2' : 0.001, 'p01' : 0.001, 'p02' : 0.001, 'p03' : 0.001, 'p10' :0.001, 'p12' : 0.001, 'p13' : 0.001, 'p20' :0.001, 'p21' :0.001, 'p23' : 0.001, 'p30' :0.001, 'p31' :0.001, 'p32' : 0.001, 'pBL' : 0.001}
    max_values = {'LocErr' : 0.023, 'D0' : 1, 'D1' : 1, 'D2' :  10, 'D3' : 100, 'F0' : 0.999,  'F1' : 0.999, 'F2' : 0.999, 'p01' : 1, 'p02' : 1, 'p03' : 1, 'p10' :1, 'p12' : 1, 'p13' : 1, 'p20' : 1, 'p21' : 1, 'p23' : 1, 'p30' : 1, 'p31' : 1, 'p32' : 1, 'pBL' : 0.99}
    '''
    '''
    if nb_states == 2:
        if not (len(min_values) == 7 and len(max_values) == 7 and len(estimated_vals) == 7 and len(vary_params) == 7):
            raise ValueError('estimated_vals, min_values, max_values and vary_params should all containing 7 parameters')
    elif nb_states == 3:
        if len(vary_params) != 13:
            vary_params = {'LocErr' : True, 'D0' : True, 'D1' :  True, 'D2' : True, 'F0' : True, 'F1' : True, 'p01' : True, 'p02' : True, 'p10' : True,'p12' :  True,'p20' :  True, 'p21' : True, 'pBL' : True},
        if len(estimated_vals) != 13:
            estimated_vals = {'LocErr' : 0.023, 'D0' : 1e-20, 'D1' : 0.02, 'D2' :  0.1, 'F0' : 0.33,  'F1' : 0.33, 'p01' : 0.1, 'p02' : 0.1, 'p10' :0.1, 'p12' : 0.1, 'p20' :0.1, 'p21' :0.1, 'pBL' : 0.1},
        if len(min_values) != 13:
            min_values = {'LocErr' : 0.007, 'D0' : 1e-20, 'D1' : 0.0000001, 'D2' :  0.000001, 'F0' : 0.001,  'F1' : 0.001, 'p01' : 0.001, 'p02' : 0.001, 'p10' :0.001, 'p12' : 0.001, 'p20' :0.001, 'p21' :0.001, 'pBL' : 0.001},
        if len(max_values) != 13:
            max_values = {'LocErr' : 0.023, 'D0' : 1, 'D1' : 1, 'D2' :  10, 'F0' : 0.999,  'F1' : 0.999, 'p01' : 1, 'p02' : 1, 'p10' :1, 'p12' : 1, 'p20' : 1, 'p21' : 1, 'pBL' : 0.99}
    elif nb_states == 4:
        if len(vary_params) != 21:
            vary_params = {'LocErr' : True, 'D0' : True, 'D1' : True, 'D2' :  True, 'D3' : True, 'F0' : True,  'F1' : True, 'F2' : True, 'p01' : True, 'p02' : True, 'p03' : True, 'p10' : True, 'p12' : True, 'p13' : True, 'p20' :True, 'p21' :True, 'p23' : True, 'p30' :True, 'p31' :True, 'p32' : True, 'pBL' : True}
        if len(estimated_vals) != 21:
            estimated_vals = {'LocErr' : 0.023, 'D0' : 1e-20, 'D1' : 0.02, 'D2' :  0.1, 'D3' : 0.5, 'F0' : 0.1,  'F1' : 0.2, 'F2' : 0.3, 'p01' : 0.1, 'p02' : 0.1, 'p03' : 0.1, 'p10' :0.1, 'p12' : 0.1, 'p13' : 0.1, 'p20' :0.1, 'p21' :0.1, 'p23' : 0.1, 'p30' :0.1, 'p31' :0.1, 'p32' : 0.1, 'pBL' : 0.1}
        if len(min_values) != 21:
            min_values = {'LocErr' : 0.005, 'D0' : 0, 'D1' : 0, 'D2' :  0.001, 'D3' : 0.001, 'F0' : 0.001,  'F1' : 0.001, 'F2' : 0.001, 'p01' : 0.001, 'p02' : 0.001, 'p03' : 0.001, 'p10' :0.001, 'p12' : 0.001, 'p13' : 0.001, 'p20' :0.001, 'p21' :0.001, 'p23' : 0.001, 'p30' :0.001, 'p31' :0.001, 'p32' : 0.001, 'pBL' : 0.001}
        if len(max_values) != 21:
            max_values = {'LocErr' : 0.023, 'D0' : 1, 'D1' : 1, 'D2' :  10, 'D3' : 100, 'F0' : 0.999,  'F1' : 0.999, 'F2' : 0.999, 'p01' : 1, 'p02' : 1, 'p03' : 1, 'p10' :1, 'p12' : 1, 'p13' : 1, 'p20' : 1, 'p21' : 1, 'p23' : 1, 'p30' : 1, 'p31' : 1, 'p32' : 1, 'pBL' : 0.99}
    
    if not str(all_tracks.__class__) == "<class 'dict'>":
        raise ValueError('all_tracks should be a dictionary of arrays with n there number of steps as keys')
    '''
    
    if params == None:
        params = generate_params(nb_states = nb_states,
                                 LocErr_type = 1,
                                 LocErr_bounds = [0.005, 0.1], # the initial guess on LocErr will be the geometric mean of the boundaries
                                 D_max = 3, # maximal diffusion length allowed
                                 Fractions_bounds = [0.001, 0.99],
                                 estimated_transition_rates = 0.1 # transition rate per step.
                                 )
    
    l_list = np.sort(np.array(list(all_tracks.keys())).astype(int)).astype(str)
    sorted_tracks = []
    sorted_LocErrs2 = []
    all_nb_locs = []
    isBLs = []
    
    min_len = np.inf
    max_len = 0
    
    for l in l_list:
        if all_tracks[l].shape[0] > 0:
            if min_len > all_tracks[l].shape[1] :
                min_len = all_tracks[l].shape[1]
            if max_len < all_tracks[l].shape[1] :
                max_len = all_tracks[l].shape[1]
        for k in range(all_tracks[l].shape[0]):
            current_track = all_tracks[l][k]
            # Next we need to extend the dimension (to later have a dimension for the possible sequences of states)
            # We also need to reverses the time points ( `::-1]`) necessary for the ireration process as currently implemented 
            current_track = current_track[None, ::-1]
            sorted_tracks.append(current_track)
            all_nb_locs.append(all_tracks[l][k].shape[0])
            if all_tracks[l][k].shape[0] == max_len:
                isBLs.append(0) # last position correspond to tracks which didn't disapear within maximum track length
            else:
                isBLs.append(1)
            if input_LocErr != None:
                sorted_LocErrs2.append(input_LocErr[l][k][None, ::-1]**2) # the compute the variance as it is the metric we need
    sorted_tracks.reverse() # We reorder the tracks to start computing the long tracks (more time consuming) 
    all_nb_locs.reverse()
    all_tracks = typed.List(sorted_tracks)
    if input_LocErr != None:
        sorted_LocErrs2.reverse()
        input_LocErr2 = typed.List(sorted_LocErrs2)
    else:
        input_LocErr2 = None
    
    nb_dims = all_tracks[0].shape[-1]
    
    isBLs = np.array(isBLs).astype(np.int8)
    all_nb_locs = np.array(all_nb_locs).astype(np.int32)
    
    if len(all_tracks) < 1:
        raise ValueError('No track could be detected. The loaded tracks seem empty. Errors often come from wrong input paths.')
    
    args = typed.List(([np.arange(nb_states)])*nb_substeps)
    cartesian_prod = cartesian_jit(args)[:,::-1].astype(np.int8)
    
    cartesian_prod_cat = to_Categorical(cartesian_prod, nb_states)
    
    cell_dims = np.array(cell_dims)
    
    fit = minimize(cum_Proba_Cs, params, args=(all_tracks, all_disps, dt, cell_dims, input_LocErr2, nb_states, min_len, cartesian_prod, cartesian_prod_cat, nb_substeps, frame_len, isBLs, all_nb_locs, nb_dims, verbose, Matrix_type, threshold, max_nb_states), method = method, nan_policy = 'propagate')
    if verbose == 0:
        print('')
    
    '''
    #to inverse state indexes:
    import copy
    idxs = [1,0,2,3]
    corr_params = copy.deepcopy(fit.params)
    for param in params[-13:-1]:
        i =  idxs[int(param[1])]
        j =  idxs[int(param[2])]
        val = float(res[param][3])
        corr_params['p' + i + j][3] = val
    '''
    return fit


def param_anomalous_fitting(all_tracks,
                  dt,
                  params = None,
                  nb_states = 2,
                  nb_substeps = 1,
                  frame_len = 6,
                  verbose = 1,
                  Matrix_type = 1,
                  method = 'BFGS',
                  steady_state = False,
                  cell_dims = [1], # list of dimensions limit for the field of view (FOV) of the cell in um, a membrane protein in a typical e-coli cell in tirf would have a cell_dims = [0.5,3], in case of cytosolic protein one should imput the depth of the FOV e.g. [0.3] for tirf or [0.8] for hilo
                  input_LocErr = None, 
                  threshold = 0.1, 
                  max_nb_states = 120):
    
    if params == None:
        params = generate_params(nb_states = nb_states,
                                 LocErr_type = 1,
                                 LocErr_bounds = [0.005, 0.1], # the initial guess on LocErr will be the geometric mean of the boundaries
                                 D_max = 3, # maximal diffusion length allowed
                                 Fractions_bounds = [0.001, 0.99],
                                 estimated_transition_rates = 0.1 # transition rate per step.
                                 )
    
    tracks_dict = all_tracks.copy()
    
    l_list = np.sort(np.array(list(all_tracks.keys())).astype(int)).astype(str)
    sorted_tracks = []
    sorted_LocErrs2 = []
    all_nb_locs = []
    isBLs = []
    
    min_len = np.inf
    max_len = 0
    
    for l in l_list:
        if all_tracks[l].shape[0] > 0:
            if min_len > all_tracks[l].shape[1] :
                min_len = all_tracks[l].shape[1]
            if max_len < all_tracks[l].shape[1] :
                max_len = all_tracks[l].shape[1]
        for k in range(all_tracks[l].shape[0]):
            current_track = all_tracks[l][k]
            # Next we need to extend the dimension (to later have a dimension for the possible sequences of states)
            # We also need to reverses the time points ( `::-1]`) necessary for the ireration process as currently implemented 
            current_track = current_track[None, ::-1]
            sorted_tracks.append(current_track)
            all_nb_locs.append(all_tracks[l][k].shape[0])
            if all_tracks[l][k].shape[0] == max_len:
                isBLs.append(0) # last position correspond to tracks which didn't disapear within maximum track length
            else:
                isBLs.append(1)
            if input_LocErr != None:
                sorted_LocErrs2.append(input_LocErr[l][k][None, ::-1]**2) # the compute the variance as it is the metric we need
    sorted_tracks.reverse() # We reorder the tracks to start computing the long tracks (more time consuming) 
    all_nb_locs.reverse()
    all_tracks = typed.List(sorted_tracks)
    if input_LocErr != None:
        sorted_LocErrs2.reverse()
        input_LocErr2 = typed.List(sorted_LocErrs2)
    else:
        input_LocErr2 = None
    
    nb_dims = all_tracks[0].shape[-1]
    
    isBLs = np.array(isBLs).astype(np.int8)
    all_nb_locs = np.array(all_nb_locs).astype(np.int32)
    
    if len(all_tracks) < 1:
        raise ValueError('No track could be detected. The loaded tracks seem empty. Errors often come from wrong input paths.')
    
    args = typed.List(([np.arange(nb_states)])*nb_substeps)
    cartesian_prod = cartesian_jit(args)[:,::-1].astype(np.int8)
    
    cartesian_prod_cat = to_Categorical(cartesian_prod, nb_states)
    
    cell_dims = np.array(cell_dims)
    
    for param in params:
        if param.startswith('gamma') or param.startswith('beta'):
            params[param].vary = False
            if param.startswith('gamma'):
                params[param].value = 0
    
    all_disps = None
    
    fit = minimize(cum_Proba_Cs, params, args=(all_tracks[::200], all_disps, dt, cell_dims, input_LocErr2, nb_states, min_len, cartesian_prod, cartesian_prod_cat, nb_substeps, frame_len, isBLs, all_nb_locs, nb_dims, verbose, Matrix_type, threshold, max_nb_states), method = method, nan_policy = 'propagate')
    
    LocErr2, ds, Fs, TrMat, gammas, betas, pBL = extract_params(params, dt, nb_states, nb_substeps, input_LocErr2, Matrix_type)
    
    ds = ds/0.7
    for state in range(nb_states):
        params['gamma'+str(state)].max = 2*ds[state]
        
    LocErr = LocErr2[0,0,0]**0.5
    TrMat = np.zeros((nb_states, nb_states)) + 0.01
    TrMat[np.arange(nb_states), np.arange(nb_states)] = 0
    TrMat[np.arange(nb_states), np.arange(nb_states)] = 1 - np.sum(TrMat, 1)
    
    all_mus, all_refined_sigmas = position_refinement(tracks_dict, LocErr, ds, Fs, TrMat, frame_len = 7)
    
    sorted_disps = []
    for l in l_list:
        for k in range(all_mus[l].shape[0]):
            current_track = all_mus[l][k]
            disps = current_track[1:] - current_track[:-1]
            disps = disps[::-1]
            # Next we need to extend the dimension (to later have a dimension for the possible sequences of states)
            # We also need to reverses the time points ( `::-1]`) necessary for the ireration process as currently implemented 
            sorted_disps.append(disps)
    sorted_disps.reverse()
    all_disps = typed.List(sorted_disps)
    
    for param in params:
        if param.startswith('gamma'):
            params[param].vary = True
        if param.startswith('beta'):
            params[param].vary = False
    
    method = 'powell'
    params['gamma1'].value = 0.02
    fit = minimize(cum_Proba_Cs, params, args=(all_tracks, all_disps, dt, cell_dims, input_LocErr2, nb_states, min_len, cartesian_prod, cartesian_prod_cat, nb_substeps, frame_len, isBLs, all_nb_locs, nb_dims, verbose, Matrix_type, threshold, max_nb_states), method = method, nan_policy = 'propagate')
    
    if verbose == 0:
        print('')
    
    '''
    #to inverse state indexes:
    import copy
    idxs = [1,0,2,3]
    corr_params = copy.deepcopy(fit.params)
    for param in params[-13:-1]:
        i =  idxs[int(param[1])]
        j =  idxs[int(param[2])]
        val = float(res[param][3])
        corr_params['p' + i + j][3] = val
    '''
    return fit

def select_model(all_tracks,
                 dt,
                 nb_states = [2, 5],
                 params = None,
                 nb_substeps = 1,
                 frame_len = 6,
                 verbose = 1,
                 workers = 1,
                 Matrix_type = 1,
                 method = 'bfgs',
                 steady_state = False,
                 cell_dims = [1], # list of dimensions limit for the field of view (FOV) of the cell in um, a membrane protein in a typical e-coli cell in tirf would have a cell_dims = [0.5,3], in case of cytosolic protein one should imput the depth of the FOV e.g. [0.3] for tirf or [0.8] for hilo
                 input_LocErr = None, 
                 threshold = 0.2, 
                 max_nb_states = 120):

    Max_Ls = []
    fits = []
    BIC = []
    
    nb_disps = 0
    for l in all_tracks:
        nb_disps = nb_disps + all_tracks[l].shape[0] * (all_tracks[l].shape[1] - 1) * all_tracks[l].shape[2]
    
    for n in range(nb_states[0], nb_states[1]+1):
        
        print('fitting for the %s state model:'%(n))
        
        fit = param_fitting(all_tracks,
                            dt,
                            params = None,
                            nb_states = 2,
                            nb_substeps = 1,
                            frame_len = 6,
                            verbose = 1,
                            workers = 1,
                            Matrix_type = 1,
                            method = 'bfgs',
                            steady_state = False,
                            cell_dims = [10], # list of dimensions limit for the field of view (FOV) of the cell in um, a membrane protein in a typical e-coli cell in tirf would have a cell_dims = [0.5,3], in case of cytosolic protein one should imput the depth of the FOV e.g. [0.3] for tirf or [0.8] for hilo
                            input_LocErr = None, 
                            threshold = 0.2, 
                            max_nb_states = 120)
        
        N + N + N*N
        k = n*(n+2) # number of parameters of the model
        
        LL = - fit.residual[0]
        Max_Ls.append(LL)
        BIC = -2*LL + k * np.log(nb_disps)
        BICs.append(BIC)
        fits.append(fit)
        fit.params
        print('BIC for %s states model: %s'%(n, ))


