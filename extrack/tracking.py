#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:23:30 2022

@author: francois
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:28:38 2022
@author: Franc
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
import itertools
import scipy
from lmfit import minimize, Parameters

import multiprocessing
try:
    multiprocessing.set_start_method('fork')
    start_method = 'fork'
except:
    print('multiprocessing is only supported on Linux and MacOS')
    start_method = 'notfork'

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

def ds_froms_states(ds, cur_states):
    cur_d2s = ds[cur_states]**2
    cur_d2s = (cur_d2s[:,:,1:] + cur_d2s[:,:,:-1]) / 2 # assuming a transition at the middle of the substeps
    # we can average the variances of displacements per step to get the actual std of displacements per step
    cur_d2s = cp.mean(cur_d2s, axis = 2)
    cur_d2s = cur_d2s[:,:,None]
    cur_d2s = cp.array(cur_d2s)
    return cur_d2s

'''
l2 = LocErr2[:,:,min(LocErr_index,nb_locs-current_step)]
Ci = Cs[:,:,nb_locs-current_step]
l2.shape
Ci.shape
m_arr.shape
s2_arr.shape
'''

def log_integrale_dif(Ci, l2, cur_d2s, m_arr, s2_arr):
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
    l2_plus_s2_arr = l2+s2_arr
    new_m = (m_arr*l2 + Ci*s2_arr)/(l2+s2_arr)
    new_s2 = ((cur_d2s*l2 + cur_d2s*s2_arr + l2*s2_arr)/l2_plus_s2_arr)
    if s2_arr.shape[2] == 1:
        new_K = m_arr.shape[2] * -0.5*cp.log(2*np.pi*(l2_plus_s2_arr[:,:,0])) - cp.sum(((Ci-m_arr).astype(float))**2/(2*l2_plus_s2_arr),axis = 2)
    else:
        new_K = np.sum(-0.5*cp.log(2*np.pi*(l2_plus_s2_arr)), 2) - cp.sum(((Ci-m_arr).astype(float))**2/(2*l2_plus_s2_arr),axis = 2)
    return new_m, new_s2, new_K

#Ci, l2 = Cs[:,:,nb_locs-current_step], LocErr2[:,:,min(LocErr_index,nb_locs-current_step)]
def first_log_integrale_dif(Ci, l2, cur_d2s):
    '''
    convolution of 2 normal laws = normal law (mean = sum of means and variance = sum of variances)
    '''
    s2_arr = l2+cur_d2s
    m_arr = Ci
    return m_arr, s2_arr
#Cs = all_tracks['60']
#Cs = args[0]
#betas[1] =0.3
#cur_Bs_cat[:,mu_mask]
#previous_mus.shape

def eta(mu_mask, gammas, betas, cur_Bs, previous_mus, state_lengths):
    '''
    Parameters
    ----------
    gamma : float
        the coefficient for anomalous diffusion, cannot equal 0. Corresponds to
        the speed of the linear motion if positive or to the factor used to compute
        the new center of mass if negative.
    beta : float
        Coefficient used to weight the contribution of each previous displacements
        if gamma is positive.
    previous_mus : numpy array
        array of the previous most likely positions.
    cur_state_length : int
        length of previous_mus. the number of previous states to consider
    cur_d : float
        current diffusion coefficient.

    Returns
    -------
    out : TYPE
        DESCRIPTION.
    '''
    max_len = np.max(state_lengths)
    
    #previous_mus = previous_mus[-max_len:]
    length_mask = np.cumsum(np.eye(max_len)[max_len - state_lengths[mu_mask]], axis=1)[:,:-1]
    length_mask = length_mask.T
    cur_gammas = gammas[cur_Bs[0,mu_mask,0]]
    
    beta_weights = np.exp(- betas[None] * np.arange(max_len-1)[:,None])[::-1]
    beta_weights = beta_weights[:, cur_Bs[0,mu_mask,0]]
    beta_weights = beta_weights * length_mask
    beta_weights = beta_weights / np.sum(beta_weights,axis = 0, keepdims = True)
    
    disps = previous_mus[-(max_len-1):, :, mu_mask] - previous_mus[-(max_len-1)-1:-1, :, mu_mask]

    disps = np.sum(disps * beta_weights[:,None,:,None], 0)
    norms = np.sum(disps**2, axis = 2, keepdims=True)**0.5
    out = cur_gammas[None, :,None] * (disps / norms)
    '''

    max_len = np.max(state_lengths)
    
    #previous_mus = previous_mus[-max_len:]
    length_mask = np.cumsum(np.eye(max_len+1)[max_len - state_lengths[mu_mask]], axis=1)[:,:-1]
    length_mask = length_mask.T
    cur_gammas = gammas[cur_Bs[0,mu_mask,0]]
    
    beta_weights = np.exp(- betas[None] * np.arange(max_len)[:,None])[::-1]
    beta_weights = beta_weights[:, cur_Bs[0,mu_mask,0]]
    beta_weights = beta_weights * length_mask
    beta_weights = beta_weights / np.sum(beta_weights,axis = 0, keepdims = True)
    #beta_weights.shape
    #previous_mus.shape
    disps = previous_mus[1:max_len, :, mu_mask] - previous_mus[-max_len:-1, :, mu_mask]
    #disps.shape
    #previous_mus[:, :, mu_mask]
    disps = np.sum(disps * beta_weights[:,None,:,None], 0)
    norms = np.sum(disps**2, axis = 2, keepdims=True)**0.5
    out = cur_gammas[None, :,None] * (disps / norms)
    '''
    return out

#plt.scatter(previous_mus[-1,:,:,0], previous_mus[-1,:,:,1])
#previous_mus.shape
#Cs, LocErr, ds, Fs, TrMat, gammas, betas, pBL,isBL, cell_dims, nb_substeps, frame_len, min_len, threshold, max_nb_states = args_prod[0]
#betas[:] = [0.1,0.5]

def P_Cs_inter_bound_stats_th(Cs, LocErr, ds, Fs, TrMat, gammas = [0,0.02], betas = [0.5,0.5], pBL=0.1, isBL = 1, cell_dims = [0.5], nb_substeps=1, frame_len = 6, do_preds = 1, min_len = 3, threshold = 0.05, max_nb_states = 30):
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
    #Cs, LocErr, ds, Fs, TrMat, pBL, isBL, cell_dims, nb_substeps, frame_len, min_len = args_prod[0]
    #max_nb_states = 120

    #ds = np.array([0,0.004, 0.25])
    nb_Tracks = Cs.shape[0]
    nb_locs = Cs.shape[1] # number of localization per track
    
    dtype = 'float64'
    
    nb_dims = Cs.shape[2] # number of spatial dimensions (x, y) or (x, y, z)
    Cs = Cs[:,None].astype('float64')
    Cs = cp.array(Cs)
    nb_states = TrMat.shape[0]
    Cs = Cs[:,:,::-1] # I built the model going from the last index to the first index of the reversed positions. Which is equivalent to an iteration from the first position to the last one. 
    LocErr = LocErr[:,None].astype(dtype)
    LocErr = LocErr[:,:,::-1] # useful when single peak localization error is inputed,
    LocErr2 = LocErr**2
    if LocErr.shape[2] == 1: # combined to min(LocErr_index, nb_locs-current_step) it will select the right index
        LocErr_index = -1
    elif LocErr.shape[2] == nb_locs:
        LocErr_index = nb_locs
    else:
        raise ValueError("Localization error is not specified correctly, in case of unique localization error specify a float number in estimated_vals['LocErr'].\n If one localization error per dimension, specify a list or 1D array of elements the localization error for each dimension.\n If localization error is predetermined by another method for each position the argument input_LocErr should be a dict for each track length of the 3D arrays corresponding to all_tracks (can be obtained from the reader functions using the opt_colname argument)")
    if do_preds:
        preds = np.zeros((nb_Tracks, nb_locs, nb_states))-1
    else :
        preds = []
    
    if nb_locs < 2:
        raise ValueError('minimal track length = 2, here track length = %s'%nb_locs)
    
    t0 = time()
    
    sub_Bs = get_all_Bs(nb_substeps, nb_states)[None]
    TrMat = cp.array(TrMat.T)
    current_step = 1
    
    cur_Bs = get_all_Bs(nb_substeps + 1, nb_states)[None] # get initial sequences of states
    cur_Bs_cat = (cur_Bs[:,:,:,None] == np.arange(nb_states)[None,None,None,:]).astype('float64')
    
    cur_states = cur_Bs[:,:,0:nb_substeps+1].astype(int) #states of interest for the current displacement
    cur_nb_Bs = cur_Bs.shape[1]
    # compute the vector of diffusion stds knowing the current states
    ds = cp.array(ds).astype(dtype)
    gammas = np.array(gammas)
    betas = np.array(betas)
    anomalous = np.any(gammas != 0)
    
    Fs = cp.array(Fs).astype(dtype)
    #beta = np.log(2)
    #np.exp(-np.arange(10)*np.log(2))
    
    LT = get_Ts_from_Bs(cur_states, TrMat) # Log proba of transitions per step
    LF = cp.log(Fs[cur_states[:,:,-1]]) # Log proba of finishing/starting in a given state (fractions)
        
    LP = LT + LF #+ compensate_leaving
    # current log proba of seeing the track
    LP = cp.repeat(LP, nb_Tracks, axis = 0)
    cur_d2s = ds[cur_states]**2
    cur_d2s = (cur_d2s[:,:,1:] + cur_d2s[:,:,:-1]) / 2 # assuming a transition at the middle of the substeps
    
    # we can average the variances of displacements per step to get the actual std of displacements per step
    cur_d2s = cp.mean(cur_d2s, axis = 2)
    cur_d2s = cur_d2s[:,:,None]
    cur_d2s = cp.array(cur_d2s)
    
    sub_Bs = cur_Bs.copy()[:,:cur_Bs.shape[1]//nb_states,:nb_substeps] # list of possible current states we can meet to compute the proba of staying in the FOV
    sub_ds = (cp.mean(ds[sub_Bs]**2, axis = 2)**0.5).astype(float) # corresponding list of d
    sub_ds = asnumpy(sub_ds)
    
    p_stay = np.ones(sub_ds.shape[-1])
    for cell_len in cell_dims:
        xs = np.linspace(0+cell_len/2000,cell_len-cell_len/2000,1000)
        cur_p_stay = ((cp.mean(scipy.stats.norm.cdf((cell_len-xs[:,None])/(sub_ds+1e-200)) - scipy.stats.norm.cdf(-xs[:,None]/(sub_ds+1e-200)),0))) # proba to stay in the FOV for each of the possible cur Bs
        p_stay = p_stay*cur_p_stay
    p_stay = cp.array(p_stay)
    Lp_stay = cp.log(p_stay * (1-pBL)) # proba for the track to survive = both stay in the FOV and not bleach
    
    # inject the first position to get the associated m_arr and Ks :
    m_arr, s2_arr = first_log_integrale_dif(Cs[:,:, nb_locs-current_step], LocErr2[:,:,min(LocErr_index, nb_locs-current_step)], cur_d2s)
    
    m_arr = cp.repeat(m_arr, cur_nb_Bs, axis = 1)
    removed_steps = 0
    
    previous_mus = np.copy(m_arr[None])
    
    current_step += 1
    
    #TrMat = np.array([[0.9,0.1],[0.2,0.8]])
    while current_step <= nb_locs-1:
        if anomalous:
            '''
            the aim of this section is to update the mus (and sigmas) to consider 
            anomalous diffusion:
            to do so, we first select the sequences of states that remain the same for at least 3 time points
            The two (or more previous positions where the particle is in the given state are then used to compute
            the orientation of the motion, gamma then serves as the norm of the displacement)
            when the particle is in persistent motion (gamma>0) for only two steps, we only have 
            information about the first position, therefore, we cannot determine a direction
            The actuall density shape would be donut like. as we cannot consider it, we approximate it
            by au gaussian with increased standard deviation (increased by gamma). 
            '''
            # the mean creates a likely fatal approximation by fusing the sequences of states from one track to another. 
            # to avoid this error, analyse tracks one by one using nb_max = 1 in cum_Proba_Cs
            state_args = np.argmax(np.mean(cur_Bs_cat[:,:,:], axis = 0), axis = 2) # 
            #mask = np.all(state_args[:,:2] == state_args[:,:1], axis = 1)
            
            #comput the umber of steps in the same state
            state_args_inf = np.concatenate((state_args, np.full((state_args.shape[0],1), np.inf)),1) #we add the inf so the sequences end with a high value to pass it to the argmin function
            state_lengths = np.argmin(state_args_inf == state_args[:,:1], 1)-1
            # A state_length == 1 means we should only consider the last mu, state_length == 2 the 2 last mu, etc
            cur_gammas = gammas[cur_Bs[0,:,0]]

            # in case of persistent motion (gamma>0), if we have a state_length == 1, we cannot have any information on the orientation of the motion. Ideally we would need to model the probability with an annular shape but this is not possible. as an approximation, we can increase sigma of gamma instead of changing mu. 
            sigma_mask = (state_lengths == 1) * (cur_gammas > 0)
            if np.any(sigma_mask):
                s2_arr[:, sigma_mask] = s2_arr[:, sigma_mask] + cur_gammas[sigma_mask][None,:,None]**2
            # mu masks (we dont conpute mu for states with a gamma of 0 as the change will be 0 anyway):
            mu_mask = (state_lengths >= 2) * (cur_gammas != 0)
            if np.any(mu_mask):
                m_arr[:, mu_mask] = m_arr[:, mu_mask] + eta(mu_mask, gammas, betas, cur_Bs, previous_mus, state_lengths)
            '''
            for seq_ID in range(cur_states.shape[1]):
                
                if sigma_mask[seq_ID]:
                    s2_arr[:, seq_ID] = s2_arr[:, seq_ID] + cur_gammas[seq_ID]**2
                elif mu_mask[seq_ID]:
                    cur_state_length = state_length[seq_ID] # current state length
                    cur_d = cur_d2s[0,seq_ID,0]
                    m_arr[:, seq_ID] = m_arr[:, seq_ID] + eta(cur_gammas, cur_betas, previous_mus[-cur_state_length:,:,seq_ID], cur_state_length, cur_d)
            '''
        # update cur_Bs to describe the states at the next step:
        # cur_Bs = get_all_Bs(current_step*nb_substeps+1 - removed_steps, nb_states)[None]
        # cur_Bs = all_Bs[:,:nb_states**(current_step + nb_substeps - removed_steps),:current_step + nb_substeps - removed_steps]
        for iii in range(nb_substeps):
            #cur_Bs = np.concatenate((np.repeat(np.mod(np.arange(cur_Bs.shape[1]*nb_states),nb_states)[None,:,None], nb_Tracks, 0), np.repeat(cur_Bs,nb_states,1)),-1)
            cur_Bs = np.concatenate((np.mod(np.arange(cur_Bs.shape[1]*nb_states),nb_states)[None,:,None], np.repeat(cur_Bs,nb_states,1)),-1)
            new_states = np.repeat(np.mod(np.arange(cur_Bs_cat.shape[1]*nb_states, dtype = 'int8'),nb_states)[None,:,None,None] == np.arange(nb_states, dtype = 'int8')[None,None,None], cur_Bs_cat.shape[0], 0).astype('int8')
            cur_Bs_cat = np.concatenate((new_states, np.repeat(cur_Bs_cat,nb_states,1)),-2)
        
        #np.mod(np.arange(cur_Bs.shape[1]*nb_states),nb_states).shape
        cur_states = cur_Bs[:1,:,0:nb_substeps+1].astype(int)
        # compute the vector of diffusion stds knowing the states at the current step
        cur_d2s = ds[cur_states]**2
        cur_d2s = (cur_d2s[:,:,1:] + cur_d2s[:,:,:-1]) / 2 # assuming a transition at the middle of the substeps
        cur_d2s = cp.mean(cur_d2s, axis = 2)
        cur_d2s = cur_d2s[:,:,None]
        
        LT = get_Ts_from_Bs(cur_states, TrMat)
        
        # repeat the previous matrix to account for the states variations due to the new position
        m_arr = cp.repeat(m_arr, nb_states**nb_substeps , axis = 1)
        previous_mus = cp.repeat(previous_mus, nb_states**nb_substeps, axis = 2)
        s2_arr = cp.repeat(s2_arr, nb_states**nb_substeps, axis = 1)
        LP = cp.repeat(LP, nb_states**nb_substeps, axis = 1)

        m_arr, s2_arr, LC = log_integrale_dif(Cs[:,:,nb_locs-current_step], LocErr2[:,:,min(LocErr_index,nb_locs-current_step)], cur_d2s, m_arr, s2_arr)
        
        if current_step >= min_len:
            LL = Lp_stay[np.argmax(np.all(cur_states[:,None,:,:-1] == sub_Bs[:,:,None],-1),1)] # pick the right proba of staying according to the current states
        else:
            LL = 0
        
        LP += LT + LC + LL # current (log) constants associated with each track and sequences of states
        del LT, LC
        
        cur_nb_Bs = len(cur_Bs[0]) # current number of sequences of states
        #print(current_step, m_arr.shape)
        
        if cur_nb_Bs>max_nb_states:
            threshold = threshold*1.2
            #print('threshold', threshold)

        '''idea : the position and the state 6 steps ago should not impact too much the 
        probability of the next position so the m_arr and s2_arr of tracks with the same 6 last 
        states must be very similar, we can then fuse the parameters of the pairs of Bs
        which vary only for the last step (7) and sum their probas'''
        
        if current_step < nb_locs-1: # do not fuse sequences at the last step as it doesn't improves speed.            
            m_arr, s2_arr, LP, cur_Bs, cur_Bs_cat, previous_mus = fuse_tracks_th(m_arr,
                                                                   s2_arr,
                                                                   LP,
                                                                   cur_Bs,
                                                                   cur_Bs_cat,
                                                                   nb_Tracks,
                                                                   previous_mus,
                                                                   gammas,
                                                                   nb_states = nb_states,
                                                                   nb_dims = nb_dims,
                                                                   do_preds = do_preds,
                                                                   threshold = threshold,
                                                                   frame_len = frame_len) # threshold on values normalized by sigma.

            cur_nb_Bs = len(cur_Bs[0])
            #print(current_step, m_arr.shape)
            removed_steps += 1
        
        previous_mus = np.concatenate((previous_mus, m_arr[None]), axis = 0)
        
        current_step += 1
    if anomalous:
        state_args = np.argmax(np.mean(cur_Bs_cat[:,:,:], axis = 0), axis = 2) # 
        state_args_inf = np.concatenate((state_args, np.full((state_args.shape[0],1), np.inf)),1) #we add the inf so the sequences end with a high value to pass it to the argmin function
        state_lengths = np.argmin(state_args_inf == state_args[:,:1], 1)-1
        cur_gammas = gammas[cur_Bs[0,:,0]]

        # in case of persistent motion (gamma>0), if we have a state_length == 1, we cannot have any information on the orientation of the motion. Ideally we would need to model the probability with an annular shape but this is not possible. as an approximation, we can increase sigma of gamma instead of changing mu. 
        sigma_mask = (state_lengths == 1) * (cur_gammas > 0)
        if np.any(sigma_mask):
            s2_arr[:, sigma_mask] = s2_arr[:, sigma_mask] + cur_gammas[sigma_mask][None,:,None]**2
        # mu masks (we dont conpute mu for states with a gamma of 0 as the change will be 0 anyway):
        mu_mask = (state_lengths >= 2) * (cur_gammas != 0)
        m_arr[:, mu_mask] = m_arr[:, mu_mask] + eta(mu_mask, gammas, betas, cur_Bs, previous_mus, state_lengths)
       
    if not isBL:
        LL = 0
    else:
        for iii in range(nb_substeps):
            #cur_Bs = np.concatenate((np.repeat(np.mod(np.arange(cur_Bs.shape[1]*nb_states),nb_states)[None,:,None], nb_Tracks, 0), np.repeat(cur_Bs,nb_states,1)),-1)
            cur_Bs = np.concatenate((np.mod(np.arange(cur_Bs.shape[1]*nb_states),nb_states)[None,:,None], np.repeat(cur_Bs,nb_states,1)),-1)
            new_states = np.repeat(np.mod(np.arange(cur_Bs_cat.shape[1]*nb_states, dtype = 'int8'),nb_states)[None,:,None,None] == np.arange(nb_states, dtype = 'int8')[None,None,None], cur_Bs_cat.shape[0], 0).astype('int8')
            cur_Bs_cat = np.concatenate((new_states, np.repeat(cur_Bs_cat,nb_states,1)),-2)
                
        cur_states = cur_Bs[:,:,0:nb_substeps+1].astype(int)
        len(cur_Bs[0])
        LT = get_Ts_from_Bs(cur_states, TrMat)
        #cur_states = cur_states[:,:,0]
        # repeat the previous matrix to account for the states variations due to the new position
        m_arr = cp.repeat(m_arr, nb_states**nb_substeps , axis = 1)
        s2_arr = cp.repeat(s2_arr, nb_states**nb_substeps, axis = 1)
        LP = cp.repeat(LP, nb_states**nb_substeps, axis = 1)
        
        #LL = Lp_stay[np.argmax(np.all(cur_states[:,None] == sub_Bs[:,:,None],-1),1)] # pick the right proba of staying according to the current states
        #end_p_stay = p_stay[np.argmax(np.all(cur_states[:,None:,:-1] == sub_Bs[:,:,None],-1),1)]
        end_p_stay = p_stay[cur_states[:,None:,:-1]][:,:,0]
        end_p_stay.shape
        LL = cp.log(pBL + (1-end_p_stay) - pBL * (1-end_p_stay)) + LT
    
    new_s2_arr = cp.array((s2_arr + LocErr2[:,:, min(LocErr_index, nb_locs-current_step)]))
    log_integrated_term = cp.sum(-0.5*cp.log(2*np.pi*new_s2_arr) - (Cs[:,:,0] - m_arr)**2/(2*new_s2_arr),axis=2)
    #LF = cp.log(Fs[cur_Bs[:,:,0].astype(int)]) # Log proba of starting in a given state (fractions)
    #LF = cp.log(0.5)
    # cp.mean(cp.log(Fs[cur_Bs[:,:,:].astype(int)]), 2) # Log proba of starting in a given state (fractions)
    LP += log_integrated_term + LL
    
    pred_LP = LP
    if np.max(LP)>600: # avoid overflow of exponentials, (drawback: mechanically also reduces the weights of longest tracks)
        pred_LP = LP - (np.max(LP)-600)
    
    P = np.exp(pred_LP)
    sum_P = np.sum(P, axis = 1, keepdims = True)[:,:,None]
    if do_preds:
        preds = np.sum(P[:,:,None,None]*cur_Bs_cat, axis = 1) / sum_P
        preds = preds[:,::-1]
    return LP, cur_Bs_cat, preds

def fuse_tracks_th(m_arr, s2_arr, LP, cur_Bs, cur_Bs_cat, nb_Tracks, previous_mus, gammas, nb_states = 2, nb_dims = 2, do_preds = 1, threshold = 0.2, frame_len = 6):
    '''
    The probabilities of the pairs of tracks must be added
    I chose to define the updated m_arr and s2_arr as the weighted average (of the variance for s2_arr)
    but other methods may be better
    As I must divid by a sum of exponentials which can be equal to zero because of underflow
    I correct the values in the exponetial to keep the maximal exp value at 0
    '''
    # cut the matrixes so the resulting matrices only vary for their last state
    m_arr.shape
    s_arr = s2_arr**0.5
    
    LL_threshold = - np.log(1e-20)
    
    groups = []
    grouped_IDs = []
    
    cur_Bs_cat.shape
    m_arr.shape
    LP.shape
    cur_Bs.shape
    
    max_LP = np.max(LP)
    
    LL_fuse = LP < max_LP - LL_threshold
    
    for state in range(nb_states):
        args = np.where(LL_fuse[0] * (cur_Bs[0,:,0]==state))[0]
        if len(args)>0:
            groups.append(args)
            grouped_IDs = grouped_IDs + list(args)    
    
    Remaining_IDs = np.delete(np.arange(m_arr.shape[1]), grouped_IDs)
    Remaining_ms = m_arr[:,Remaining_IDs]
    Remaining_ss = s_arr[:,Remaining_IDs]
    Remaining_cur_Bs = cur_Bs[:,Remaining_IDs]
    #cur_Bs[0,:,0]
    #Remaining_ms.shape
    #cur_Bs_cat.shape
    # group sequences of states according to their similarities
    for Bs_ID in Remaining_IDs:
        if not np.isin(Bs_ID, grouped_IDs):
            cur_m_arr = m_arr[:,Bs_ID]
            cur_s_arr = s_arr[:,Bs_ID]
            cur_B = cur_Bs[0,Bs_ID,0]
            state_mask = Remaining_cur_Bs[0,:,0] == cur_B
            
            m_mask_relative = (np.mean(np.abs(Remaining_ms[0, state_mask] - cur_m_arr[0,None]), 1)/Remaining_ss[0, state_mask,0]) < threshold
            #s_mask_relative = ((cur_s_arr[:,None] > (1 - threshold) * s_arr)*(cur_s_arr[:,None] < (1 + threshold) * s_arr))[0,:,0]
            s_mask_relative = (np.mean(np.abs(Remaining_ss[0, state_mask] - cur_s_arr[0,None]), 1)/Remaining_ss[0, state_mask,0]) < threshold
            
            args = np.where(m_mask_relative * s_mask_relative)[0]
            args = Remaining_IDs[state_mask][args] # go from the arg in the reduced selection to the global arg
            
            args = args[np.isin(args, grouped_IDs) == False] # remove elements that already belongs to a group
            
            groups.append(args)
            grouped_IDs = grouped_IDs + list(args)
            
    if len(grouped_IDs) != m_arr.shape[1]:
        raise ValueError('problem with grouping: len(grouped_IDs)=' + str(len(grouped_IDs)) + ' and m_arr.shape[1]=' + str( m_arr.shape[1]))

    # the grouped sequences of states need to share the same current state to be equivalent. To take this into account we split the groups according to this property
    # tracks with a unique state also need to not be merged for the anomalous diffusion analysis
    cur_Bs_cat.shape
    cur_Bs.shape
    
    # part that is too time consuming, find a way to make it faster
    nb_groups = len(groups)
    new_cur_Bs = np.empty((1, nb_groups, 1), dtype=(int))
    if not do_preds:
        cur_Bs_cat = cur_Bs_cat[:,:,:frame_len]
    '''
    new_cur_Bs_cat = np.zeros((nb_Tracks, nb_subgroups, cur_Bs_cat.shape[2], nb_dims), dtype = cur_Bs_cat.dtype)

    new_m_arr = np.zeros((nb_Tracks, nb_subgroups, m_arr.shape[2]),  dtype = m_arr.dtype)
    new_s2_arr = np.zeros((nb_Tracks, nb_subgroups, s2_arr.shape[2]),  dtype = s2_arr.dtype)
    new_LP = np.zeros((nb_Tracks, nb_subgroups),  dtype = LP.dtype)
    
    for Bs_ID, subgroup in enumerate(subgroups):
        
        max_LP = LP[:, subgroup].max(axis = 1, keepdims = True)
        weights = np.exp(LP[:, subgroup] - max_LP)
        sum_weights = np.sum(weights, 1, keepdims = True)
        new_cur_Bs[:, Bs_ID] = cur_Bs[:, subgroup[:1], 0]
        
        new_cur_Bs_cat[:, Bs_ID] = (np.sum(weights[:,:,None,None] * cur_Bs_cat[:, subgroup, :], 1) / sum_weights[:,:,None])
        #new_cur_Bs[:, Bs_ID] = (np.sum(weights[:,:,None] * cur_Bs[:, subgroup, :], 1) / )
        
        new_m_arr[:, Bs_ID] = np.sum(weights[:,:,None] * m_arr[:, subgroup, :], 1) / sum_weights
        new_s2_arr[:, Bs_ID] = np.sum(weights[:,:,None] * s2_arr[:, subgroup, :], 1) / sum_weights
        new_LP[:, Bs_ID] = np.log(np.sum(np.exp(LP[:, subgroup]-max_LP), axis = 1)) + np.squeeze(max_LP, axis = 1)

    # part that is too time consuming, find a way to make it faster
    nb_subgroups = len(subgroups)
    new_cur_Bs = np.zeros((1, nb_subgroups, 1), dtype=(int))
    if not do_preds:
        cur_Bs_cat = cur_Bs_cat[:,:,:frame_len]
    '''  
    new_cur_Bs_cat = np.zeros((nb_Tracks, nb_groups, cur_Bs_cat.shape[2], nb_states), dtype = cur_Bs_cat.dtype)
    new_m_arr = np.zeros((nb_Tracks, nb_groups, m_arr.shape[2]),  dtype = m_arr.dtype)
    new_s2_arr = np.zeros((nb_Tracks, nb_groups, s2_arr.shape[2]),  dtype = s2_arr.dtype)
    new_LP = np.zeros((nb_Tracks, nb_groups),  dtype = LP.dtype)
    
    new_previous_mus = np.zeros((len(previous_mus),nb_Tracks, nb_groups, m_arr.shape[2]),  dtype = m_arr.dtype)
    
    for Bs_ID, group in enumerate(groups):
        
        max_LP = LP[:, group].max(axis = 1, keepdims = True)
        weights = np.exp(LP[:, group] - max_LP)
        sum_weights = np.sum(weights, 1, keepdims = True)
        new_cur_Bs[:, Bs_ID] = cur_Bs[:, group[:1], 0]
        
        if len(group)>1:
            #sum_masks = np.zeros((nb_Tracks, cur_Bs_cat.shape[2], nb_states), dtype= bool)
            new_cur_Bs_cat[:, Bs_ID] = (np.sum(weights[:,:,None,None] * cur_Bs_cat[:, group, :], 1) / sum_weights[:,:,None])
        else:
            new_cur_Bs_cat[:, Bs_ID] =  cur_Bs_cat[:, group[0]]    
        
        new_m_arr[:, Bs_ID] = np.sum(weights[:,:,None] * m_arr[:, group, :], 1) / sum_weights
        new_s2_arr[:, Bs_ID] = np.sum(weights[:,:,None] * s2_arr[:, group, :], 1) / sum_weights
        new_LP[:, Bs_ID] = np.log(np.sum(np.exp(LP[:, group]-max_LP), axis = 1)) + np.squeeze(max_LP, axis = 1)
        new_previous_mus[:,:,Bs_ID] = np.sum(weights[None,:,:,None] * previous_mus[:,:, group, :], 2) / sum_weights[None]
        
    return new_m_arr, new_s2_arr, new_LP, new_cur_Bs, new_cur_Bs_cat, new_previous_mus

def get_all_Bs(nb_Cs, nb_states):
    '''
    produces a matrix of the possible sequences of states
    '''
    Bs_ID = np.arange(nb_states**nb_Cs)
    all_Bs = np.zeros((nb_states**nb_Cs, nb_Cs), int)
    
    for k in range(all_Bs.shape[1]):
        cur_row = np.mod(Bs_ID,nb_states**(k+1))
        Bs_ID = (Bs_ID - cur_row)
        all_Bs[:,k] = cur_row//nb_states**k
    return all_Bs

def get_Ts_from_Bs(all_Bs, TrMat):
    '''
    compute the probability of the sequences of states according to the markov transition model
    '''
    LT = cp.zeros((all_Bs.shape[:2]), dtype = float)
    # change from binary base 10 numbers to identify the consecutive states (from ternary if 3 states) 
    for k in range(len(all_Bs[0,0])-1):
        LT += cp.log(TrMat[all_Bs[:,:,k], all_Bs[:,:,k+1]])
    return LT

def Proba_Cs(Cs, LocErr, ds, Fs, TrMat, gammas, betas, pBL, isBL, cell_dims, nb_substeps, frame_len, min_len, threshold, max_nb_states):
    '''
    inputs the observed localizations and determine the probability of 
    observing these data knowing the localization error, D the diffusion coef,
    pu the proba of unbinding per step and pb the proba of binding per step
    sum the proba of Cs inter Bs (calculated with P_Cs_inter_bound_stats)
    over all Bs to get the proba of Cs (knowing the initial position c0)
    '''
    
    LP_CB, _, _  = P_Cs_inter_bound_stats_th(Cs, LocErr, ds, Fs, TrMat, gammas, betas, pBL,isBL,cell_dims, nb_substeps, frame_len, do_preds = 0, min_len = min_len, threshold = threshold, max_nb_states = max_nb_states)
    np.sum(LP_CB)
    # calculates P(C) the sum of P(C inter B) for each track
    max_LP = np.max(LP_CB, axis = 1, keepdims = True)
    LP_CB = LP_CB - max_LP
    max_LP = max_LP[:,0]
    P_CB = np.exp(LP_CB)
    P_C = cp.sum(P_CB, axis = 1) # sum over B
    LP_C = np.log(P_C) + max_LP # back to log proba of C without overflow due to exponential
    return LP_C

def Pool_star_P_inter(args):
    return P_Cs_inter_bound_stats(*args)[2] # returns the 3rd output which is the predictions

def predict_Bs(all_tracks,
               dt,
               params,
               cell_dims=[1],
               nb_states=2,
               frame_len=8,
               workers = 1,
               input_LocErr = None):
    '''
    inputs the observed localizations and parameters and determines the proba
    of each localization to be in a given state.
    
    arguments:
    all_tracks: dict describing the tracks with track length as keys (number of time positions, e.g. '23') of 3D arrays: dim 0 = track, dim 1 = time position, dim 2 = x, y position.
    params: lmfit parameters used for the model.
    dt: time in between frames.
    cell_dims: dimension limits (um). estimated_vals, min_values, max_values should be changed accordingly to describe all states and transitions.
    nb_states: number of states. estimated_vals, min_values, max_values should be changed accordingly to describe all states and transitions.
    frame_len: number of frames for which the probability is perfectly computed. See method of the paper for more details.
    
    outputs:
    pred_Bs: dict describing the state probability of each track for each time position with track length as keys (number of time positions, e.g. '23') of 3D arrays: dim 0 = track, dim 1 = time position, dim 2 = state.
    extrack.visualization.visualize_states_durations
    '''
    
    l_list = np.sort(np.array(list(all_tracks.keys())).astype(int)).astype(str)
    sorted_tracks = []
    sorted_LocErrs = []
    for l in l_list:
        if len(all_tracks[l]) > 0 :
            sorted_tracks.append(all_tracks[l])
            if input_LocErr != None:
                sorted_LocErrs.append(input_LocErr[l])
    all_tracks = sorted_tracks
    if input_LocErr != None:
        input_LocErr = sorted_LocErrs
    
    nb_substeps=1 # substeps should not impact the step labelling
    if type(params) == type(Parameters()):
        LocErr, ds, Fs, TrMat, gammas, betas, pBL = extract_params(params, dt, nb_states, nb_substeps, input_LocErr)
        '''elif type(params) == type({}):
        param_kwargs = []
        for param in params:
            param_kwargs.append({'name' : param, 'value' : params[param], 'vary': False})
        new_params = Parameters()
        [new_params.add(**param_kwargs[k]) for k in range(len(params))]
        LocErr, ds, Fs, TrMat, gammas, betas, pBL = extract_params(new_params, dt, nb_states, nb_substeps)'''
    else:
        raise TypeError("params must be either of the class 'lmfit.parameter.Parameters' or a dictionary of the relevant parameters")
    all_pred_Bs = []

    min_len = int(l_list[0])
    max_len = int(l_list[-1])
    
    Csss = []
    sigss = []
    isBLs = []
    for k in range(len(all_tracks)):
        if k == len(all_tracks)-1:
            isBL = 0 # last position correspond to tracks which didn't disapear within maximum track length
        else:
            isBL = 1
        Css = all_tracks[k]
        if input_LocErr != None:
            sigs = LocErr[k]
        nb_max = 50
        for n in range(int(np.ceil(len(Css)/nb_max))):
            Csss.append(Css[n*nb_max:(n+1)*nb_max])
            if input_LocErr != None:
                sigss.append(sigs[n*nb_max:(n+1)*nb_max])
            if Css.shape[1] == max_len:
                isBLs.append(0) # last position correspond to tracks which didn't disapear within maximum track length
            else:
                isBLs.append(1)
    
    do_preds = 1
    args_prod = np.array(list(product(Csss, [0], [ds], [Fs], [TrMat],[pBL], [0],[cell_dims], [nb_substeps], [frame_len], [do_preds], [min_len])), dtype=object)
    args_prod[:, 6] = isBLs
    if input_LocErr != None:
        args_prod[:,1] = sigss
    else:
        args_prod[:,1] = LocErr

    #Cs, LocErr, ds, Fs, TrMat,pBL,isBL, cell_dims, nb_substeps, frame_len, do_preds, min_len = args_prod[3]
    
    if workers >= 2:
        with multiprocessing.Pool(workers) as pool:
            all_pred_Bs = pool.map(Pool_star_P_inter, args_prod)
    else:
        all_pred_Bs = []
        for args in args_prod:
            all_pred_Bs.append(Pool_star_P_inter(args))
    
    all_pred_Bs_dict = {}
    for l in l_list:
        all_pred_Bs_dict[l] = np.empty((0,int(l),nb_states))
    for pred_Bs in all_pred_Bs:
        all_pred_Bs_dict[str(pred_Bs.shape[1])] = np.concatenate((all_pred_Bs_dict[str(pred_Bs.shape[1])],pred_Bs))

    return all_pred_Bs_dict

def extract_params(params, dt, nb_states, nb_substeps, input_LocErr = None, Matrix_type = 1):
    '''
    turn the parameters which differ deppending on the number of states into lists
    ds (diffusion lengths), Fs (fractions), TrMat (substep transiton matrix)
    '''
    param_names = np.sort(list(params.keys()))
    
    LocErr = []
    for param in param_names:
        if param.startswith('LocErr'):
            LocErr.append(params[param].value)

    LocErr = [np.array(LocErr)[None,None]]
    if input_LocErr != None:
        LocErr = []
        if np.any(np.array(list(params.keys())) == 'slope_LocErr'):
            for l in range(len(input_LocErr)):
                LocErr.append(np.clip(input_LocErr[l] * params['slope_LocErr'].value + params['offset_LocErr'].value, 0.000001, np.inf))
        else:
            LocErr = input_LocErr
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
    return LocErr, ds, Fs, TrMat, gammas, betas, pBL

def pool_star_proba(args):
    return Proba_Cs(*args)

def cum_Proba_Cs(params, all_tracks, dt, cell_dims, input_LocErr, nb_states, nb_substeps, frame_len, verbose = 1, workers = 1, Matrix_type = 1, threshold = 0.2, max_nb_states = 120):
    '''
    each probability can be multiplied to get a likelihood of the model knowing
    the parameters LocErr, D0 the diff coefficient of state 0 and F0 fraction of
    state 0, D1 the D coef at state 1, p01 the probability of transition from
    state 0 to 1 and p10 the proba of transition from state 1 to 0.
    here sum the logs(likelihood) to avoid too big numbers
    '''
    t0 = time()
    
    LocErr, ds, Fs, TrMat, gammas, betas, pBL = extract_params(params, dt, nb_states, nb_substeps, input_LocErr, Matrix_type)
    # LocErr[0,0,1] = 0.028
    
    '''
    if input_LocErr != None:
        LocErr = input_LocErr
    else:
        LocErr = [LocErr] # putting LocErr in a list to perform the cartesian product of lists for parallelisation
    '''
    min_len = all_tracks[0].shape[1]
    max_len = all_tracks[-1].shape[1]

    if np.all(TrMat>0) and np.all(Fs>0):# and np.all(ds[1:]-ds[:-1]>=0):
        Cum_P = 0
        Csss = []
        sigss = []
        isBLs = []
        for k in range(len(all_tracks)):
            Css = all_tracks[k]
            if input_LocErr != None:
                sigs = LocErr[k]
            nb_max = 100
            if np.any(gammas != 0):
                nb_max = 1
            for n in range(int(np.ceil(len(Css)/nb_max))):
                Csss.append(Css[n*nb_max:(n+1)*nb_max])
                if input_LocErr != None:
                    sigss.append(sigs[n*nb_max:(n+1)*nb_max])
                if Css.shape[1] == max_len:
                    isBLs.append(0) # last position correspond to tracks which didn't disapear within maximum track length
                else:
                    isBLs.append(1)
        Csss.reverse()
        sigss.reverse()
        
        args_prod = np.array(list(product(Csss, [0], [ds], [Fs], [TrMat], [gammas], [betas], [pBL], [0],[cell_dims], [nb_substeps], [frame_len], [min_len], [threshold], [max_nb_states])), dtype=object)
        args_prod[:, 8] = isBLs
        if input_LocErr != None:
            args_prod[:,1] = sigss
        else:
            args_prod[:,1] = LocErr

        #Cs, LocErr, ds, Fs, TrMat,gammas, betas, pBL,isBL, cell_dims, nb_substeps, frame_len, min_len, threshold, max_nb_states = args_prod[0]
        
        if workers >= 2 and start_method == 'fork':
            with multiprocessing.Pool(workers) as pool:
                LP = pool.map(pool_star_proba, args_prod)
        else:
            LP = []
            for args in args_prod:
                LP.append(pool_star_proba(args))
        
        Cum_P += cp.sum(cp.concatenate(LP))
        Cum_P = asnumpy(Cum_P)
        
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
            #print(Cum_P, q)
    if np.isnan(out):
        out = np.inf
        print('input parameters give nans, you may want to pick more suitable parameter initial values')
    #print(time() - t0)
    return out

def get_params(nb_states = 2,
               steady_state = False,
               vary_params = {'LocErr' : True, 'D0' : True, 'D1' : True, 'F0' : True, 'p01' : True, 'p10' : True, 'pBL' : True},
               estimated_vals = {'LocErr' : 0.025, 'D0' : 1e-20, 'D1' : 0.05, 'F0' : 0.45, 'p01' : 0.05, 'p10' : 0.05, 'pBL' : 0.1},
               min_values = {'LocErr' : 0.007, 'D0' : 1e-12, 'D1' : 0.00001, 'F0' : 0.001, 'p01' : 0.01, 'p10' : 0.01, 'pBL' : 0.01},
               max_values = {'LocErr' : 0.6, 'D0' : 1, 'D1' : 10, 'F0' : 0.999, 'p01' : 1., 'p10' : 1., 'pBL' : 0.99}):
    if 0:
        1
        '''
    if  nb_states == 2:
        if steady_state:
            print(estimated_vals)
            param_kwargs = [{'name' : 'D0', 'value' : estimated_vals['D0'], 'min' : min_values['D0'], 'max' : max_values['D0'], 'vary' : vary_params['D0']},
                            {'name' : 'D1_minus_D0', 'value' : estimated_vals['D1'] - estimated_vals['D0'], 'min' : min_values['D1']-min_values['D0'], 'max' : max_values['D1'], 'vary' : vary_params['D1']},
                            {'name' : 'D1', 'expr' : 'D0 + D1_minus_D0'},
                            {'name' : 'LocErr', 'value' : estimated_vals['LocErr'], 'min' :  min_values['LocErr'],'max' :  max_values['LocErr'], 'vary' : vary_params['LocErr']},
                            {'name' : 'F0', 'value' : estimated_vals['F0'], 'min' :  min_values['F0'], 'max' :  max_values['F0'], 'vary' :  vary_params['F0']},
                            {'name' : 'F1', 'expr' : '1 - F0'},
                            {'name' : 'p01', 'value' : estimated_vals['p01'], 'min' :  min_values['p01'], 'max' :  max_values['p01'], 'vary' :  vary_params['p01']},
                            {'name' : 'p10', 'expr' : 'p01/(1/F0-1)'},
                            {'name' : 'pBL', 'value' : estimated_vals['pBL'], 'min' :  min_values['pBL'], 'max' :  max_values['pBL'], 'vary' : vary_params['pBL']}]
        else :
            param_kwargs = [{'name' : 'D0', 'value' : estimated_vals['D0'], 'min' : min_values['D0'], 'max' : max_values['D0'], 'vary' : vary_params['D0']},
                            {'name' : 'D1_minus_D0', 'value' : estimated_vals['D1'] - estimated_vals['D0'], 'min' : min_values['D1']-min_values['D0'], 'max' : max_values['D1'], 'vary' : vary_params['D1']},
                            {'name' : 'D1', 'expr' : 'D0 + D1_minus_D0' },
                            {'name' : 'LocErr', 'value' : estimated_vals['LocErr'], 'min' :  min_values['LocErr'],'max' :  max_values['LocErr'], 'vary' : vary_params['LocErr']},
                            {'name' : 'F0', 'value' : estimated_vals['F0'], 'min' :  min_values['F0'], 'max' :  max_values['F0'], 'vary' :  vary_params['F0']},
                            {'name' : 'F1', 'expr' : '1 - F0'},
                            {'name' : 'p01', 'value' : estimated_vals['p01'], 'min' :  min_values['p01'], 'max' :  max_values['p01'], 'vary' :  vary_params['p01']},
                            {'name' : 'p10', 'value' : estimated_vals['p10'], 'min' :  min_values['p10'], 'max' :  max_values['p10'], 'vary' : vary_params['p10']},
                            {'name' : 'pBL', 'value' : estimated_vals['pBL'], 'min' :  min_values['pBL'], 'max' :  max_values['pBL'], 'vary' : vary_params['pBL']}]

    elif nb_states == 3:
        if not (len(min_values) == 13 and len(max_values) == 13 and len(estimated_vals) == 13 and len(vary_params) == 13):
            raise ValueError('estimated_vals, min_values, max_values and vary_params should all containing 13 parameters for a 3 states model')

        if steady_state:
            param_kwargs = [{'name' : 'LocErr', 'value' : estimated_vals['LocErr'], 'min' : min_values['LocErr'], 'max' : max_values['LocErr'] , 'vary' : vary_params['LocErr']},
                            {'name' : 'D0', 'value' : estimated_vals['D0'], 'min' : min_values['D0'], 'max' : 0.3, 'brute_step' : 0.04, 'vary' : vary_params['D0']},
                            {'name' : 'D1_minus_D0', 'value' : estimated_vals['D1'] - estimated_vals['D0'], 'min' : 0, 'max' : max_values['D1'], 'brute_step' : 0.04, 'vary' : vary_params['D1']},
                            {'name' : 'D1', 'expr' : 'D0+D1_minus_D0'},
                            {'name' : 'D2_minus_D1', 'value' : estimated_vals['D2'] - estimated_vals['D1'], 'min' : 0, 'max' : max_values['D2'], 'vary' : vary_params['D2']},
                            {'name' : 'D2', 'expr' : 'D1+D2_minus_D1'},
                            {'name' : 'p01', 'value' : estimated_vals['p01'], 'min' : min_values['p01'], 'max' : max_values['p01'], 'vary' : vary_params['p01']},
                            {'name' : 'p02', 'value' : estimated_vals['p02'], 'min' : min_values['p02'], 'max' : max_values['p02'], 'vary' : vary_params['p02']},
                            {'name' : 'p10', 'value' : estimated_vals['p10'], 'min' : min_values['p10'], 'max' : max_values['p10'], 'vary' : vary_params['p10']},
                            {'name' : 'p12', 'value' : estimated_vals['p12'], 'min' : min_values['p12'], 'max' : max_values['p12'], 'vary' : vary_params['p12']},
                            {'name' : 'p20', 'value' : estimated_vals['p20'], 'min' : min_values['p20'], 'max' : max_values['p20'], 'vary' : vary_params['p20']},
                            {'name' : 'p21', 'value' : estimated_vals['p21'], 'min' : min_values['p21'], 'max' : max_values['p21'], 'vary' : vary_params['p21']},
                            {'name' : 'F0', 'expr' : '(p10*(p21+p20)+p20*p12)/((p01)*(p12 + p21) + p02*(p10 + p12 + p21) + p01*p20 + p21*p10 + p20*(p10+p12))'},
                            {'name' : 'F1', 'expr' : '(F0*p01 + (1-F0)*p21)/(p10 + p12 + p21)'},
                            {'name' : 'F2', 'expr' : '1-F0-F1'},
                            {'name' : 'pBL', 'value' : estimated_vals['pBL'], 'min' :  min_values['pBL'], 'max' :  max_values['pBL'], 'vary' : vary_params['pBL']}]
        else:
            param_kwargs = [{'name' : 'LocErr', 'value' : estimated_vals['LocErr'], 'min' : min_values['LocErr'], 'max' : max_values['LocErr'] , 'vary' : vary_params['LocErr']},
                            {'name' : 'D0', 'value' : estimated_vals['D0'], 'min' : min_values['D0'], 'max' : 0.3, 'brute_step' : 0.04, 'vary' : vary_params['D0']},
                            {'name' : 'D1_minus_D0', 'value' : estimated_vals['D1'] - estimated_vals['D0'], 'min' : 0, 'max' : max_values['D1'], 'brute_step' : 0.04, 'vary' : vary_params['D1']},
                            {'name' : 'D1', 'expr' : 'D0+D1_minus_D0'},
                            {'name' : 'D2_minus_D1', 'value' : estimated_vals['D2'] - estimated_vals['D1'], 'min' : 0, 'max' : max_values['D2'], 'vary' : vary_params['D2']},
                            {'name' : 'D2', 'expr' : 'D1+D2_minus_D1'},
                            {'name' : 'p01', 'value' : estimated_vals['p01'], 'min' : min_values['p01'], 'max' : max_values['p01'], 'vary' : vary_params['p01']},
                            {'name' : 'p02', 'value' : estimated_vals['p02'], 'min' : min_values['p02'], 'max' : max_values['p02'], 'vary' : vary_params['p02']},
                            {'name' : 'p10', 'value' : estimated_vals['p10'], 'min' : min_values['p10'], 'max' : max_values['p10'], 'vary' : vary_params['p10']},
                            {'name' : 'p12', 'value' : estimated_vals['p12'], 'min' : min_values['p12'], 'max' : max_values['p12'], 'vary' : vary_params['p12']},
                            {'name' : 'p20', 'value' : estimated_vals['p20'], 'min' : min_values['p20'], 'max' : max_values['p20'], 'vary' : vary_params['p20']},
                            {'name' : 'p21', 'value' : estimated_vals['p21'], 'min' : min_values['p21'], 'max' : max_values['p21'], 'vary' : vary_params['p21']},
                            #{'name' : 'F0', 'value' : estimated_vals['F0'], 'min' : min_values['F0'], 'max' : max_values['F0'], 'vary' : vary_params['F0']},
                            #{'name' : 'F1_minus_F0', 'value' : (estimated_vals['F1'])/(1-estimated_vals['F0']), 'min' : min_values['F1'], 'max' : max_values['F1'], 'vary' : vary_params['F1']},
                            #{'name' : 'F1', 'expr' : 'F1_minus_F0*(1-F0)'},
                            {'name' : 'F0', 'value' : estimated_vals['F0'], 'min' : min_values['F0'], 'max' : max_values['F0'], 'vary' : vary_params['F0']},
                            {'name' : 'F1', 'value' : estimated_vals['F1'], 'min' : min_values['F1'], 'max' : max_values['F1'], 'vary' : vary_params['F1']},
                            {'name' : 'F2', 'expr' : '1-F0-F1'},
                            {'name' : 'pBL', 'value' : estimated_vals['pBL'], 'min' :  min_values['pBL'], 'max' :  max_values['pBL'], 'vary' : vary_params['pBL']}]
        '''
    else :
        param_kwargs = []
        if np.any(np.array(list(estimated_vals.keys())) == 'slope_LocErr'):
            param_kwargs.append({'name' : 'slope_LocErr', 'value' :  estimated_vals['slope_LocErr'], 'min' :  min_values['slope_LocErr'], 'max' :  max_values['slope_LocErr'], 'vary' :  vary_params['slope_LocErr']})
            param_kwargs.append({'name' : 'offset_LocErr', 'value' :  estimated_vals['offset_LocErr'], 'min' :  min_values['offset_LocErr'], 'max' :  max_values['offset_LocErr'], 'vary' :  vary_params['offset_LocErr']})
    
        if np.any(np.array(list(estimated_vals.keys())) == 'LocErr'):
            LocErr = estimated_vals['LocErr']
            # consider LocErr as a parameter or not depending of its format
            if type(LocErr) == float:
                param_kwargs.append({'name' : 'LocErr', 'value' : estimated_vals['LocErr'], 'min' : min_values['LocErr'], 'max' : max_values['LocErr'] , 'vary' : vary_params['LocErr']})
            elif type(LocErr) == np.ndarray or type(LocErr) == list: # if one Localization error parameter per axis
                for s in range(len(LocErr)):
                    param_kwargs.append({'name' : 'LocErr' + str(s), 'value' : estimated_vals['LocErr'][s], 'min' : min_values['LocErr'][s], 'max' : max_values['LocErr'][s] , 'vary' : vary_params['LocErr'][s]})
        Ds = []
        Fs = []
        for param in list(vary_params.keys()):
            if param.startswith('D'):
                Ds.append(param)
            if param.startswith('F'):
                Fs.append(param)
        param_kwargs.append({'name' : 'D0', 'value' : estimated_vals['D0'], 'min' : min_values['D0'], 'max' : 0.3, 'brute_step' : 0.04, 'vary' : vary_params['D0']})
        last_D = 'D0'
        sum_Ds =  estimated_vals['D0']
        expr = 'D0'
        for D in Ds[1:]:
            param_kwargs.append({'name' : D + '_minus_' + last_D, 'value' : estimated_vals[D] - sum_Ds, 'min' : 0, 'max' : max_values[D] , 'vary' : vary_params[D]})
            expr = expr + '+' + D + '_minus_' + last_D
            param_kwargs.append({'name' : D, 'expr' : expr})
            last_D = D
            sum_Ds += estimated_vals[D]
        
        param_kwargs.append({'name' : 'F0', 'value' : estimated_vals['F0'], 'min' : min_values['F0'], 'max' : max_values['F0'], 'brute_step' : 0.04, 'vary' : vary_params['F0']})
        frac = 1-estimated_vals['F0']
        expr = '1-F0'        
        
        for F in Fs[1:len(Ds)-1]:
            param_kwargs.append({'name' : F , 'value' : estimated_vals[F], 'min' : 0.001, 'max' : 0.99 , 'vary' : vary_params[F]})
            frac = frac - 1
            expr = expr + '-' + F
        param_kwargs.append({'name' : 'F'+str(len(Ds)-1), 'expr' : expr})
        
        for param in list(vary_params.keys()):
            if param.startswith('p'):
                param_kwargs.append({'name' : param, 'value' : estimated_vals[param], 'min' : min_values[param], 'max' : max_values[param] , 'vary' : vary_params[param]})
    
    params = Parameters()
    [params.add(**param_kwargs[k]) for k in range(len(param_kwargs))]
    return params

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

from copy import deepcopy

def Maximize(params, args):
    var_params = []
    for param in params:
        if params[param].vary == True:
            var_params.append(param)

    L = - cum_Proba_Cs(params, *args)

    best_L = deepcopy(L)

    history = []
    history.append({'params': params, 'L': best_L})
    steps = {}
    momentum = {}

    for param in var_params:
        steps[param] = 0.2
        momentum[param] = 1

    while not np.all(np.array(list(steps.values()))<0.01):
        for param in var_params:
            if steps[param] > np.mean(list(steps.values()))/5: # condition to avoid some parameters to be tested until obtaining ridiculously small steps. Increase speed and allow those parameters to change if the landscape requires it.
                cur_steps = [- steps[param] * (params[param].value - params[param].min), steps[param] * (params[param].max - params[param].value)]
                cur_params = deepcopy(params)
                cur_params[param].value = params[param].value + cur_steps[momentum[param]]
                cur_L = - cum_Proba_Cs(cur_params, *args)
                if cur_L > L:
                    params[param].value = cur_params[param].value
                    L = cur_L
                    history.append({'params': params, 'L': L, 'steps': steps, 'momentum': momentum})
                    steps[param] = steps[param]*1.2 # if the momentum is good we slightly increase the step size to favor faster convergence
                else:
                    cur_params[param].value = params[param].value + cur_steps[np.abs(momentum[param]-1)]
                    cur_L = -cum_Proba_Cs(cur_params, *args)
                    if cur_L > L:
                        params[param].value = cur_params[param].value
                        L = cur_L
                        momentum[param] = np.abs(momentum[param]-1)
                        history.append({'params': params, 'L': L, 'steps': steps, 'momentum': momentum})
                        steps[param] = steps[param]*0.8
                    else: # if none of the values tested are better, we reduce the step size and redo an iteration
                        steps[param] = steps[param]*0.5
                    #    step = np.min([steps[param] * (params[param].value - params[param].min), steps[param] * (params[param].max - params[param].value)])
                    #    cur_steps = [- step, step]
                    #    cur_params[param].value = params[param].value + cur_steps[momentum[param]]
                    #    cur_L = - cum_Proba_Cs(cur_params, *args)
                    #    if cur_L > L:
                    #        params[param].value = cur_params[param].value
                    #        L = cur_L
                    #        history.append({'params': params, 'L': L})
                    #    else:
                    #        cur_params[param].value = params[param].value + cur_steps[np.abs(momentum[param]-1)]
                    #        cur_L = -cum_Proba_Cs(cur_params, *args)
                    #        if cur_L > L:
                    #            params[param].value = cur_params[param].value
                    #            L = cur_L
                    #            momentum[param] = np.abs(momentum[param]-1)
                    #            history.append({'params': params, 'L': L})
                    #        else: # if none of the values tested are better, we reduce the step size
                    #            steps[param] = steps[param]*0.5
        #print('L', L)
        #q = ''
        #for param in params:
        #    q = q + param + ':' + str(np.round(params[param].value, 6)) + ', '
        #print(q)
        #print('steps:' , steps)

    class Fit(object):
        pass

    fit = Fit()
    fit.params = params
    fit.residual = [L]
    fit.history = history
    
    return fit

#all_tracks = tracks
def param_fitting(all_tracks,
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
                  cell_dims = [1], # list of dimensions limit for the field of view (FOV) of the cell in um, a membrane protein in a typical e-coli cell in tirf would have a cell_dims = [0.5,3], in case of cytosolic protein one should imput the depth of the FOV e.g. [0.3] for tirf or [0.8] for hilo
                  input_LocErr = None, 
                  threshold = 0.2, 
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
    sorted_LocErrs = []
    for l in l_list:
        if len(all_tracks[l]) > 0 :
            sorted_tracks.append((all_tracks[l]))
            if input_LocErr != None:
                sorted_LocErrs.append(input_LocErr[l])
    all_tracks = sorted_tracks
    if len(all_tracks) < 1:
        raise ValueError('No track could be detected. The loaded tracks seem empty. Errors often come from wrong input paths.')

    if input_LocErr != None:
        input_LocErr = sorted_LocErrs
    #if frame_len <= nb_substeps:
    #    print('Warning frame_len has to be at least nb_substeps + 1')
    #    frame_len = nb_substeps + 1
    
    fit = minimize(cum_Proba_Cs, params, args=(all_tracks, dt, cell_dims, input_LocErr, nb_states, nb_substeps, frame_len, verbose, workers, Matrix_type, threshold, max_nb_states), method = method, nan_policy = 'propagate')
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


