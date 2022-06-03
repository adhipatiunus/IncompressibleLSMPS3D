#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 10:00:13 2022

@author: adhipatiunus
"""
#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
import numba as nb
import numpy as np
from scipy import sparse

def LSMPS(particle, neighbor_all, neighbor_xneg, neighbor_xpos, neighbor_yneg, neighbor_ypos, neighbor_zneg, neighbor_zpos, R_e, typ):
    if typ == 'all':
        EtaDxAll, EtaDyAll, EtaDzAll, EtaDxxAll, EtaDyyAll, EtaDzzAll = calculate_derivative_CDS(particle, R_e, neighbor_all, 'all')
        return EtaDxAll, EtaDyAll, EtaDzAll, EtaDxxAll, EtaDyyAll, EtaDzzAll
@nb.njit   
def LSMPS_test(particle, neighbor_all, index_end, R_e, typ):
    if typ == 'all':
        EtaDxAll, EtaDyAll, EtaDzAll, EtaDxxAll, EtaDyyAll, EtaDzzAll = calculate_derivative_CDS(particle, R_e, neighbor_all, index_end, 'all')
        return EtaDxAll, EtaDyAll, EtaDzAll, EtaDxxAll, EtaDyyAll, EtaDzzAll
    
@nb.njit
def calculate_derivative_CDS(particle, R_e, neighbor_list, index_end, typ):
    N = len(particle.x)
    #b_data = [np.array([])] * N
    #index_lsmps = particle.index
    
    #index_lsmps = [particle.index[i] for i in range(len(particle.x)) if particle.boundary[i] == False]
    if typ == 'x' or typ == 'y' or typ == 'z':
        index_lsmps = particle.index[~particle.boundary]
    else:
        index_lsmps = particle.index
        
    n_data = len(neighbor_list)
    
    data_dx = np.ones(n_data)
    data_dxx = np.ones(n_data)
    data_dy = np.ones(n_data)
    data_dyy = np.ones(n_data)
    data_dz = np.ones(n_data)
    data_dzz = np.ones(n_data)
    
    start = int(0)
    
    for i in index_lsmps:
        end = index_end[i]
        
        n_neighbor = end - start
        
        H_rs = np.zeros((7,7))
        M = np.zeros((7,7))
        P = np.zeros((7,1))
        b_temp = [np.zeros((7,1))] * n_neighbor
        
        print('Calculating derivative for particle ' + str(i) + '/' + str(N))
        
        neighbor_idx = neighbor_list[start:end]
        
        #print(np.dtype(neighbor_idx[0]))
        
        #neighbor_idx = neighbor_list[start:end]
        
        #idx_begin = neighbor_idx[0]
        #idx_end = neighbor_idx[-1]
        #Li = np.average(particle.diameter[idx_begin:idx_end])
        
        Li = np.mean(particle.diameter[neighbor_idx])
        
        #Li = 0
        #for j in range(start, end):
        #    Li += particle.diameter[j]
        #Li = Li / (end - start)
        
        idx_i = i
        x_i = particle.x[idx_i]
        y_i = particle.y[idx_i]
        z_i = particle.z[idx_i]
        R_i = R_e * Li
        
        H_rs[0, 0] = 1
        H_rs[1, 1] = Li**-1 # Dx
        H_rs[2, 2] = Li**-1 # Dy
        H_rs[3, 3] = Li**-1 # Dz
        H_rs[4, 4] = 2 * Li**-2 # Dxx
        H_rs[5, 5] = 2 * Li**-2 #Dyy
        H_rs[6, 6] = 2 * Li**-2 # Dzz
        
        for j in range(len(neighbor_idx)):
            idx_j = neighbor_idx[j]
            x_j = particle.x[idx_j]
            y_j = particle.y[idx_j]
            z_j = particle.z[idx_j]
            
            R_j = R_e * particle.diameter[idx_j]
            
            R_ij = (R_i + R_j) / 2
            x_ij = x_j - x_i
            y_ij = y_j - y_i
            z_ij = z_j - z_i
            r_ij = np.sqrt(x_ij**2 + y_ij**2 + z_ij**2)
             
            p_x = x_ij / Li
            p_y = y_ij / Li
            p_z = z_ij / Li
            
            P[0, 0] = 1.0
            P[1, 0] = p_x # Dx
            P[2, 0] = p_y # Dy
            P[3, 0] = p_z # Dz
            P[4, 0] = p_x**2 # Dxx
            P[5, 0] = p_y**2 # Dyy
            P[6, 0] = p_z**2 # Dzz
                       
            if r_ij < R_ij:
                w_ij = (1 - r_ij/R_ij)**2
            else:
                w_ij = 0
            M = M + w_ij * np.dot(P, P.T)
            b_temp[j] = w_ij * P
        M_inv = np.linalg.inv(M)
        MinvHrs = np.dot(H_rs, M_inv)
        """
        for j in range(start, end):
            idx_j = neighbor_idx[j]
            #i[indexdx_i].append(idx_j)
            Eta = np.dot(MinvHrs, b_temp[j])
            
            data_dx[j] = Eta[1] 
            data_dxx[j] = Eta[2]
            data_dy[j] = Eta[3]
            data_dyy[j] = Eta[4]
            data_dz[j] = Eta[5]
            data_dzz[j] = Eta[6]
        """
    return data_dx, data_dy, data_dz, data_dxx, data_dyy, data_dzz  
      



