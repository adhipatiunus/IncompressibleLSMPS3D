#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 10:00:13 2022

@author: adhipatiunus
"""

from generate_particles import generate_particles
from neighbor_search import neighbor_search_cell_list
import numpy as np

def LSMPS(particle, R_e, typ):
    if typ == 'x':
        EtaDxPos, EtaDxxPos = calculate_derivative(particle, R_e, particle.neighbor_xpos, 'x')
        EtaDxNeg, EtaDxxNeg = calculate_derivative(particle, R_e, particle.neighbor_xneg, 'x')
        return EtaDxPos, EtaDxxPos, EtaDxNeg, EtaDxxNeg
    if typ == 'y':
        EtaDyPos, EtaDyyPos = calculate_derivative(particle, R_e, particle.neighbor_ypos, 'y')
        EtaDyNeg, EtaDyyNeg = calculate_derivative(particle, R_e, particle.neighbor_yneg, 'y')
        return EtaDyPos, EtaDyyPos, EtaDyNeg, EtaDyyNeg
    if typ == 'z':
        EtaDzPos, EtaDzzPos = calculate_derivative(particle, R_e, particle.neighbor_zpos, 'y')
        EtaDzNeg, EtaDzzNeg = calculate_derivative(particle, R_e, particle.neighbor_zneg, 'y')
        return EtaDzPos, EtaDzzPos, EtaDzNeg, EtaDzzNeg
    if typ == 'all':
        EtaDxAll, EtaDyAll, EtaDzAll, EtaDxxAll, EtaDxyAll, EtaDxzAll, EtaDyyAll, EtaDyzAll, EtaDzzAll = calculate_derivative(particle, R_e, particle.neighbor_all, 'all')
        return EtaDxAll, EtaDyAll, EtaDzAll, EtaDxxAll, EtaDxyAll, EtaDxzAll, EtaDyyAll, EtaDyzAll, EtaDzzAll
    
    
def calculate_derivative(particle, R_e, neighbor_list, typ):
    N = len(particle.x)
    b_data = [np.array([])] * N
    EtaDx   = np.zeros((N, N))
    EtaDy   = np.zeros((N, N))
    EtaDz   = np.zeros((N, N))
    EtaDxx  = np.zeros((N, N))
    EtaDxy  = np.zeros((N, N))
    EtaDxz  = np.zeros((N, N))
    EtaDyy  = np.zeros((N, N))
    EtaDyz  = np.zeros((N, N))
    EtaDzz  = np.zeros((N, N))
    index   = np.zeros((N, N))

    if typ == 'x' or typ == 'y' or typ == 'z':
        index_lsmps = [particle.index[i] for i in range(N) if particle.boundary[i] == False]
    else:
        index_lsmps = particle.index
    #index_lsmps = particle.index
    
    #index_lsmps = [particle.index[i] for i in range(len(particle.x)) if particle.boundary[i] == False]

    for i in index_lsmps:
        H_rs = np.zeros((10,10))
        M = np.zeros((10,10))
        P = np.zeros((10,1))
        b_temp = [np.array([])] * len(neighbor_list[i])
        
        print('Calculating derivative for particle ' + str(i) + '/' + str(N))
        
        neighbor_idx = neighbor_list[i]
        
        idx_begin = neighbor_idx[0]
        idx_end = neighbor_idx[-1]
        Li = np.average(particle.diameter[idx_begin:idx_end])
        
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
        H_rs[5, 5] = Li**-2 # Dxy
        H_rs[6, 6] = Li**-2 # Dxz
        H_rs[7, 7] = 2 * Li**-2 #Dyy
        H_rs[8, 8] = Li**-2 # Dyz
        H_rs[9, 9] = 2 * Li**-2 # Dzz
                
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
            P[5, 0] = p_x * p_y # Dxy
            P[6, 0] = p_x * p_z # Dxz
            P[7, 0] = p_y**2 # Dyy
            P[8, 0] = p_y * p_z # Dyz
            P[9, 0] = p_z**2 # Dzz
            
            if r_ij < R_ij:
                w_ij = (1 - r_ij/R_ij)**2
            else:
                w_ij = 0
            M = M + w_ij * np.matmul(P, P.T)
            b_temp[j] = w_ij * P
        M_inv = np.linalg.inv(M)
        MinvHrs = np.matmul(H_rs, M_inv)
        b_data[i] = b_temp
        
        for j in range(len(neighbor_idx)):
            idx_j = neighbor_idx[j]
            #i[indexdx_i].append(idx_j)
            Eta = np.matmul(MinvHrs, b_data[i][j])
            EtaDx[idx_i,idx_j] = Eta[1]
            EtaDy[idx_i,idx_j] = Eta[2]
            EtaDz[idx_i, idx_j] = Eta[3]
            EtaDxx[idx_i, idx_j] = Eta[4]
            EtaDxy[idx_i, idx_j] = Eta[5]
            EtaDxz[idx_i, idx_j] = Eta[6]
            EtaDyy[idx_i, idx_j] = Eta[7]
            EtaDyz[idx_i, idx_j] = Eta[8]
            EtaDzz[idx_i, idx_j] = Eta[9]
            
    if typ == 'all':
        return EtaDx, EtaDy, EtaDz, EtaDxx, EtaDxy, EtaDxz, EtaDyy, EtaDyz, EtaDzz
    elif typ == 'x':
        return EtaDx, EtaDxx
    elif typ == 'y':
        return EtaDy, EtaDyy
    elif typ == 'z':
        return EtaDz, EtaDzz
        
      



