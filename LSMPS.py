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
    if typ == 'x':
        EtaDxPos, EtaDxxPos = calculate_derivative(particle, R_e, neighbor_xpos, 'x')
        EtaDxNeg, EtaDxxNeg = calculate_derivative(particle, R_e, neighbor_xneg, 'x')
        return EtaDxPos, EtaDxxPos, EtaDxNeg, EtaDxxNeg
    if typ == 'y':
        EtaDyPos, EtaDyyPos = calculate_derivative(particle, R_e, neighbor_ypos, 'y')
        EtaDyNeg, EtaDyyNeg = calculate_derivative(particle, R_e, neighbor_yneg, 'y')
        return EtaDyPos, EtaDyyPos, EtaDyNeg, EtaDyyNeg
    if typ == 'z':
        EtaDzPos, EtaDzzPos = calculate_derivative(particle, R_e, neighbor_zpos, 'y')
        EtaDzNeg, EtaDzzNeg = calculate_derivative(particle, R_e, neighbor_zneg, 'y')
        return EtaDzPos, EtaDzzPos, EtaDzNeg, EtaDzzNeg
    if typ == 'all':
        EtaDxAll, EtaDyAll, EtaDzAll, EtaDxxAll, EtaDyyAll, EtaDzzAll = calculate_derivative(particle, R_e, particle.neighbor_all, 'all')
        return EtaDxAll, EtaDyAll, EtaDzAll, EtaDxxAll, EtaDyyAll, EtaDzzAll
    
@nb.njit
def calculate_derivative(particle, R_e, neighbor_list, typ):
    N = len(particle.x)
    b_data = [np.array([])] * N
    #index_lsmps = particle.index
    
    #index_lsmps = [particle.index[i] for i in range(len(particle.x)) if particle.boundary[i] == False]
    if typ == 'x' or typ == 'y' or typ == 'z':
        index_lsmps = particle.index[~particle.boundary]
        data_d = nb.typed.List()
        data_d2 = nb.typed.List()

    else:
        index_lsmps = particle.index
        data_dx = nb.typed.List()
        data_dxx = nb.typed.List()
        data_dy = nb.typed.List()
        data_dyy = nb.typed.List()
        data_dz = nb.typed.List()
        data_dzz = nb.typed.List()
    
    for i in index_lsmps:
        H_rs = np.zeros((7,7))
        M = np.zeros((7,7))
        P = np.zeros((7,7))
        b_temp = [np.array([])] * len(neighbor_list[i])
        
        print('Calculating derivative for particle ' + str(i) + '/' + str(N))
        
        neighbor_idx = np.array(neighbor_list[i])
        
        #idx_begin = neighbor_idx[0]
        #idx_end = neighbor_idx[-1]
        #Li = np.average(particle.diameter[idx_begin:idx_end])
        
        Li = np.average(particle.diameter[neighbor_idx])
        
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
            M = M + w_ij * np.matmul(P, P.T)
            b_temp[j] = w_ij * P
        M_inv = np.linalg.inv(M)
        MinvHrs = np.matmul(H_rs, M_inv)
        b_data[i] = b_temp
        
        if typ == 'x' or typ == 'y' or typ == 'z':
            element_d = nb.typed.List()
            element_d2 = nb.typed.List()

        else:
            element_dx = nb.typed.List()
            element_dxx = nb.typed.List()
            element_dy = nb.typed.List()
            element_dyy = nb.typed.List()
            element_dz = nb.typed.List()
            element_dzz = nb.typed.List()
        
        for j in range(len(neighbor_idx)):
            idx_j = neighbor_idx[j]
            #i[indexdx_i].append(idx_j)
            Eta = np.matmul(MinvHrs, b_data[i][j])
            
            if typ == 'x':               
                #EtaDx[idx_i,idx_j] = Eta[1]
                #EtaDxx[idx_i, idx_j] = Eta[2]
                element_d.append(Eta[1])
                element_d2.append(Eta[4])
            elif typ == 'y':
                #EtaDy[idx_i,idx_j] = Eta[1]
                #EtaDyy[idx_i, idx_j] = Eta[2]
                element_d.append(Eta[2])
                element_d2.append(Eta[5])
            elif typ == 'z':
                #EtaDz[idx_i, idx_j] = Eta[1]
                #EtaDzz[idx_i, idx_j] = Eta[2]
                element_d.append(Eta[3])
                element_d2.append(Eta[6])
            else:
                #EtaDx[idx_i,idx_j] = Eta[1]
                #EtaDy[idx_i,idx_j] = Eta[2]
                #EtaDz[idx_i, idx_j] = Eta[3]
                #EtaDxx[idx_i, idx_j] = Eta[4]
                #EtaDyy[idx_i, idx_j] = Eta[5]
                #EtaDzz[idx_i, idx_j] = Eta[6]
                element_dx.append(Eta[1])
                element_dy.append(Eta[2])
                element_dz.append(Eta[3])
                element_dxx.append(Eta[4])
                element_dyy.append(Eta[5])
                element_dzz.append(Eta[6])
                
        if typ == 'x' or typ == 'y' or typ == 'z':
            data_d.append(element_d)
            data_d2.append(element_d2)

        else:
            data_dx.append(element_dx)
            data_dxx.append(element_dx)
            data_dy.append(element_dy)
            data_dyy.append(element_dyy)
            data_dz.append(element_dz)
            data_dzz.append(element_dzz)
            
    if typ == 'all':
        return data_dx, data_dy, data_dz, data_dxx, data_dyy, data_dzz
    elif typ == 'x':
        return data_dx, data_dxx
    elif typ == 'y':
        return data_dy, data_dyy
    elif typ == 'z':
        return data_dz, data_dzz
        
      



