#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 20:19:38 2022

@author: adhipatiunus
"""

import numpy as np

def calculate_dn_operator(n_bound, dx, dy, dz, normal_x_bound, normal_y_bound, normal_z_bound):
    return np.multiply(dx[:n_bound], normal_x_bound) \
            + np.multiply(dy[:n_bound], normal_y_bound) \
            + np.multiply(dz[:n_bound], normal_z_bound)
            
def calculate_darcy_drag(node_x_inner, node_y_inner, node_z_inner, 
                         I_inner_3d, x_center, y_center, z_center, 
                         R, eta):
    solid = (node_x_inner - x_center)**2 + (node_y_inner - y_center)**2 + (node_z_inner - z_center)**2 <= R**2
    darcy_drag = (1 / eta) * solid.as_type(float)
    ddrag_3d = np.multiply(I_inner_3d, darcy_drag.reshape(-1,1))
    return ddrag_3d
            
def rotinc_pc(particle, dx_all, dy_all, dz_all, dxx_all, dyy_all, dzz_all,
              dx_pos, dy_pos, dz_pos, dxx_pos, dyy_pos, dzz_pos,
              dx_neg, dy_neg, dz_neg, dxx_neg, dyy_neg, dzz_neg,
              u, v, w,
              u_bound, v_bound, w_bound,
              rhs_u, rhs_v, rhs_w, rhs_p,
              poisson_3d,
              dt, t_final, nu,
              x_center, y_center, z_center, R, 
              eta, omega):
    n_total = len(particle.x)
    n_bound = particle.n_bound
    n_inner = n_total - n_bound
    
    # Derivatives are only for the inner particle
    dx = dx_all[n_bound:]
    dy = dy_all[n_bound:]
    dz = dz_all[n_bound:]
    
    dxx = dxx_all[n_bound:]
    dyy = dxx_all[n_bound:]
    dzz = dxx_all[n_bound:]
    
    # Initialize variable
    I_3d = np.eye(n_total)
    ddrag_3d = np.zeros((n_inner, n_total))
    
    # Brinkman Penalization
    solid = (particle.x - x_center)**2 + (particle.y - y_center)**2 + (particle.z - z_center)**2 <= R**2
    u_rot = 1 / eta * omega * (particle.z[solid] - z_center)
    w_rot = -1 / eta * omega * (particle.x[solid] - x_center)
    ddrag_3d = calculate_darcy_drag(particle.x[n_bound:], particle.y[n_bound:], particle.z[n_bound:], I_3d[n_bound:], x_center, y_center, z_center, R, eta)
    
    # First step
    # First substep: convection and diffusion
    t_elapsed = 0
    u_old = u
    v_old = v
    w_old = w
    
    u_inner = u[n_bound:].reshape(n_inner, 1)
    v_inner = v[n_bound:].reshape(n_inner, 1)
    w_inner = w[n_bound:].reshape(n_inner, 1)
    
    # Diffusion
    diff_3d = I_3d[n_bound:] - dt * nu * (dxx + dyy + dzz)
    
    # Convection
    in_LHS_3d = diff_3d + dt * (np.multiply(dx, u_inner) 
                                + np.multiply(dy, v_inner) 
                                + np.multiply(dz, w_inner) 
                                + ddrag_3d)
    
    LHS_3d = np.vstack([u_bound, in_LHS_3d])
    rhs = np.copy(u)
    rhs[:n_bound] = rhs_u
    rhs[solid] += u_rot * dt
    
    u = np.linalg.solve(LHS_3d, rhs)
    
    LHS_3d = np.vstack([v_bound, in_LHS_3d])
    rhs = np.copy(v)
    rhs[:n_bound] = rhs_v
    
    v = np.linalg.solve(LHS_3d, rhs)
    
    LHS_3d = np.vstack([w_bound, in_LHS_3d])
    rhs = np.copy(w)
    rhs[:n_bound] = rhs_w
    rhs[solid] += w_rot * dt
    
    w = np.linalg.solve(LHS_3d, rhs)
    
    # Second substep: pressure correction
    rhs = 1 / dt * (np.dot(dx, u) + np.dot(dy, v) + np.dot(dz, w))
    rhs = np.concatenate((rhs_p, rhs))
    p = np.linalg.solve(poisson_3d, rhs)
    
    u[n_bound:] = u[n_bound:] - dt * np.dot(dx, p)
    v[n_bound:] = v[n_bound:] - dt * np.dot(dx, p)
    w[n_bound:] = w[n_bound:] - dt * np.dot(dz, p)
    
    t_elapsed += dt
    
    # Diffusion
    diff_3d = I_3d[n_bound:] - 2 / 3 * dt * nu * (dxx + dyy + dzz)
    
    while t_elapsed < t_final:
        # 1st substep
        # Convection - diffusion
        u_inner = u[n_bound:].reshape(n_inner, 1)
        v_inner = v[n_bound:].reshape(n_inner, 1)
        w_inner = w[n_bound:].reshape(n_inner, 1)
        
        in_LHS_3d = diff_3d + 2 / 3 * dt * (
                    dx_neg * np.maximum(u_inner, 0) + dx_pos * np.minimum(u_inner, 0) +
                    dy_neg * np.maximum(v_inner, 0) + dy_pos * np.minimum(v_inner, 0) +
                    dz_neg * np.maximum(v_inner, 0) + dz_pos * np.minimum(v_inner, 0) +
                    ddrag_3d)
        
        LHS_3d = np.vstack([u_bound. in_LHS_3D])
        
        rhs = 4 / 3 * u[n_bound:] - 1 / 3 * u_old[n_bound:] + 2 / 3 * dt * dx @ p
        rhs = np.concatenate((rhs_u, rhs))
        rhs[solid] += 2 / 3 * u_rot * dt
        
        u_old = u.copy()
        
        u = np.linalg.solve(LHS_3d, rhs)
        
        LHS_3d = np.vstack([v_bound. in_LHS_3D])
        
        rhs = 4 / 3 * v[n_bound:] - 1 / 3 * v_old[n_bound:] + 2 / 3 * dt * dy @ p
        rhs = np.concatenate((rhs_v, rhs))
        
        v_old = v.copy()
        
        v = np.linalg.solve(LHS_3d, rhs)
        
        LHS_3d = np.vstack([w_bound. in_LHS_3D])
        
        rhs = 4 / 3 * w[n_bound:] - 1 / 3 * w_old[n_bound:] + 2 / 3 * dt * dz @ p
        rhs = np.concatenate((rhs_w, rhs))
        rhs[solid] += 2 / 3 * w_rot * dt
        
        w_old = w.copy()
        
        w = np.linalg.solve(LHS_3d, rhs)
        
        # 2nd substep
        # Pressure correction
        divV = 3 / (2 * dt) * (dx_all @ u + dy_all @ v + dz_all @ w)
        rhs = divV[n_bound:]
        rhs = np.concatenate((rhs_p, rhs))
        
        p_old = p.copy()
        phi = np.solve(poisson_3d, rhs)
        p = phi + p_old - nu * divV
        
        u[n_bound:] = u[n_bound:] - 2 / 3 * dt * np.dot(dx, p)
        v[n_bound:] = v[n_bound:] - 2 / 3 * dt * np.dot(dx, p)
        w[n_bound:] = w[n_bound:] - 2 / 3 * dt * np.dot(dz, p)
        
        t_elapsed += dt
        
        
        