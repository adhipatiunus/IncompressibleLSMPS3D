#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 19:20:38 2022

@author: adhipatiunus
"""
import numpy as np
import matplotlib.pyplot as plt

from particle import Particle
from generate_particles import generate_particles
from neighbor_search import neighbor_search_cell_list
from LSMPS import LSMPS
from pressure_correction import calculate_dn_operator

x_min = 0
x_max = 10
x_center = 5
y_min = 0
y_max = 10
y_center = 5
z_min = 0
z_max = 10
z_center = 5
sigma = 1
cell_size = 2.1 * sigma
R_e = 2.1
R = 0.5

particle = Particle()
particle, normal_x_bound, normal_y_bound, normal_z_bound = generate_particles(x_min, x_max, y_min, y_max, z_min, z_max, sigma, R)
N = len(particle.x)


#%%
#=============================================================================#
# Checking geometry
#=============================================================================#
# Checking nodes on middle of x axis
mid_x = abs(particle.x - x_center)<1e-8
plt.scatter(particle.y[mid_x], particle.z[mid_x], particle.diameter[mid_x])
plt.axis('equal')

#%%
#=============================================================================#
# Checking sphere volume
#=============================================================================#
solid = (particle.x - x_center)**2 + (particle.y - y_center)**2 + (particle.z - z_center)**2 <= R**2
diameter_solid = particle.diameter[solid]
vol_true = 4 / 3 * np.pi * R**3
vol_computed = np.sum(diameter_solid**3)
print('Vol_true=',vol_true,'\t Vol_computed=', vol_computed)

#%%
#=============================================================================#
# Perform neighbor search
#=============================================================================#
print('Neighbor search')
neighbor_search_cell_list(particle, cell_size, y_max, y_min, x_max, x_min, z_max, z_min)

#%%
#=============================================================================#
# LSMPS derivative
#=============================================================================#
# Calculating x derivative
# Upwind x derivative
print('Calculating upwind x derivative')
EDxPos, DxxPos, DxNeg, DxxNeg = LSMPS(particle, R_e, 'x')

# Upwind y derivative
print('Calculating upwind y derivative')
DyPos, DyyPos, DyNeg, DyyNeg = LSMPS(particle, R_e, 'y')

# Upwind z derivative
print('Calculating upwind z derivative')
DzPos, DzzPos, DzNeg, DzzNeg = LSMPS(particle, R_e, 'z')

# CDS derivative
print('Calculating CDS derivative')
DxAll, DyAll, DzAll, DxxAll, DxyAll, DxzAll, DyyAll, DyzAll, DzzAll = LSMPS(particle, R_e, 'all')

#%%
#=============================================================================#
# Initializing boundary condition
#=============================================================================#
n_total = len(particle.x)

boundary_particle = [particle.index[i] for i in range(n_total) if particle.boundary[i] == True]
n_bound = len(boundary_particle)

V0_3d = np.zeros((n_total, 3))

I_3d = np.eye(n_total)

# LHS boundary condition
# Dirichlet BC for velocity
u_bound = I_3d[:n_bound]
v_bound = u_bound.copy()
w_bound = u_bound.copy()
# Neumann BC for pressure
p_bound = calculate_dn_operator(n_bound, DxAll, DyAll, DzAll, normal_x_bound, normal_y_bound, normal_z_bound)

# RHS boundary condition
rhs_u = np.zeros(n_bound)
rhs_v = np.zeros(v_bound)
rhs_w = np.zeros(n_bound)
rhs_p = np.zeros(n_bound)

#%%
#=============================================================================#
# Boundary on each domain side
#=============================================================================#
# 1. East
idx_begin = 0
idx_end = particle.n_east

u_bound[idx_begin:idx_end] = p_bound[idx_begin:idx_end] # Neumann velocity
p_bound[idx_begin:idx_end] = I_3d[idx_begin:idx_end] # Dirichlet pressure

# 2. West
idx_begin = idx_end
idx_end = particle.n_west

rhs_u[idx_begin:idx_end] = 1

# 3. North
idx_begin = idx_end
idx_end = particle.n_north

rhs_u[idx_begin:idx_end] = 1

# 4. North
idx_begin = idx_end
idx_end = particle.n_south

rhs_u[idx_begin:idx_end] = 1

# 5. Top
idx_begin = idx_end
idx_end = particle.n_top

rhs_u[idx_begin:idx_end] = 1

# 6. Bottom
idx_begin = idx_end
idx_end = particle.n_bottom

rhs_u[idx_begin:idx_end] = 1
#%%
#=============================================================================#
# Solving Poisson equation
#=============================================================================#
poisson_3d = DxxAll + DyyAll + DzzAll




