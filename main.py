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
from neighbor_search_verlet import multiple_verlet
from LSMPS import LSMPS
from pressure_correction import calculate_dn_operator
import json
import numba as nb
import time

x_min = 0
x_max = 4
x_center = 2
y_min = 0
y_max = 4
y_center = 2
z_min = 0
z_max = 4
z_center = 2
sigma = 1
cell_size = 3.5 * (sigma / 2)
R_e = 3.5
R = 0.5

particle = Particle()
particle, normal_x_bound, normal_y_bound, normal_z_bound = generate_particles(x_min, x_max, y_min, y_max, z_min, z_max, sigma, R)
N = len(particle.x)

#%%
NNPS = 'search'

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
#neighbor_search_cell_list(particle, cell_size, y_max, y_min, x_max, x_min, z_max, z_min)
n_bound = particle.n_bound
h = particle.diameter
rc = np.concatenate((h[:n_bound] * R_e, h[n_bound:] * R_e))
upwind = True
nodes_3d = np.concatenate((particle.x.reshape(-1,1), particle.y.reshape(-1,1), particle.z.reshape(-1,1)), axis = 1)
if NNPS == 'search':
    neighbor_all, neighbor_xneg, neighbor_xpos, neighbor_yneg, neighbor_ypos, neighbor_zneg, neighbor_zpos = multiple_verlet(particle, nodes_3d, n_bound, rc, upwind)
else:
    start = time.time()
    with open('neighbor_all.txt', 'r') as file:
        neighbor_all = json.load(file)
    print('loaded neighbor in all direction')
    with open('neighbor_xneg.txt', 'r') as file:
        neighbor_xneg = json.load(file)
    with open('neighbor_xpos.txt', 'r') as file:
        neighbor_xpos = json.load(file)
    print('loaded neighbor in x direction')
    with open('neighbor_yneg.txt', 'r') as file:
        neighbor_yneg = json.load(file)
    with open('neighbor_ypos.txt', 'r') as file:
        neighbor_ypos = json.load(file)
    print('loaded neighbor in y direction')
    with open('neighbor_zneg.txt', 'r') as file:
        neighbor_zneg = json.load(file)
    with open('neighbor_zpos.txt', 'r') as file:
        neighbor_zpos = json.load(file)
    print('loaded neighbor in z direction')
    print('Loaded neighbor in ' + str(time.time() - start) + 's')
#%%
neighbor_len = np.array([len(n) for n in neighbor_all])
neighbor_all_flattened = np.array([i for sub in neighbor_all for i in sub], dtype = np.int64)
index_end = np.array([np.sum(neighbor_len[:i]) for i in range(1, N+1)], dtype = np.int64)
#print('Converting list to typed list')
#neighbor_all = nb.typed.List(neighbor_all)
#neighbor_xneg = nb.typed.List(neighbor_xneg)
#neighbor_xpos = nb.typed.List(neighbor_xpos)
#neighbor_yneg = nb.typed.List(neighbor_yneg)
#neighbor_ypos = nb.typed.List(neighbor_ypos)
#neighbor_zneg = nb.typed.List(neighbor_zneg)
#neighbor_zpos = nb.typed.List(neighbor_zpos)
    
#%%
#=============================================================================#
# LSMPS derivative
#=============================================================================#
from LSMPS import LSMPS_test
# Calculating x derivative
# Upwind x derivative
#print('Calculating upwind x derivative')
#DxPos, DxxPos, DxNeg, DxxNeg = LSMPS(particle, neighbor_all, neighbor_xneg, neighbor_xpos, neighbor_yneg, neighbor_ypos, neighbor_zneg, neighbor_zpos, R_e, 'x')

# Upwind y derivative
#print('Calculating upwind y derivative')
#DyPos, DyyPos, DyNeg, DyyNeg = LSMPS(particle, neighbor_all, neighbor_xneg, neighbor_xpos, neighbor_yneg, neighbor_ypos, neighbor_zneg, neighbor_zpos, R_e, 'y')

# Upwind z derivative
#print('Calculating upwind z derivative')
#DzPos, DzzPos, DzNeg, DzzNeg = LSMPS(particle, neighbor_all, neighbor_xneg, neighbor_xpos, neighbor_yneg, neighbor_ypos, neighbor_zneg, neighbor_zpos, R_e, 'z')

# CDS derivative
print('Calculating CDS derivative')
DxAll, DyAll, DzAll, DxxAll, DyyAll, DzzAll = LSMPS_test(particle, neighbor_all_flattened, index_end, R_e, 'all')

#%%
"""
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
poisson_3d = np.vstack([p_bound, poisson_3d[n_bound:]])
"""


