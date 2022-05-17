#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 15:13:03 2022
@author: adhipatiunus
"""

from particle import Particle
import matplotlib.pyplot as plt
import numpy as np

def generate_particles(x_min, x_max, y_min, y_max, z_min, z_max, sigma, R):
    h1 = 1
    h2 = 1/2
    h3 = 1/4
    h4 = 1/8
    h5 = 1/16
    h6 = 1/32
    h7 = 1/64
    
    particle = Particle()
    
    h = h2 * sigma
    
    lx = x_max - x_min
    ly = y_max - y_min
    lz = z_max - z_min
    
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2
    
    nx = int(lx / h) + 1
    ny = int(ly / h) + 1
    nz = int(lz / h) + 1
    
    # Generate Boundary Particle
    
    # 1. East
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)
    
    y_3d, z_3d = np.meshgrid(y, z)
    y_east = y_3d.flatten()
    z_east = z_3d.flatten()
    x_east = x_min * np.ones_like(y_east)
    
    n_east = x_east.shape[0]
    normal_x_east = np.ones(n_east)
    normal_y_east = np.zeros(n_east)
    normal_z_east = np.zeros(n_east)
    
    particle.n_east = len(x_east)
    
    # 2. West
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)
    
    y_3d, z_3d = np.meshgrid(y, z)
    y_west = y_3d.flatten()
    z_west = z_3d.flatten()
    x_west = x_max * np.ones_like(y_east)
    
    n_west = x_west.shape[0]
    normal_x_west = -1 * np.ones(n_east)
    normal_y_west = np.zeros(n_east)
    normal_z_west = np.zeros(n_east)
    
    particle.n_west = particle.n_east + len(x_west)
    
    # 3. North
    x = np.linspace(x_min + h, x_max - h, nx-2)
    z = np.linspace(z_min, z_max, nz)
    
    x_3d, y_3d = np.meshgrid(x, z)
    x_north = x_3d.flatten()
    z_north = y_3d.flatten()
    y_north = y_max * np.ones_like(x_north)  
    
    n_north = y_north.shape[0]
    normal_x_north = np.zeros(n_north)
    normal_y_north = np.ones(n_north)
    normal_z_north = np.zeros(n_north)
    
    particle.n_north = particle.n_west + len(x_north)
    
    # 4. South
    x = np.linspace(x_min + h, x_max - h, nx-2)
    z = np.linspace(z_min, z_max, nz)
    
    x_3d, y_3d = np.meshgrid(x, z)
    x_south = x_3d.flatten()
    z_south = y_3d.flatten()
    y_south = y_min * np.ones_like(x_south)  
    
    n_south = y_south.shape[0]
    normal_x_south = np.zeros(n_south)
    normal_y_south = np.ones(n_south)
    normal_z_south = np.zeros(n_south)
    
    particle.n_south = particle.n_north + len(x_south)
    
    # 5. Top
    x = np.linspace(x_min + h, x_max - h, ny - 2)
    y = np.linspace(y_min + h, y_max - h, nz - 2)
    
    x_3d, y_3d = np.meshgrid(x, y)
    x_top = x_3d.flatten()
    y_top = y_3d.flatten()
    z_top = z_max * np.ones_like(y_top) 
    
    n_top = z_top.shape[0]
    normal_x_top = np.zeros(n_top)
    normal_y_top = np.zeros(n_top)
    normal_z_top = -1 * np.ones(n_top)
    
    particle.n_top = particle.n_south + len(x_top)
    
    # 6. Bottom
    x = np.linspace(x_min + h, x_max - h, ny - 2)
    y = np.linspace(y_min + h, y_max - h, nz - 2)
    
    x_3d, y_3d = np.meshgrid(x, y)
    x_bottom = x_3d.flatten()
    y_bottom = y_3d.flatten()
    z_bottom = z_min * np.ones_like(y_top) 
    
    particle.n_bottom = particle.n_top + len(x_bottom)
    
    n_bottom = z_bottom.shape[0]
    normal_x_bottom = np.zeros(n_bottom)
    normal_y_bottom = np.zeros(n_bottom)
    normal_z_bottom = np.ones(n_bottom)
    
    normal_x_bound = np.concatenate((normal_x_east, normal_x_west, normal_x_north, normal_x_south, normal_x_top, normal_x_bottom))
    normal_y_bound = np.concatenate((normal_y_east, normal_y_west, normal_y_north, normal_y_south, normal_y_top, normal_y_bottom))
    normal_z_bound = np.concatenate((normal_z_east, normal_z_west, normal_z_north, normal_z_south, normal_z_top, normal_z_bottom))
    
    particle.x = np.concatenate((x_east, x_west, x_north, x_south, x_top, x_bottom))
    particle.y = np.concatenate((y_east, y_west, y_north, y_south, y_top, y_bottom))
    particle.z = np.concatenate((z_east, z_west, z_north, z_south, z_top, z_bottom))
    
    
    particle.n_bound = len(particle.x)
    n_bound = particle.n_bound
    particle.diameter = h * np.ones(n_bound)
    
    # Inner Sphere
    # First layer
    h = h5
    R_in = 0
    R_out = R - 2 * h
    node_x, node_y, node_z, sp = generate_node_spherical(x_center, y_center, z_center, R_in, R_out, h)
    
    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.z = np.concatenate((particle.z, node_z))
    particle.diameter = np.concatenate((particle.diameter, sp))
    
    # Second layer
    h = h6
    R_in = R_out
    R_out = R - 1 * h
    node_x, node_y, node_z, sp = generate_node_spherical(x_center, y_center, z_center, R_in, R_out, h)

    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.z = np.concatenate((particle.z, node_z))
    particle.diameter = np.concatenate((particle.diameter, sp))

    # Third layer    
    h = h7
    R_in = R_out
    R_out = R
    node_x, node_y, node_z, sp = generate_node_spherical(x_center, y_center, z_center, R_in, R_out, h)

    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.z = np.concatenate((particle.z, node_z))
    particle.diameter = np.concatenate((particle.diameter, sp))
    
    # Outside sphere
    # First layer
    h = h7
    n_layer = 3
    R_in = R_out
    R_out = R + n_layer * h
    node_x, node_y, node_z, sp = generate_node_spherical(x_center, y_center, z_center, R_in, R_out, h)

    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.z = np.concatenate((particle.z, node_z))
    particle.diameter = np.concatenate((particle.diameter, sp))
    
    # Second layer
    h = h6
    n_layer = 3
    R_in = R_out
    R_out = R + n_layer * h
    node_x, node_y, node_z, sp = generate_node_spherical(x_center, y_center, z_center, R_in, R_out, h)

    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.z = np.concatenate((particle.z, node_z))
    particle.diameter = np.concatenate((particle.diameter, sp))
    
    # Intermediate
    # Inner boundary: circle
    # Outer boundary: rectangle
    h = h5
    n_layer = 3
    X_MIN = x_min + h
    X_MAX = x_max - h
    Y_MIN = y_min + h
    Y_MAX = y_max - h
    Z_MIN = z_min + h
    Z_MAX = z_max - h
    
    nx = int((X_MAX - X_MIN) / h) + 1
    ny = int((Y_MAX - Y_MIN) / h) + 1
    nz = int((Z_MAX - Z_MIN) / h) + 1
    
    x = np.linspace(X_MIN, X_MAX, nx)
    y = np.linspace(Y_MIN, Y_MAX, ny)
    z = np.linspace(Z_MIN, Z_MAX, nz)
    
    x_3d, y_3d, z_3d = np.meshgrid(x, y, z)
    
    node_x = x_3d.flatten()
    node_y = y_3d.flatten()
    node_z = z_3d.flatten()
    
    delete_inner = (node_x - x_center)**2 + (node_y - y_center)**2 + (node_z - z_center)**2 <= R_out
    
    X_MIN = (x_center - R_out) - n_layer * h
    X_MAX = (x_center + R_out) + n_layer * h
    Y_MIN = (y_center - R_out) - n_layer * h
    Y_MAX = (y_center + R_out) + n_layer * h
    Z_MIN = (z_center - R_out) - n_layer * h
    Z_MAX = (z_center + R_out) + n_layer * h
    
    delete_outer = (node_x < X_MIN) + (node_x > X_MAX) + (node_y < Y_MIN) + (node_y > Y_MAX) + (node_z < Z_MIN) + (node_z > Z_MAX)
    delete_node = delete_inner + delete_outer
    
    node_x = node_x[~delete_node]
    node_y = node_y[~delete_node]
    node_z = node_z[~delete_node]
    sp = h * np.ones_like(node_x)
    
    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.z = np.concatenate((particle.z, node_z))
    particle.diameter = np.concatenate((particle.diameter, sp))
    
    # Box of nodes
    # First box
    h = h4
    n_layer = 3
    x_bound_min = X_MIN
    x_bound_max = X_MAX
    y_bound_min = Y_MIN
    y_bound_max = Y_MAX
    z_bound_min = Z_MIN
    z_bound_max = Z_MAX
    
    node_x, node_y, node_z, sp = generate_node_box(x_min, x_max, y_min, y_max, z_min, z_max, x_bound_min, x_bound_max, y_bound_min, y_bound_max, z_bound_min, z_bound_max, n_layer, h)
    
    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.z = np.concatenate((particle.z, node_z))
    particle.diameter = np.concatenate((particle.diameter, sp))
    
    x_bound_min = x_bound_min - n_layer * h
    x_bound_max = x_bound_max + n_layer * h
    y_bound_min = y_bound_min - n_layer * h
    y_bound_max = y_bound_max + n_layer * h
    z_bound_min = z_bound_min - n_layer * h
    z_bound_max = z_bound_max + n_layer * h
    
    # Second box
    h = h3
    n_layer = 3
    node_x, node_y, node_z, sp = generate_node_box(x_min, x_max, y_min, y_max, z_min, z_max, x_bound_min, x_bound_max, y_bound_min, y_bound_max, z_bound_min, z_bound_max, n_layer, h)
    
    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.z = np.concatenate((particle.z, node_z))
    particle.diameter = np.concatenate((particle.diameter, sp))
    
    x_bound_min = x_bound_min - n_layer * h
    x_bound_max = x_bound_max + n_layer * h
    y_bound_min = y_bound_min - n_layer * h
    y_bound_max = y_bound_max + n_layer * h
    z_bound_min = z_bound_min - n_layer * h
    z_bound_max = z_bound_max + n_layer * h
    
    # Third box
    h = h2
    
    nx = int((x_max - x_min) / h) + 1
    ny = int((y_max - y_min) / h) + 1
    nz = int((z_max - z_min) / h) + 1
    
    x = np.linspace(x_min + h, x_max - h, nx)
    y = np.linspace(y_min + h, y_max - h, ny)
    z = np.linspace(z_min + h, z_max - h, nz)
    
    x_3d, y_3d, z_3d = np.meshgrid(x, y, z)
    
    node_x = x_3d.flatten()
    node_y = y_3d.flatten()
    node_z = z_3d.flatten()
    
    delete_inner = (node_x >= x_bound_min) * (node_x <= x_bound_max) \
                    *(node_y >= y_bound_min) * (node_y <= y_bound_max) \
                    *(node_z >= z_bound_min) * (node_z <= z_bound_max)
                    
    node_x = node_x[~delete_inner]
    node_y = node_y[~delete_inner]
    node_z = node_z[~delete_inner]
    sp = h * np.ones_like(node_x)
    
    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.z = np.concatenate((particle.z, node_z))
    particle.diameter = np.concatenate((particle.diameter, sp))
    
    N = len(particle.x)
    particle.index = np.arange(0, N)
    
    particle.boundary = np.full(N, False)
    particle.boundary[:n_bound] = True
    
    return particle, normal_x_bound, normal_y_bound, normal_z_bound
    
def generate_node_spherical(x_center, y_center, z_center, R_in, R_out, h):
    x_min = x_center - 2 * R_out
    x_max = x_center + 2 * R_out
    y_min = y_center - 2 * R_out
    y_max = y_center + 2 * R_out
    z_min = z_center - 2 * R_out
    z_max = z_center + 2 * R_out
    
    lx = x_max - x_min
    ly = y_max - y_min
    lz = z_max - z_min

    nx = int(lx / h) + 1
    ny = int(ly / h) + 1
    nz = int(lz / h) + 1
    
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)
    
    x_3d, y_3d, z_3d = np.meshgrid(x, y, z)
    
    node_x = x_3d.flatten()
    node_y = y_3d.flatten()
    node_z = z_3d.flatten()
    
    delete_inner = (node_x - x_center)**2 + (node_y - y_center)**2 + (node_z - z_center)**2 <= R_in
    delete_outer = (node_x - x_center)**2 + (node_y - y_center)**2 + (node_z - z_center)**2 > R_out
    delete_node = delete_inner + delete_outer
    
    node_x = node_x[~delete_node]
    node_y = node_y[~delete_node]
    node_z = node_z[~delete_node]
    sp = h * np.ones_like(node_x)
    
    return node_x, node_y, node_z, sp

def generate_node_box(x_min, x_max, y_min, y_max, z_min, z_max, x_bound_min, x_bound_max, y_bound_min, y_bound_max, z_bound_min, z_bound_max, n_layer, h):    
    nx = int((x_max - x_min) / h) + 1
    ny = int((y_max - y_min) / h) + 1
    nz = int((z_max - z_min) / h) + 1
    
    safety = 1e-10
    
    x_outer_min = x_bound_min - n_layer * h
    x_outer_max = x_bound_max + n_layer * h
    y_outer_min = y_bound_min - n_layer * h
    y_outer_max = y_bound_max + n_layer * h
    z_outer_min = z_bound_min - n_layer * h
    z_outer_max = z_bound_max + n_layer * h
    
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)
    
    x_3d, y_3d, z_3d = np.meshgrid(x, y, z)
    
    node_x = x_3d.flatten()
    node_y = y_3d.flatten()
    node_z = z_3d.flatten()
    
    delete_inner = (node_x >= x_bound_min) * (node_x <= x_bound_max) \
                    *(node_y >= y_bound_min) * (node_y <= y_bound_max) \
                    *(node_z >= z_bound_min) * (node_z <= z_bound_max)
                    
    delete_outer = (node_x < x_outer_min - safety) + (node_x > x_outer_max + safety) \
                    + (node_y < y_outer_min - safety) + (node_y > y_outer_max + safety) \
                    + (node_z < z_outer_min - safety) + (node_z > z_outer_max + safety)
                    
    delete = delete_inner + delete_outer
            
    node_x = node_x[~delete]
    node_y = node_y[~delete]
    node_z = node_z[~delete]
    sp = h * np.ones_like(node_x)
    
    return node_x, node_y, node_z, sp
    
def generate_particle_multires(x_min, x_max, y_min, y_max, x_center, y_center, R, sigma):
    h1 = 1/64
    h2 = 1/32
    h3 = 1/16
    h4 = 1/8 
    h5 = 1/4 
    h6 = 1/2  

    h = h1 * sigma
    lx = x_max - x_min
    ly = y_max - y_min

    nx = int(lx / h) + 1
    ny = int(ly / h) + 1

    particle = Particle()
    
    # West Boundary
    y_west = np.linspace(y_min, y_max, ny)
    x_west = np.linspace(x_min, x_min + h, 2)
    X_west, Y_west = np.meshgrid(x_west, y_west)
    X_west = X_west.flatten()
    Y_west = Y_west.flatten()
    sp_west = sigma * np.ones_like(X_west)

    # East Boundary
    y_east = np.linspace(y_min, y_max, ny)
    x_east = np.linspace(x_max, x_max - h, 2)
    X_east, Y_east = np.meshgrid(x_east, y_east)
    X_east = X_east.flatten()
    Y_east = Y_east.flatten()
    sp_east = sigma * np.ones_like(X_east)

    # North Boundary
    x_north = np.linspace(x_min + 2 * h, x_max - 2 * h, nx - 4)
    y_north = np.linspace(y_max, y_max - h, 2)
    X_north, Y_north = np.meshgrid(x_north, y_north)
    X_north = X_north.flatten()
    Y_north = Y_north.flatten()
    sp_north = sigma * np.ones_like(X_north)

    # South Boundary
    x_south = np.linspace(x_min + 2 * h, x_max - 2 * h, nx - 4)
    y_south = np.linspace(y_min, y_min + h, 2)
    X_south, Y_south = np.meshgrid(x_south, y_south)
    X_south = X_south.flatten()
    Y_south = Y_south.flatten()
    sp_south = sigma * np.ones_like(X_south)

    """
    y_west = np.linspace(y_min, y_max, ny)
    x_west = x_min * np.ones_like(y_west)
    sp_west = h * np.ones_like(y_west)
    # East Boundary
    y_east = np.linspace(y_min, y_max, ny)
    x_east = x_max * np.ones_like(y_east)
    sp_east = h * np.ones_like(y_east)
    # North Boundary
    x_north = np.linspace(x_min + h, x_max - h, nx - 2)
    y_north = y_max * np.ones_like(x_north)
    sp_north = h * np.ones_like(x_north)
    # South Boundary
    x_south = np.linspace(x_min + h, x_max - h, nx - 2)
    y_south = y_max * np.ones_like(x_south)
    sp_south = h * np.ones_like(x_south)
    """
    particle.x = np.concatenate((X_west, X_east, X_north, X_south))
    particle.y = np.concatenate((Y_west, Y_east, Y_north, Y_south))
    particle.diameter = np.concatenate((sp_west, sp_east, sp_north, sp_south))
    n_boundary = len(particle.x)

    # Inside Sphere
    R_in = 0
    R_out = R / 4
    h = h5 * sigma

    node_x, node_y, sp = generate_node_spherical(x_center, y_center, R, R_in, R_out, h)

    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.diameter = np.concatenate((particle.diameter, sp))

    R_in = R_out
    R_out = R / 2
    h = h4 * sigma
    node_x, node_y, sp = generate_node_spherical(x_center, y_center, R, R_in, R_out, h)

    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.diameter = np.concatenate((particle.diameter, sp))

    R_in = R_out
    R_out = 3 * R / 4
    h = h3 * sigma
    node_x, node_y, sp = generate_node_spherical(x_center, y_center, R, R_in, R_out, h)

    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.diameter = np.concatenate((particle.diameter, sp))

    R_in = R_out
    R_out = 7 * R / 8
    h = h2 * sigma
    node_x, node_y, sp = generate_node_spherical(x_center, y_center, R, R_in, R_out, h)

    R_in = R_out
    R_out = R
    h = h1 * sigma
    node_x, node_y, sp = generate_node_spherical(x_center, y_center, R, R_in, R_out, h)

    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.diameter = np.concatenate((particle.diameter, sp))
    
    n_sphere = len(particle.x)
    
    # Outside Sphere
    n_layer = 4
    h = h1 * sigma
    R_in = R_out
    R_out = R_in + n_layer * h
    node_x, node_y, sp = generate_node_spherical(x_center, y_center, R, R_in, R_out, h)

    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.diameter = np.concatenate((particle.diameter, sp))

    n_layer = 4
    h = h2 * sigma
    R_in = R_out
    R_out = R_in + n_layer * h
    node_x, node_y, sp = generate_node_spherical(x_center, y_center, R, R_in, R_out, h)

    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.diameter = np.concatenate((particle.diameter, sp))

    n_layer = 4
    h = h3 * sigma
    R_in = R_out
    R_out = R_in + n_layer * h
    node_x, node_y, sp = generate_node_spherical(x_center, y_center, R, R_in, R_out, h)

    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.diameter = np.concatenate((particle.diameter, sp))
    
    # Intermediate Particle
    h = h3 * sigma
    n_layer = 4;
    xmin = x_center - R_out - n_layer * h
    xmax = x_center + R_out + n_layer * h
    ymin = y_center - R_out - n_layer * h
    ymax = y_center + R_out + n_layer * h

    lx_ = xmax - xmin
    ly_ = ymax - ymin

    nx = int(lx_ / h) + 1
    ny = int(ly_ / h) + 1

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    X, Y = np.meshgrid(x, y)

    node_x = X.flatten()
    node_y = Y.flatten()

    delete_inner = (node_x - x_center)**2 + (node_y - y_center)**2 <= R_out**2
    delete_outer = (node_x <= x_min) + (node_x >= x_max) +  (node_y <= y_min) + (node_y >= y_max)
    delete = delete_inner + delete_outer
    node_x = node_x[~delete]
    #print(node_x)
    node_y = node_y[~delete]
    sp = h * np.ones_like(node_x)

    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.diameter = np.concatenate((particle.diameter, sp))
    
    # Box Particle
    x_min_inner = xmin
    x_max_inner = xmax
    y_min_inner = ymin
    y_max_inner = ymax
    
    n_layer = 16
    
    h = h4 * sigma
    xmin = x_min + h
    xmax = x_max - h
    ymin = y_min + h
    ymax = y_max - h
    
    lx_ = xmax - xmin
    ly_ = ymax - ymin

    nx = int(lx_ / h) + 1
    ny = int(ly_ / h) + 1

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    X, Y = np.meshgrid(x, y)

    node_x = X.flatten()
    node_y = Y.flatten()

    delete_inner = (node_x >= x_min_inner) * (node_x <= x_max_inner) * (node_y >= y_min_inner) * (node_y <= y_max_inner)
    node_x = node_x[~delete_inner]
    node_y = node_y[~delete_inner]
    sp = h * np.ones_like(node_x)

    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.diameter = np.concatenate((particle.diameter, sp))
    
    N = len(particle.x)
    particle.index = np.arange(0, N)
    particle.boundary = np.full(N, False)
    particle.boundary[:n_boundary] = True
    particle.solid = np.full(N, False)
    particle.solid[n_boundary:n_sphere] = True
    
    
    return particle, n_boundary
    
