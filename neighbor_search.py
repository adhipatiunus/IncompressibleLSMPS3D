#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 13:15:05 2022

@author: adhipatiunus
"""

import numpy as np

def neighbor_search_naive(particle, cell_size):
    N = len(particle.index)
    particle.neighbor_all   = [[] for i in range(N)]
    particle.neighbor_xpos  = [[] for i in range(N)]
    particle.neighbor_xneg  = [[] for i in range(N)]
    particle.neighbor_ypos  = [[] for i in range(N)]
    particle.neighbor_yneg  = [[] for i in range(N)]
    for i in range(N):
        print(str(i/N*100)+'%')
        for j in range(N):
            if distance(particle.x[i], particle.x[j], particle.y[i], particle.y[j]) < cell_size:
                x_ij = particle.x[i] - particle.x[j]
                y_ij = particle.y[i] - particle.y[j]
                if x_ij < 10**-6:
                    particle.neighbor_xpos[i].append(j)
                elif x_ij > -10**-6:
                    particle.neighbor_xneg[i].append(j)
                if y_ij < 10**-6:
                    particle.neighbor_ypos[i].append(j)
                elif y_ij > -10**-6:
                    particle.neighbor_yneg[i].append(j)
                particle.neighbor_all[i].append(j)
                
def neighbor_search_cell_list(particle, cell_size, y_max, y_min, x_max, x_min, z_max, z_min):
    nx = int((x_max - x_min) / cell_size) + 1
    ny = int((y_max - y_min) / cell_size) + 1
    nz = int((z_max - z_min) / cell_size) + 1

    cell = [[[[] for i in range(nx)] for j in range(ny)] for k in range(nz)]

    N = len(particle.index)

    for i in range(N):
        listx = int((particle.x[i] - x_min) / cell_size)
        listy = int((particle.y[i] - y_min) / cell_size)
        listz = int((particle.z[i] - z_min) / cell_size)
        #print(particle.x[i])
        cell[listx][listy][listz].append(particle.index[i])
        
    particle.neighbor_all   = [[] for i in range(N)]
    particle.neighbor_xpos  = [[] for i in range(N)]
    particle.neighbor_xneg  = [[] for i in range(N)]
    particle.neighbor_ypos  = [[] for i in range(N)]
    particle.neighbor_yneg  = [[] for i in range(N)]
    particle.neighbor_zpos  = [[] for i in range(N)]
    particle.neighbor_zneg  = [[] for i in range(N)]
    
    pcnt = 0
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for pi in cell[i][j][k]:
                    neigh_x, neigh_y, neigh_z = i - 1, j - 1, k - 1
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i - 1, j - 1, k
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i - 1, j - 1, k + 1
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i - 1, j, k - 1
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i - 1, j, k
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i - 1, j, k + 1
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i - 1, j + 1, k - 1
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i - 1, j + 1, k
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i - 1, j + 1, k + 1
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i, j - 1, k - 1
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i, j - 1, k
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i, j - 1, k + 1
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i, j, k - 1
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i, j, k
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i, j, k + 1
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i, j + 1, k - 1
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i, j + 1, k
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i, j + 1, k + 1
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i + 1, j - 1, k - 1
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i + 1, j - 1, k
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i + 1, j - 1, k + 1
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i + 1, j, k - 1
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i + 1, j, k
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i + 1, j, k + 1
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i + 1, j + 1, k - 1
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i + 1, j + 1, k
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    
                    neigh_x, neigh_y, neigh_z = i + 1, j + 1, k + 1
                    push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz)
                    print(str(round(pcnt/N*100,2))+'%')
                    pcnt += 1
        print(str(round(pcnt/N*100,2))+'%')
    
def push_back_particle(particle, cell_size, cell, pi, neigh_x, neigh_y, neigh_z, nx, ny, nz):
    if neigh_x >= 0 and neigh_y >= 0 and neigh_z >= 0 and neigh_x < nx and neigh_y < ny and neigh_z < nz:
        for pj in cell[neigh_x][neigh_y][neigh_z]:
            if distance(particle.x[pi], particle.x[pj], particle.y[pi], particle.y[pj], particle.z[pi], particle.z[pj]) < cell_size and pi != pj:
                x_ij = particle.x[pi] - particle.x[pj]
                y_ij = particle.y[pi] - particle.y[pj]
                z_ij = particle.z[pi] - particle.z[pj]
                if x_ij < 10**-6:
                    particle.neighbor_xpos[pi].append(pj)
                elif x_ij > -10**-6:
                    particle.neighbor_xneg[pi].append(pj)
                if y_ij < 10**-6:
                    particle.neighbor_ypos[pi].append(pj)
                elif y_ij > -10**-6:
                    particle.neighbor_yneg[pi].append(pj)
                if z_ij < 10**-6:
                    particle.neighbor_zpos[pi].append(pj)
                elif z_ij > -10**-6:
                    particle.neighbor_zneg[pi].append(pj)
                particle.neighbor_all[pi].append(pj)
                            
                    
def distance(x1, x2, y1, y2, z1, z2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

