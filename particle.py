#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 11:39:38 2022

@author: adhipatiunus
"""
import numpy as np
import numba as nb

spec = {}
spec['x'] = nb.float64[:]
spec['y'] = nb.float64[:]
spec['z'] = nb.float64[:]
spec['index'] = nb.int32[:]
spec['diameter'] = nb.float64[:]
spec['boundary'] = nb.boolean[:]
spec['solid'] = nb.boolean[:]
spec['neighbor_all'] = nb.types.List(nb.int32)
spec['neighbor_xpos'] = nb.types.List(nb.int32)
spec['neighbor_xneg'] = nb.types.List(nb.int32)
spec['neighbor_ypos'] = nb.types.List(nb.int32)
spec['neighbor_yneg'] = nb.types.List(nb.int32)
spec['neighbor_zpos'] = nb.types.List(nb.int32)
spec['neighbor_zneg'] = nb.types.List(nb.int32)
spec['n_east'] = nb.int32
spec['n_west'] = nb.int32
spec['n_north'] = nb.int32
spec['n_south'] = nb.int32
spec['n_top'] = nb.int32
spec['n_bottom'] = nb.int32

@nb.njit
def empty_int32_list():
    l = [nb.int32(10)]
    l.clear()
    return l

@nb.njit
def empty_bool_list():
    l = [nb.boolean(True)]
    l.clear()
    return l

@nb.experimental.jitclass(spec)
class Particle:
    def __init__(self):
        self.x = np.zeros(0, dtype=np.float64)
        self.y = np.zeros(0, dtype=np.float64)
        self.z = np.zeros(0, dtype=np.float64)
        self.index = np.zeros(0, dtype=np.int32)
        self.diameter = np.zeros(0, dtype=np.float64)
        self.boundary = np.full(0, True)
        self.solid = np.full(0, True)
        self.neighbor_all = empty_int32_list()
        self.neighbor_xpos = empty_int32_list()
        self.neighbor_xneg = empty_int32_list()
        self.neighbor_ypos = empty_int32_list()
        self.neighbor_yneg = empty_int32_list()
        self.neighbor_zpos = empty_int32_list()
        self.neighbor_zneg = empty_int32_list()
        self.n_east = 0
        self.n_west = 0
        self.n_north = 0
        self.n_south = 0
        self.n_top = 0
        self.n_bottom = 0
        
        