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
spec['index'] = nb.int64[:]
spec['diameter'] = nb.float64[:]
spec['boundary'] = nb.boolean[:]
spec['solid'] = nb.boolean[:]
spec['n_east'] = nb.int64
spec['n_west'] = nb.int64
spec['n_north'] = nb.int64
spec['n_south'] = nb.int64
spec['n_top'] = nb.int64
spec['n_bottom'] = nb.int64
spec['n_bound'] = nb.int64

@nb.experimental.jitclass(spec)
class Particle:
    def __init__(self):
        self.x = np.zeros(0, dtype=np.float64)
        self.y = np.zeros(0, dtype=np.float64)
        self.z = np.zeros(0, dtype=np.float64)
        self.index = np.zeros(0, dtype=np.int64)
        self.diameter = np.zeros(0, dtype=np.float64)
        self.boundary = np.full(0, True)
        self.solid = np.full(0, True)
        self.n_east = 0
        self.n_west = 0
        self.n_north = 0
        self.n_south = 0
        self.n_top = 0
        self.n_bottom = 0
        self.n_bound = 0
        
        