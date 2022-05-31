#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 11:39:38 2022

@author: adhipatiunus
"""
import numpy as np
import numba as nb

@nb.njit
def empty_int32_list():
    l = [nb.int32(10)]
    l.clear()
    return l

spec = {}
spec['x'] = nb.float64[:]
spec['y'] = nb.float64[:]
spec['z'] = nb.float64[:]
spec['index'] = nb.int32
spec['boundary'] = nb.boolean
spec['solid'] = nb.boolean
spec['neighbor_all'] = empty_int32_list()
spec['neighbor_xpos'] = empty_int32_list()
spec['neighbor_xneg'] = empty_int32_list()
spec['neighbor_ypos'] = empty_int32_list()
spec['neighbor_yneg'] = empty_int32_list()
spec['neighbor_zpos'] = empty_int32_list()
spec['neighbor_zneg'] = empty_int32_list()
spec['n_east'] = nb.int32
spec['n_west'] = nb.int32
spec['n_north'] = nb.int32
spec['n_south'] = nb.int32
spec['n_top'] = nb.int32
spec['n_bottom'] = nb.int32

@nb.jitclass(spec)
class Particle:
    def __init__(self):
        self.x = np.array([])
        self.y = np.array([])
        self.z = np.array([])
        self.index = np.array([])
        self.diameter = np.array([])
        self.boundary = np.array([])
        self.solid = np.array([])
        self.neighbor_all = []
        self.neighbor_xpos = []
        self.neighbor_xneg = []
        self.neighbor_ypos = []
        self.neighbor_yneg = []
        self.neighbor_zpos = []
        self.neighbor_zneg = []
        self.n_east = 0
        self.n_west = 0
        self.n_north = 0
        self.n_south = 0
        self.n_top = 0
        self.n_bottom = 0
        