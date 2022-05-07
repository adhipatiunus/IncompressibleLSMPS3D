#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 11:39:38 2022

@author: adhipatiunus
"""
import numpy as np

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
        