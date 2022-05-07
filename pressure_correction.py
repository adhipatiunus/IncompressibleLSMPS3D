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