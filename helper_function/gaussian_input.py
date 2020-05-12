# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 22:08:33 2020

@author: saki
"""

import numpy as np
from scipy import stats


def gaussian_input(num_neurons, features, width=2):
    """   
    Parameters
    ----------
    num_neurons(int): # of neurons
    features(list): Centers of gaussian input in degree([0, 360])
    width(float): standard deviation of gaussian input in degree
    
    Returns
    -------
    stim(array): Gaussian stimuli to neurons in ring
    """    
    neurons = np.arange(0, 360, 360/num_neurons)
    stim = np.zeros(num_neurons)
    for feature in features:
        tuning = stats.norm.pdf(neurons, loc=feature, scale=width)
        tuning = np.sqrt(2*np.pi) * width * tuning
        stim += tuning
    return stim


def direct_connect(num_neurons, features, width=10):
    """   
    Parameters
    ----------
    num_neurons(int): # of neurons
    features(list): 2d centers of gaussian input in degree, e.g., [[0, 360],[120, 240]]
    width(float): standard deviation of gaussian input in degree
    
    Returns
    -------
    stim(array): 2d Gaussian stimuli to neurons in torus
    """    
    x, y = np.mgrid[0:360:360/num_neurons, 0:360:360/num_neurons]
    neuron = np.dstack((x, y))
    stim = np.zeros([num_neurons, num_neurons])
    for feature in features:    
        tuning = stats.multivariate_normal.pdf(neuron,
                mean=feature, cov=[[width, 0], [0, width]])
        tuning = 2 * np.pi * width**2 * tuning
        stim += tuning     
    return stim / stim.max()