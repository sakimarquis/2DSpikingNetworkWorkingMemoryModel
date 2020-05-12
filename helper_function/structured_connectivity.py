# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:53:47 2020

@author: saki
"""

import numpy as np


def structured_connectivity(N_excitatory, Jpos, sigma):
    """   
    J– = (360 - sqrt(2*pi) * sigma * J+)/ (360 - sqrt(2*pi) * sigma)
    W(θi – θj) = J– + (J+ – J–)exp[–(θi – θj)^2/2σ^2]
    
    Parameters
    ----------
    N_excitatory(int): # of the excitatory population
    Jpos(float): Strength of the structured excitatory-to-excitatory 
        connectivity in recurrent population
    sigma(float): standard deviation of the structured excitatory-to-excitatory 
        connectivity in recurrent population

    Returns
    -------
    presyn_weight(array): weight kernel of the structured 
        excitatory-to-excitatory connectivity in recurrent population
    """
    tmp = np.sqrt(2*np.pi) * sigma
    Jneg = (360 - tmp * Jpos) / (360 - tmp)
    neuron = np.arange(N_excitatory)
    delta_theta = 360 * np.minimum(neuron, N_excitatory-neuron) / N_excitatory
    presyn_weight = (Jneg + (Jpos - Jneg) 
                     * np.exp(-1/2 * delta_theta**2 / sigma**2))
    return presyn_weight / sum(presyn_weight) * 360


def structured_connectivity_2d(N_excitatory, Jpos, sigma):
    tmp = np.sqrt(2*np.pi) * sigma
    Jneg = (360 - tmp * Jpos) / (360 - tmp)
    x, y = np.mgrid[0:N_excitatory, 0:N_excitatory]
    neurons = np.dstack((x, y)).reshape(-1, 2)
    neurons = np.linalg.norm(neurons, axis=1).reshape(N_excitatory, -1)
    dist = np.linalg.norm([N_excitatory, N_excitatory])
    delta_theta = 360 * np.minimum(neurons, dist - neurons) / dist
    presyn_weight = (Jneg + (Jpos - Jneg) 
                     * np.exp(-1/2 * delta_theta**2 / sigma**2))
    return presyn_weight