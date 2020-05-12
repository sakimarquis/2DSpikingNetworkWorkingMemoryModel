# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def plot_raster_1d(neuron_idx, spike_time, name):
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.plot(spike_time, neuron_idx,  c="k", marker=".", markersize=2, linewidth=0)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron index')
    ax.set_title("Raster Plot", fontsize=10)
    plt.savefig(name+'raster_1d.png', quality = 95)
    plt.close()
    return None #fig, ax

def plot_raster_3d(num_neuron_1d, neuron_idx, spike_time, name):
    fig, ax = plt.subplots(figsize=(20, 12))
    ax = Axes3D(fig)
    y = np.array(neuron_idx) % num_neuron_1d
    z = np.array(neuron_idx) // num_neuron_1d
    x = spike_time
    ax.plot(x, y, z, c="k", marker=".", markersize=2, linewidth=0, alpha=0.7)
    ax.set_xlabel('Time')
    #ax.set_xlim(0, 500)
    ax.set_ylabel('1st dimension')
    ax.set_ylim(0, num_neuron_1d)
    ax.set_zlabel('2nd dimension')
    ax.set_zlim(0, num_neuron_1d)
    ax.view_init(30, -70)
    plt.savefig(name+'raster_3d.png', quality = 95)
    plt.close()
    return None #fig, ax

files = os.listdir()
for file in files:
    if "pyr_cell" in file:
        data = pd.read_csv(file, sep=",", header=None)
        name = file.split('_')[-1].split('.')[0]
        data.columns = ["Neuron idx", "Spike time"]
        low = data[data["Neuron idx"] <= 4096 * 2]
        plot_raster_1d(low["Neuron idx"], low["Spike time"], name)
        high = data[data["Neuron idx"] > 4096 * 2]
        plot_raster_3d(2048, high["Neuron idx"], high["Spike time"], name)

