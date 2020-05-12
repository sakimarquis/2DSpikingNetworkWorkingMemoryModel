# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:46:37 2020

@author: saki
"""

from brian2 import *
from brian2.monitors import StateMonitor, SpikeMonitor, PopulationRateMonitor
import brian2 as b2
import brian2.numpy_ as np
from numpy.fft import rfft, irfft
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cython
import gc as gcPython

b2.prefs.codegen.target = 'cython'
b2.defaultclock.dt = 0.05 * b2.ms
#set_device('cpp_standalone', build_on_run=False)


def gaussian_input(num_neurons, features, width):
    neurons = np.arange(0, 360, 360/num_neurons)
    stim = np.zeros(num_neurons)
    for feature in features:
        tuning = stats.norm.pdf(neurons, loc=feature, scale=width)
        tuning = np.sqrt(2*np.pi) * width * tuning
        stim += tuning
    return stim


def structured_connectivity(N_excitatory, Jpos, sigma):
    tmp = np.sqrt(2*np.pi) * sigma
    Jneg = (360 - tmp * Jpos) / (360 - tmp)
    neuron = np.arange(N_excitatory)
    delta_theta = 360 * np.minimum(neuron, N_excitatory-neuron) / N_excitatory
    presyn_weight = (Jneg + (Jpos - Jneg) 
                     * np.exp(-1/2 * delta_theta**2 / sigma**2))
    return presyn_weight / sum(presyn_weight) * 360


def plot_network_activity(spike_monitor, voltage_monitor=None, spike_train_idx_list=None,
                          t_min=None, t_max=None, N_highlighted_spiketrains=3, avg_window_width=1.0 * b2.ms,
                          sup_title=None, figure_size=(10, 6), if3d=None, num_neuron=None, num_trial='0'):
    assert isinstance(spike_monitor, b2.SpikeMonitor), "spike_monitor is not of type SpikeMonitor"

    def plot_raster():
        ax.plot(spike_monitor.t/ms, spike_monitor.i,  c="k", marker=".", markersize=2, linewidth=0)
        ax.set_xlabel('Time (ms)')
        ax.set_xticks(np.arange(0, 2001, 100))
        ax.set_ylabel('Neuron index')
        ax.set_ylim(0, num_neuron)
        ax.set_title("Raster Plot", fontsize=10)
        plt.savefig(num_trial+'raster_1d.png', quality = 95)

    def plot_raster3d():
        ax = Axes3D(fig)
        y = np.array(spike_monitor.i) % num_neuron
        z = np.array(spike_monitor.i) // num_neuron
        x = spike_monitor.t/ms
        ax.plot(x, y, z, c="k", marker=".", markersize=2, linewidth=0, alpha=0.7)
        ax.set_xlabel('Time')
        ax.set_xlim(0, 500)
        ax.set_ylabel('1st dimension')
        ax.set_ylim(0, num_neuron)
        ax.set_zlabel('2nd dimension')
        ax.set_zlim(0, num_neuron)
        ax.view_init(30, -70)
        plt.savefig(num_trial+'raster_2d.png', quality = 95)
       
    fig, ax = plt.subplots(figsize=figure_size)
    if if3d:
        plot_raster3d()
    else:
        plot_raster()     
    # if sup_title is not None:
    #     plt.suptitle(sup_title)
    plt.close()
    return None # fig, ax


def simulate_wm(
    N_excitatory=4096, 
    N_inhibitory=1024,
    N_extern_poisson=1000, 
    poisson_firing_rate=1 * b2.Hz, 
    features=[120, 240], 
    stimulus_width=2, 
    strength=0.4 * b2.namp,
    t_stimulus_start=200 * b2.ms, 
    t_stimulus_duration=100 * b2.ms,
    sim_time=1500. * b2.ms,
    Jpos_excit2excit=4.02,
    sigma_weight=5.,
    G_inhib2excit=0.6681 * b2.nS, # 1.336
    G_inhib2inhib=0.512 * b2.nS, # 1.024
    G_excit2excit=0.1905 * b2.nS, # 0.381
    G_excit2inhib=0.146 * b2.nS, # 1.2 * 0.292
    G_extern2excit=2.48 * b2.nS, # 3.1
    G_extern2inhib=1.9 * b2.nS, # 2.38 
    scaling_inhib=1,
    scaling_excit=1):
    """
    Args:
        N_excitatory (int): Size of the excitatory population
        N_inhibitory (int): Size of the inhibitory population
        weight_scaling_factor (float): weight prefactor. When increasing the size of the populations,
            the synaptic weights have to be decreased. Using the default values, we have
            N_excitatory*weight_scaling_factor = 2048 and N_inhibitory*weight_scaling_factor=512
        N_extern_poisson (int): Size of the external input population (Poisson input)
        poisson_firing_rate (Quantity): Firing rate of the external population
        sigma_weight_profile (float): standard deviation of the gaussian input profile in
            the excitatory population.
        Jpos_excit2excit (float): Strength of the recurrent input within the excitatory population.
            Jneg_excit2excit is computed from sigma_weight_profile, Jpos_excit2excit and the normalization
            condition.
        features (list): Center of gaussian input in degree([0, 360])
        stimulus_width (float): Sigma of gaussian input
        strength (Quantity): Input current to the neurons at stimulus_center_deg +\- (stimulus_width_deg/2)
        t_stimulus_start (Quantity): time when the input stimulus is turned on
        t_stimulus_duration (Quantity): duration of the stimulus.
        distractor_center_deg (float): Center of the distractor in [0, 360]
        distractor_width_deg (float): width of the distractor. All neurons in
            distractor_center_deg +\- (distractor_width_deg/2) receive the same input current
            distractor_strength (Quantity): Input current to the neurons at
            distractor_center_deg +\- (distractor_width_deg/2)
        t_distractor_start (Quantity): time when the distractor is turned on
        t_distractor_duration (Quantity): duration of the distractor.
        G_inhib2inhib (Quantity): projections from inhibitory to inhibitory population (later
            rescaled by weight_scaling_factor)
        G_inhib2excit (Quantity): projections from inhibitory to excitatory population (later
            rescaled by weight_scaling_factor)
        G_excit2excit (Quantity): projections from excitatory to excitatory population (later
            rescaled by weight_scaling_factor)
        G_excit2inhib (Quantity): projections from excitatory to inhibitory population (later
            rescaled by weight_scaling_factor)
        monitored_subset_size (int): nr of neurons for which a Spike- and Voltage monitor
            is registered.
        sim_time (Quantity): simulation time

    Returns:

       results (tuple):
       rate_monitor_excit (Brian2 PopulationRateMonitor for the excitatory population),
        spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45 (The weights profile for the neuron with preferred direction = 45deg).
    """
    # =============================================================================
    # 参数设定
    # =============================================================================
    # specify the excitatory pyramidal cells:
    Cm_excit = 0.5 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak_excit = 25.0 * b2.nS  # leak conductance
    E_leak_excit = -70.0 * b2.mV  # reversal potential
    v_firing_threshold_excit = -50.0 * b2.mV  # spike condition
    v_reset_excit = -60.0 * b2.mV  # reset voltage after spike
    t_abs_refract_excit = 2.0 * b2.ms  # absolute refractory period
    
    # specify the inhibitory interneurons:
    Cm_inhib = 0.2 * b2.nF
    G_leak_inhib = 20.0 * b2.nS
    E_leak_inhib = -70.0 * b2.mV
    v_firing_threshold_inhib = -50.0 * b2.mV
    v_reset_inhib = -60.0 * b2.mV
    t_abs_refract_inhib = 1.0 * b2.ms

    # specify the AMPA synapses
    E_AMPA = 0.0 * b2.mV
    tau_AMPA = 2.0 * b2.ms

    # specify the GABA synapses
    E_GABA = -70.0 * b2.mV
    tau_GABA = 10.0 * b2.ms

    # specify the NMDA synapses
    E_NMDA = 0.0 * b2.mV
    tau_NMDA_s = 100.0 * b2.ms 
    tau_NMDA_x = 2.0 * b2.ms
    alpha_NMDA = 0.5 * b2.kHz

    # projectsions from the inhibitory populations
    G_inhib2excit *= scaling_inhib
    G_inhib2inhib *= scaling_inhib

    # projections from the excitatory population
    G_excit2excit *= scaling_excit
    G_excit2inhib *= scaling_excit
    
    # =============================================================================
    # 感觉1d网络
    # =============================================================================

    # define the inhibitory population
    inhib_lif_dynamics = """
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_inhib * (v-E_leak_inhib)
        - G_extern2inhib * s_AMPA * (v-E_AMPA)
        - G_inhib2inhib * s_GABA * (v-E_GABA)
        - G_excit2inhib * s_NMDA_total * (v-E_NMDA) / (1.0+1.0*exp(-0.062*v/volt)/3.57)
        )/Cm_inhib : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA / tau_AMPA : 1
        ds_GABA/dt = -s_GABA / tau_GABA : 1
    """

    # specify the excitatory population:
    excit_lif_dynamics = """
        I_stim : amp
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_excit * (v-E_leak_excit)
        - G_extern2excit * s_AMPA * (v-E_AMPA)
        - G_inhib2excit * s_GABA * (v-E_GABA)
        - G_excit2excit * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*v/volt)/3.57)
        + I_stim
        )/Cm_excit : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA / tau_AMPA : 1
        ds_GABA/dt = -s_GABA / tau_GABA : 1
        ds_NMDA/dt = -s_NMDA / tau_NMDA_s + alpha_NMDA * x * (1-s_NMDA) : 1
        dx/dt = -x / tau_NMDA_x : 1
    """
    
    # 抑制神经元群
    inhib_pop = NeuronGroup(N_inhibitory, model=inhib_lif_dynamics, threshold="v>v_firing_threshold_inhib", 
        reset="v=v_reset_inhib", refractory=t_abs_refract_inhib, method="rk2")
    
    # initialize with random voltages:
    inhib_pop.v = np.random.uniform(v_reset_inhib / b2.mV, high=v_firing_threshold_inhib / b2.mV,
                                    size=N_inhibitory) * b2.mV
    # set the connections: inhib2inhib 
    syn_inhib2inhib = Synapses(inhib_pop, target=inhib_pop, on_pre="s_GABA += 1.0", delay=0.0 * b2.ms)
    syn_inhib2inhib.connect(condition="i!=j", p=1.0)
    
    # set the connections: extern2inhib
    input_ext2inhib = PoissonInput(target=inhib_pop, target_var="s_AMPA",
                                N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    #兴奋神经元群
    excit_pop = NeuronGroup(N_excitatory, model=excit_lif_dynamics,
                            threshold="v>v_firing_threshold_excit", reset="v=v_reset_excit; x+=1.0",
                            refractory=t_abs_refract_excit, method="rk2")
    
    # initialize with random voltages:
    excit_pop.v = np.randomz.uniform(v_reset_excit / b2.mV, high=v_firing_threshold_excit / b2.mV,
                                    size=N_excitatory) * b2.mV
    excit_pop.I_stim = 0. * b2.namp
    
    # set the connections: extern2excit
    input_ext2excit = PoissonInput(target=excit_pop, target_var="s_AMPA",
                                N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # set the connections: inhibitory to excitatory
    syn_inhib2excit = Synapses(inhib_pop, target=excit_pop, on_pre="s_GABA += 1.0")
    syn_inhib2excit.connect(p=1.0)

    # set the connections: excitatory to inhibitory NMDA connections
    syn_excit2inhib = Synapses(excit_pop, inhib_pop,
                            model="s_NMDA_total_post = s_NMDA_pre : 1 (summed)", method="rk2")
    syn_excit2inhib.connect(p=1.0)


    # =============================================================================
    # 开始模拟！
    # =============================================================================
    # precompute the weight profile for the recurrent population
    presyn_weight = structured_connectivity(N_excitatory, Jpos_excit2excit, sigma_weight)
    fft_presyn_weight = rfft(presyn_weight)
      
    # set the STRUCTURED recurrent input. use a network_operation
    @network_operation()
    def update_nmda_sum():
        fft_s_NMDA = rfft(excit_pop.s_NMDA)
        fft_s_NMDA_total = np.multiply(fft_presyn_weight, fft_s_NMDA)
        excit_pop.s_NMDA_total_ = irfft(fft_s_NMDA_total)  
         
    
    stimulus = gaussian_input(N_excitatory, features, stimulus_width)
    t_stimulus_end = t_stimulus_start + t_stimulus_duration
    
    @network_operation(dt=1 * b2.ms)
    def stimulate_network(t):
        if t >= t_stimulus_start and t < t_stimulus_end:
            excit_pop.I_stim = stimulus * strength
        else:
            excit_pop.I_stim = 0. * b2.namp

    # collect data of a subset of neurons:
    spike_monitor_sensory = SpikeMonitor(excit_pop)
    # spike_monitor_high = SpikeMonitor(high_excit_pop)
    b2.run(sim_time)
    return spike_monitor_sensory#, spike_monitor_high
