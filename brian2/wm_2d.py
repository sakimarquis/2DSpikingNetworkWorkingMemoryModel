# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:46:37 2020

@author: saki
"""

from numpy.fft import rfft, irfft, rfft2, irfft2
from scipy import stats
import brian2 as b2
import brian2.numpy_ as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from brian2 import *
from itertools import product
import cython
import gc as gcPython

prefs.codegen.target = 'cython'
#set_device('cpp_standalone', build_on_run=False)
# b2.defaultclock.dt = 0.05 * b2.ms


def gaussian_input(num_neurons, center_deg, width_deg):
    neuron = np.arange(0, 360, 360/num_neurons)
    tuning = stats.norm.pdf(neuron, loc=center_deg, scale=width_deg)
    return tuning * width_deg


def direct_connect(num_neurons, centers_deg, width_deg):
    width = width_deg * 5
    x, y = np.mgrid[0:360:360/num_neurons, 0:360:360/num_neurons]
    neuron = np.dstack((x, y))
    tuning = stats.multivariate_normal.pdf(neuron,
            mean=centers_deg, cov=[[width, 0], [0, width]])
    return tuning * 50 #.reshape(-1)


def structured_connectivity(N_excitatory, Jpos, sigma):
    """   
    J– = (360 - math.sqrt(2*pi) * sigma * J+)/ (360 - math.sqrt(2*pi) * sigma)
    W(θi – θj) = J– + (J+ – J–)exp[–(θi – θj)^2/2σ^2]
    
    Parameters
    ----------
    N_excitatory(int): Size of the excitatory population
    Jpos(float): Strength of the recurrent input within the excitatory population.
    sigma(float): standard deviation of the gaussian input profile in the 
        excitatory population.

    Returns
    -------
    presyn_weight : array
        weight profile for the structured excitatory-to-excitatory connectivity
        in recurrent population

    """
    tmp = np.sqrt(2*np.pi) * sigma
    Jneg = (360 - tmp * Jpos) / (360 - tmp)
    neurons = np.arange(N_excitatory)
    delta_theta = 360 * np.minimum(neurons, N_excitatory - neurons) / N_excitatory
    presyn_weight = (Jneg + (Jpos - Jneg) 
                     * np.exp(-1/2 * delta_theta**2 / sigma**2))
    return presyn_weight


def structured_connectivity_2d(N_excitatory, Jpos, sigma):
    tmp = np.sqrt(2*np.pi) * sigma
    Jneg = (360 - tmp * Jpos) / (360 - tmp)
    neurons = list(product(range(N_excitatory), repeat=2))
    neurons = np.linalg.norm(neurons, axis=1).reshape(N_excitatory, -1)
    dist = np.linalg.norm([N_excitatory, N_excitatory])
    delta_theta = 360 * np.minimum(neurons, dist - neurons) / dist
    presyn_weight = (Jneg + (Jpos - Jneg) 
                     * np.exp(-1/2 * delta_theta**2 / sigma**2))
    return presyn_weight


def plot_network_activity(spike_monitor, voltage_monitor=None, spike_train_idx_list=None,
                          t_min=None, t_max=None, N_highlighted_spiketrains=3, avg_window_width=1.0 * b2.ms,
                          sup_title=None, figure_size=(10, 6), if3d=None, num_neuron=None, num_trial='0'):
    """
    Visualizes the results of a network simulation: spike-train, population activity and voltage-traces.

    Args:
        rate_monitor (PopulationRateMonitor): rate of the population
        spike_monitor (SpikeMonitor): spike trains of individual neurons
        voltage_monitor (StateMonitor): optional. voltage traces of some (same as in spike_train_idx_list) neurons
        spike_train_idx_list (list): optional. A list of neuron indices whose spike-train is plotted.
            If no list is provided, all (up to 500) spike-trains in the spike_monitor are plotted. If None, the
            the list in voltage_monitor.record is used.
        t_min (Quantity): optional. lower bound of the plotted time interval.
            if t_min is None, it is set to the larger of [0ms, (t_max - 100ms)]
        t_max (Quantity): optional. upper bound of the plotted time interval.
            if t_max is None, it is set to the timestamp of the last spike in
        N_highlighted_spiketrains (int): optional. Number of spike trains visually highlighted, defaults to 3
            If N_highlighted_spiketrains==0 and voltage_monitor is not None, then all voltage traces of
            the voltage_monitor are plotted. Otherwise N_highlighted_spiketrains voltage traces are plotted.
        avg_window_width (Quantity): optional. Before plotting the population rate (PopulationRateMonitor), the rate
            is smoothed using a window of width = avg_window_width. Defaults is 1.0ms
        sup_title (String): figure suptitle. Default is None.
        figure_size (tuple): (width,height) tuple passed to pyplot's figsize parameter.

    Returns:
        Figure: The whole figure
        Axes: Top panel, Raster plot
        Axes: Middle panel, population activity
        Axes: Bottom panel, voltage traces. None if no voltage monitor is provided.
    """
    assert isinstance(spike_monitor, b2.SpikeMonitor), \
        "spike_monitor is not of type SpikeMonitor"

    def plot_raster():
        ax.plot(spike_monitor.t/ms, spike_monitor.i,  c="k", marker=".", markersize=2, linewidth=0)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron index')
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
    N_excitatory=720, 
    N_inhibitory=180,
    N_extern_poisson=1000, 
    poisson_firing_rate=1.6 * b2.Hz, 
    stimulus_center_deg=180, 
    stimulus_width_deg=4, 
    stimulus_strength= 1.0 * b2.namp,
    t_stimulus_start=0 * b2.ms, 
    t_stimulus_duration=0 * b2.ms,
    sim_time=500. * b2.ms,
    Jpos_excit2excit=3,
    sigma_weight_profile=10., 
    Jpos_high=6,
    scaling_sensory_inhib=1,
    scaling_sensory_excit=1,
    scaling_high_inhib=1,
    scaling_high_excit=1, 
    direct_input=True):
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
        stimulus_center_deg (float): Center of gaussian input in [0, 360]
        stimulus_width_deg (float): Sigma of gaussian input
        stimulus_strength (Quantity): Input current to the neurons at stimulus_center_deg +\- (stimulus_width_deg/2)
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
    compress_factor = 4
    
    # this parameters are adopted by paper 2012 JN
    # 到所有神经元中，抑制的大小
    G_inhib2excit = 1.336 * b2.nS
    G_inhib2inhib = 1.024 * b2.nS
    # 到所有神经元中，兴奋的大小
    G_excit2excit = 0.381 * b2.nS
    G_excit2inhib = 1.1 * 0.292 * b2.nS
    # projections from the external population
    G_extern2excit = 3.1 * b2.nS * 0.8
    G_extern2inhib = 2.38 * b2.nS * 0.8
    
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
    G_inhib2excit *= scaling_sensory_inhib
    G_inhib2inhib *= scaling_sensory_inhib

    # projections from the excitatory population
    G_excit2excit *= scaling_sensory_excit
    G_excit2inhib *= scaling_sensory_excit 
    
# =============================================================================
# 感觉网络，两个rings
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
    inhib_pop = NeuronGroup(N_inhibitory*2, model=inhib_lif_dynamics, threshold="v>v_firing_threshold_inhib", 
        reset="v=v_reset_inhib", refractory=t_abs_refract_inhib, method="rk2")
    
    # initialize with random voltages:
    inhib_pop.v = np.random.uniform(v_reset_inhib / b2.mV, high=v_firing_threshold_inhib / b2.mV,
                                    size=N_inhibitory*2) * b2.mV
    
    # set the connections: inhib2inhib 
    syn_inhib2inhib = Synapses(inhib_pop, target=inhib_pop, on_pre="s_GABA += 1.0", delay=0.0 * b2.ms)
    syn_inhib2inhib.connect(j='k for k in range(N_inhibitory) if i<N_inhibitory and i != j')
    syn_inhib2inhib.connect(j='k for k in range(N_inhibitory, 2*N_inhibitory) if i>=N_inhibitory and i != j')
    
    # set the connections: extern2inhib
    input_ext2inhib = PoissonInput(target=inhib_pop, target_var="s_AMPA",
                                N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    #兴奋神经元群
    excit_pop = NeuronGroup(N_excitatory*2, model=excit_lif_dynamics,
                            threshold="v>v_firing_threshold_excit", reset="v=v_reset_excit; x+=1.0",
                            refractory=t_abs_refract_excit, method="rk2")
    
    # initialize with random voltages:
    excit_pop.v = np.random.uniform(v_reset_excit / b2.mV, high=v_firing_threshold_excit / b2.mV,
                                    size=N_excitatory*2) * b2.mV
    excit_pop.I_stim = 0. * b2.namp
    
    # set the connections: extern2excit
    input_ext2excit = PoissonInput(target=excit_pop, target_var="s_AMPA",
                                N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # set the connections: inhibitory to excitatory
    syn_inhib2excit = Synapses(inhib_pop, target=excit_pop, on_pre="s_GABA += 1.0")
    syn_inhib2excit.connect(j='k for k in range(N_excitatory) if i<N_inhibitory')
    syn_inhib2excit.connect(j='k for k in range(N_excitatory, 2*N_excitatory) if i>=N_inhibitory')

    # set the connections: excitatory to inhibitory NMDA connections
    syn_excit2inhib = Synapses(excit_pop, inhib_pop,
                            model="s_NMDA_total_post = s_NMDA_pre : 1 (summed)", method="rk2")
    syn_excit2inhib.connect(j='k for k in range(N_inhibitory) if i<N_excitatory')
    syn_excit2inhib.connect(j='k for k in range(N_inhibitory, 2*N_inhibitory) if i>=N_excitatory')
    
# =============================================================================
# 高级网络
# =============================================================================
    HG_inhib2excit = 1.336 * b2.nS
    HG_inhib2inhib = 1.024 * b2.nS
    # 到所有神经元中，兴奋的大小
    HG_excit2excit = 0.381 * b2.nS
    HG_excit2inhib = 0.292 * b2.nS
    # projections from the external population
    HG_extern2excit = 3.1 * b2.nS * 0.8
    HG_extern2inhib = 2.38 * b2.nS * 0.8

    renormalize_factor = 50
    N_excit_high_1d = int(N_excitatory / compress_factor)
    N_excit_high = N_excit_high_1d**2
    N_inbit_high = int(N_excit_high / 4 / renormalize_factor)


    # 到所有神经元中，抑制的大小
    HG_inhib2excit = HG_inhib2excit * scaling_high_inhib * renormalize_factor
    HG_inhib2inhib = HG_inhib2inhib * scaling_high_inhib * renormalize_factor
    # 到所有神经元中，兴奋的大小
    HG_excit2excit = HG_excit2excit * scaling_high_excit
    HG_excit2inhib = HG_excit2inhib * scaling_high_excit
    
    # define the inhibitory population
    high_inhib_lif = """
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_inhib * (v-E_leak_inhib)
        - HG_extern2inhib * s_AMPA * (v-E_AMPA)
        - HG_inhib2inhib * s_GABA * (v-E_GABA)
        - HG_excit2inhib * s_NMDA_total * (v-E_NMDA) / (1.0+1.0*exp(-0.062*v/volt)/3.57)
        )/Cm_inhib : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA / tau_AMPA : 1
        ds_GABA/dt = -s_GABA / tau_GABA : 1
    """

    # specify the excitatory population:
    high_excit_lif = """
        I_stim : amp
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_excit * (v-E_leak_excit)
        - HG_extern2excit * s_AMPA * (v-E_AMPA)
        - HG_inhib2excit * s_GABA * (v-E_GABA)
        - HG_excit2excit * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*v/volt)/3.57)
        + I_stim
        )/Cm_excit : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA / tau_AMPA : 1
        ds_GABA/dt = -s_GABA / tau_GABA : 1
        ds_NMDA/dt = -s_NMDA / tau_NMDA_s + alpha_NMDA * x * (1-s_NMDA) : 1
        dx/dt = -x / tau_NMDA_x : 1
    """
    
    # 高级区域抑制神经元
    high_inhib_pop = NeuronGroup(N_inbit_high, model=high_inhib_lif, threshold="v>v_firing_threshold_inhib", 
        reset="v=v_reset_inhib", refractory=t_abs_refract_inhib, method="rk2")
    # initialize with random voltages:
    high_inhib_pop.v = np.random.uniform(v_reset_inhib / b2.mV, high=v_firing_threshold_inhib / b2.mV,
                                    size=N_inbit_high) * b2.mV
    
    # set the connections: inhib2inhib
    high_syn_inhib2inhib = Synapses(high_inhib_pop, target=high_inhib_pop, on_pre="s_GABA += 1", delay=0.0 * b2.ms)
    high_syn_inhib2inhib.connect(condition="i!=j", p=1.0)
    
    # set the connections: extern2inhib
    high_input_ext2inhib = PoissonInput(target=high_inhib_pop, target_var="s_AMPA",
                                N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    
    # 高级区域兴奋神经元
    high_excit_pop = NeuronGroup(N_excit_high, model=high_excit_lif,
                            threshold="v>v_firing_threshold_excit", reset="v=v_reset_excit; x+=1",
                            refractory=t_abs_refract_excit, method="rk2")
    
    # initialize with random voltages:
    high_excit_pop.v = np.random.uniform(v_reset_excit / b2.mV, high=v_firing_threshold_excit / b2.mV,
                                    size=N_excit_high) * b2.mV
    high_excit_pop.I_stim = 0. * b2.namp
    
    # set the connections: extern2excit
    high_input_ext2excit = PoissonInput(target=high_excit_pop, target_var="s_AMPA",
                                N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)
    
    # set the connections: inhibitory to excitatory
    high_syn_inhib2excit = Synapses(high_inhib_pop, target=high_excit_pop, on_pre="s_GABA += 1")
    high_syn_inhib2excit.connect(p=1.0)
    
    # set the connections: excitatory to inhibitory NMDA connections
    high_syn_excit2inhib = Synapses(high_excit_pop, high_inhib_pop,
                            model="s_NMDA_total_post = s_NMDA_pre : 1 (summed)", method="rk2")
    high_syn_excit2inhib.connect(p=1.0)
    
    
    # 感觉到高级区域的连接，4倍压缩
    sensory2high = Synapses(excit_pop, target=high_excit_pop, on_pre="s_AMPA += 1")
    # 第一个ring每4个神经元连接每一行，(0~3)连接到(0~N/4)，(4~8)连接到(N/4~2N/4), ……
    sensory2high.connect(j='k for k in range((i//compress_factor)*N_excit_high_1d, ((i//compress_factor)+1)*N_excit_high_1d) if i < N_excitatory')
    # 第二个ring每4个神经元连接每一列，(0, N/4, 2N/4, 3N/4……), (0+1, N/4+1, 2N/4+1, 3N/4+1……),
    sensory2high.connect(j='(i-N_excitatory) // compress_factor + k*N_excit_high_1d for k in range(N_excit_high_1d) if i >= N_excitatory')

    
    # 高级到感觉区域的连接
    high2sensory = Synapses(high_excit_pop, target=excit_pop, on_pre="s_AMPA += 1")
    # 每一行(0~N/4)，(N/4~2N/4), ……连接第一个ring的每4个神经元
    high2sensory.connect(j='(i // N_excit_high_1d)*compress_factor + k for k in range(compress_factor)')
    # 每一列(0, N, 2N, 3N……), (0+1, N+1, 2N+1, 3N+1……), ……连接第二个ring的每4个神经元
    high2sensory.connect(j='(i % N_excit_high_1d)*compress_factor + N_excitatory + k for k in range(compress_factor)')
    

# =============================================================================
# 开始！
# =============================================================================
    # precompute the weight profile for the recurrent population
    presyn_weight_kernel = structured_connectivity(N_excitatory, Jpos_excit2excit, sigma_weight_profile)
    fft_presyn_weight_kernel = rfft(presyn_weight_kernel)
      
    # set the STRUCTURED recurrent input. use a network_operation
    @network_operation()
    def update_nmda_sum():
        for i in range(2):
            fft_s_NMDA = rfft(excit_pop[i*N_excitatory: (i+1)*N_excitatory].s_NMDA)
            fft_s_NMDA_total = np.multiply(fft_presyn_weight_kernel, fft_s_NMDA)
            s_NMDA_tot = irfft(fft_s_NMDA_total)
            excit_pop[i*N_excitatory: (i+1)*N_excitatory].s_NMDA_total_ = s_NMDA_tot
            

    high_weight_kernel = structured_connectivity_2d(N_excit_high_1d, Jpos_high, sigma_weight_profile)
    fft_high_kernel = rfft2(high_weight_kernel)


    @network_operation()
    def update_nmda_high():
        high_nmda = np.array(high_excit_pop.s_NMDA).reshape(N_excit_high_1d, -1)
        fft_s_NMDA_2d = rfft2(high_nmda)
        fft_s_NMDA_total_2d = np.multiply(fft_high_kernel, fft_s_NMDA_2d)
        s_NMDA_tot_2d = irfft2(fft_s_NMDA_total_2d)
        high_excit_pop.s_NMDA_total_ = s_NMDA_tot_2d.reshape(-1)
        
    
    feature1 = 120.0
    feature2 = 240.0
    t_stimulus_end = t_stimulus_start + t_stimulus_duration

    # stim 1/3，2/3
    stim1d = gaussian_input(N_excitatory, feature1, stimulus_width_deg)
    stim2d = gaussian_input(N_excitatory, feature1, stimulus_width_deg)
    sensory_stimulus1 = np.hstack([stim1d, stim2d])
    stim1d = gaussian_input(N_excitatory, feature2, stimulus_width_deg) 
    stim2d = gaussian_input(N_excitatory, feature2, stimulus_width_deg)
    sensory_stimulus2 = np.hstack([stim1d, stim2d])
    sensory_stimulus = sensory_stimulus1 + sensory_stimulus2 
    
    if direct_input:
        high_stimulus1 = direct_connect(N_excit_high_1d, [feature1, feature2], stimulus_width_deg)
        high_stimulus2 = direct_connect(N_excit_high_1d, [feature2, feature1], stimulus_width_deg)
        high_stimulus = (high_stimulus1 + high_stimulus2).reshape(-1)
    else:
        high_stimulus = 0
    
    @network_operation(dt=1 * b2.ms)
    def stimulate_network(t):
        if t >= t_stimulus_start and t < t_stimulus_end:
            excit_pop.I_stim = sensory_stimulus * stimulus_strength
            high_excit_pop.I_stim = high_stimulus * stimulus_strength
        else:
            excit_pop.I_stim = 0. * b2.namp
            high_excit_pop.I_stim = 0. * b2.namp
    
    # collect data of a subset of neurons:
    spike_monitor_sensory = SpikeMonitor(excit_pop)
    spike_monitor_high = SpikeMonitor(high_excit_pop)
    b2.run(sim_time)
    return spike_monitor_sensory, spike_monitor_high


def getting_started(
    N=720, num_trial=0,
    scaling_sensory_inhib=1, scaling_sensory_excit=1,
    scaling_high_inhib=1, scaling_high_excit=1,
    jpos=1.0, Jpos_high=1.0, sigma_weight_profile = 10.,
    strength=1, direct_input=True):
    compressed_factor = 4
    b2.defaultclock.dt = 0.1 * b2.ms
    spike_monitor_sensory, spike_monitor_high = simulate_wm(
        N_excitatory=int(N), N_inhibitory=int(N/4), sim_time=500. * b2.ms,
        t_stimulus_start=100 * b2.ms, t_stimulus_duration=100 * b2.ms,
        stimulus_strength=strength * 1.0 * b2.namp, 
        scaling_sensory_inhib=scaling_sensory_inhib, scaling_sensory_excit=scaling_sensory_excit, 
        scaling_high_inhib=scaling_high_inhib, scaling_high_excit=scaling_high_excit,
        Jpos_excit2excit=jpos, Jpos_high=Jpos_high, sigma_weight_profile=sigma_weight_profile,
        direct_input=direct_input)
    plot_network_activity(spike_monitor_sensory, t_min=0. * b2.ms, num_trial=str(num_trial))
    plot_network_activity(spike_monitor_high, t_min=0. * b2.ms, if3d=True, num_neuron=N//compressed_factor, num_trial=str(num_trial))
    gcPython.collect()