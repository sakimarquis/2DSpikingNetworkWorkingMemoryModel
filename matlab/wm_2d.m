% This code is provided by Ziqiang Wei and modified by me to produce 2d neurons
% If you have any problem in this version, please contact sakimarquis@qq.com
% 
% GPL3 License
% 
% Copyright (c) 2016 Ziqiang Wei
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.


function wm_2d(datanum, n_stimuli, root_dir_now, pyr_input, t_end, t_stimuli, mini_diff)


%% Intro:
%  This code is for the working memory capacity paper
%  Wei Z., Wang X.-J., and Wang D.H. (2012), JNS
%  
%  All the parameters are named the same as those in the paper with the
%  following abbreviations, which indicate the neuron's group
%  
%  Abbreviations:
%  'pyr_' : pyramidal cells
%  'inh_' : interneurons
%
%  Parameters: 
%  'root_dir_now' is the directory of the recording data
%                 This must be a full directory, e.g. '~/Desktop/'
%
%  'datanum' is an index of the simulation
%  't_end' is the length of the simulation (in ms)
%  'n_stimuli' is the number of the stimuli
%  't_stimuli' is the type of the stimuli: 'u' (uniform) or 'r' (random)
%  'mini_diff' is the minimum distance between the neighbouring stimuli (deg).
%                 This parameter must be given if type of the stimuli is
%                 'r' (random). For uniform case, 'mini_diff' can be any
%                 number.
%  'pyr_input' is [sample_on, sample_off] times ([250, 500]) in Wei et al., 2012
%  
%  Simulation data:
%  Files:
%    pyramidal cells: [root_dir,'pyr_cell_',num2str(datanum),'.txt'];
%    interneuorns: [root_dir,'inh_cell_',num2str(datanum),'.txt'];
%    first column: index of the neuron that fires
%    second column: the firing time
%  
%  The code was based on a previous version by Furman M. (2008)
%  Author: Wei Z.
%  Contact: weiz@janelia.hhmi.org
%  Current version date: 09/19/2012

%% Part 0: checking inputs & showing the recoding pathway:

    % if nargin ~= 6
    %     error ('Please check the number of parameters~~~ :(');
    % end
        
    if nargin < 2
        error('Please check the number of parameters');
    end

    if nargin == 2
        %n_stimuli = 4;
        root_dir_now = './';
        pyr_input = [200, 300]; % [200, 300]
        t_end = 1000; %1200;
        t_stimuli = 'r';
        mini_diff = 80; % 24
    end

    % if isempty(root_dir_now)
    %     error ('Please give the directory correctly ~~~~~ :P');
    % end
    
    if ~exist(root_dir_now,'dir')
        mkdir (root_dir_now)
    end

    root_dir = root_dir_now;
    
    to_print = 0:999;
    
%% Part 1: simulation parameters:

% number of pyramidal cells and interneurons
    excit_1d = 4096;
    inhib_1d = excit_1d / 4;
    compress = 2;
    excit_2d = (excit_1d / compress)^2;
    inhib_2d = excit_2d / 4;
    
    pyr_group1 = 1: excit_1d;
    pyr_group2 = excit_1d+1: excit_1d*2;
    pyr_low = 1: excit_1d*2;
    pyr_high = excit_1d*2 + 1: excit_1d*2 + excit_2d;
    
    inh_group1 = 1: inhib_1d;
    inh_group2 = inhib_1d+1: inhib_1d*2;
    inh_low = 1: inhib_1d*2;
    inh_high = inhib_1d*2 + 1: inhib_1d*2 + inhib_2d;
    
    pyr_Nneurons = excit_1d * 2 + excit_2d;
    inh_Nneurons = inhib_1d * 2 + inhib_2d;

    scaling = excit_1d / excit_2d;
    
    dt = 0.05;
    
    G_high2low_AMPA = 0.001;
    G_low2high_AMPA = 0.001 * 5;
    
% Transmission delays:
    pyr_trans_delays = zeros(pyr_Nneurons, 1);
    inh_trans_delays = zeros(inh_Nneurons, 1);


% =========================================================================
% Membrane Parameters
% =========================================================================

% ** Pyramidal Cells **
    pyr_Cm = 0.5; % [nF]
    pyr_g_leak = 0.025; % [microSiemens] 
    pyr_Vm_leak = -70; % [mV]
    pyr_Vm_thresh = -50; % [mV]
    pyr_Vm_reset = -60; % [mV]
    pyr_tau_ref = 2; % [ms]

% ** Interneurons **
    inh_Cm = 0.2; % [nF]
    inh_g_leak = 0.020; % [microS] 
    inh_Vm_leak = -70; % [mV]
    inh_Vm_thresh = -50; % [mV]
    inh_Vm_reset = -60; % [mV]
    inh_tau_ref = 1; % [ms]

% =========================================================================
% Synapese parameters
% =========================================================================

% ** AMPA **
    syn_Ve_AMPA = 0; % [mV]
    syn_tau_s_AMPA = 2; % [ms]
    syn_alpha_s_AMPA = 1; % NOTE: what the fuck is this alpha?

% ** NMDA **
    syn_Ve_NMDA = 0; % [mV]
    syn_Mg_concentration = 1; % [mM]
    syn_tau_x_NMDA = 2; % [ms]
    syn_alpha_x_NMDA = 1; %
    syn_tau_s_NMDA = 100; % [ms]
    syn_alpha_s_NMDA = 0.5; % [1/ms]

% ** GABA **
    syn_Ve_GABA = -70; % [mV]
    syn_tau_s_GABA = 10; % [ms]
    syn_alpha_s_GABA = 1; % [ms] 

% =========================================================================
% Noise
% =========================================================================

% External Poisson spiking train (noise)
    pyr_ext_noise_rate = 1.0; % [1/ms];
    % equivalently: 1000 synapses with 1.0Hz input;
    inh_ext_noise_rate = 1.0;
 
% =========================================================================
% Maximal Conductances, Connectivity Coefficients
% =========================================================================

% Maximal Conductances
    popA_G_pyr2pyr_NMDA = 0.0001905; % [microS]
    popA_G_pyr2inh_NMDA = 0.0001460; % [microS]

    popA_G_inh2pyr_GABA = 0.0006681; % [microS]
    popA_G_inh2inh_GABA = 0.0005120; % [microS]

% External (noise) to pyramidal cells and interneurons:
    pyr_ext_noise_g_AMPA = 0.00248; % [microS]
    inh_ext_noise_g_AMPA = 0.0019; % [microS]
    
    % Connectivity, E->E
    %sigma_pyr2pyr = 9.25;
    %J_plus_pyr2pyr = 2.72;
    
    %sigma_pyr2pyr = 11.25;
    %J_plus_pyr2pyr = 3.62;
    
    J_plus_pyr2pyr = 4;
    sigma_pyr2pyr = 5;
    J_plus_high = 12;
    sigma_high = 10;
    J_plus_high_ei = 3;
    sigma_high_ei = 30;
    
    
% =========================================================================
% Input stimuli - 'free style'
% =========================================================================

%     if n_stimuli <=0
%         error ('Please use a postive number for the number of stimuli in the cue');
%     end
% 
%     if t_stimuli == 'u'
%         theta_stim = 180/n_stimuli:360/n_stimuli:360-180/n_stimuli;
%     elseif t_stimuli == 'r'
%         theta_stim = get_rand_stimuli(n_stimuli,mini_diff);
%     elseif t_stimuli == 'c'
%         theta_stim = mini_diff;
%     else
%         error ('Please use the correct type of stimuli within "u" and "r"');
%     end

%     fname_input = [root_dir,'input_',num2str(datanum),'.txt'];
%     f_input = fopen(fname_input, 'w');
%     fprintf(f_input, '%f,', theta_stim);
%     fclose(f_input);

    sigma_stim = 2;
    max_stim_current = 0.4;
    max_stim_high = 0.6;
    
    % Generate input signal
%     pyr_profile = zeros(excit_1d, 1);
%     for n_stim = 1:n_stimuli
%         pyr_profile = pyr_profile + max_stim_current*...
%             circular_gaussian(excit_1d, sigma_stim, theta_stim(n_stim));
%     end

    features_1d = transpose((1:360/n_stimuli:360) + 360/n_stimuli/2 - 1);
    
    pyr_profile = max_stim_current * gaussian_input(excit_1d, features_1d, sigma_stim);
    pyr_profile_1d = [pyr_profile; pyr_profile];
    
    features_2d = repmat(features_1d, 1, 2); %[[45, 45]; [135, 135]; [225, 225]; [315, 315]];
    width = 10;
    pyr_profile_2d = max_stim_high * gaussian_input_2d(sqrt(excit_2d), features_2d, width);
    
    pyr_profile = [pyr_profile_1d; pyr_profile_2d];
    
    
    % start point and end point of the input
    % pyr_input = [250, 500] ;

    
% =========================================================================
% Initial value
% =========================================================================

    pyr_popA_init_Vm = -51;% [mV]
    pyr_popA_init_x_NMDA = 0.1;  %
    pyr_popA_init_s_NMDA = 0.05 ;
    pyr_popA_init_s_AMPA = 0.05 ;

    inh_popA_init_Vm = -51;% [mV]
    inh_popA_init_s_GABA = 0.5; %

    pyr_popA_init_s_AMPA_ext_noise = .6;
    inh_popA_init_s_AMPA_ext_noise = .6;

    % disp('All inputs are valid! & Let the simulation begin :)');

%% Random number seed
    rand_seed = sum(100*clock);
    
% old version of random seed generator
%    rand('state',rand_seed);

% new version of randoom seed generator
    rng (rand_seed);

%% Part 2: Initial Network for simulations

% =========================================================================
% Transmission delay 1d
% =========================================================================

    % excit pop
    pyr_popA_transmission_delay_t_steps = max(round(pyr_trans_delays/dt), 1);
    pyr_popA_delay_bank_size = max(pyr_popA_transmission_delay_t_steps);
    pyr_popA_WhoFired_bank = zeros(pyr_Nneurons, pyr_popA_delay_bank_size);
    pyr_popA_ind_now_in_bank = 0;

    % inhib pop
    inh_popA_transmission_delay_t_steps = max(round(inh_trans_delays/dt), 1);
    inh_popA_delay_bank_size = max(inh_popA_transmission_delay_t_steps);
    inh_popA_WhoFired_bank = zeros(inh_Nneurons, inh_popA_delay_bank_size);
    inh_popA_ind_now_in_bank = 0;

% =========================================================================
% Initialize 1d
% =========================================================================

    pyr_x_NMDA = pyr_popA_init_x_NMDA(ones(pyr_Nneurons, 1));
    pyr_s_NMDA = pyr_popA_init_s_NMDA(ones(pyr_Nneurons, 1));
    pyr_s_AMPA = pyr_popA_init_s_AMPA(ones(pyr_Nneurons, 1));
    inh_s_GABA = inh_popA_init_s_GABA(ones(inh_Nneurons, 1));

    pyr_Vm_prev = pyr_popA_init_Vm(ones(pyr_Nneurons, 1));
    pyr_Vm_new = pyr_popA_init_Vm(ones(pyr_Nneurons, 1));
    pyr_ext_noise_s_AMPA = pyr_popA_init_s_AMPA_ext_noise(ones(pyr_Nneurons, 1));

    inh_Vm_prev = inh_popA_init_Vm(ones(inh_Nneurons, 1));
    inh_Vm_new = inh_popA_init_Vm(ones(inh_Nneurons, 1));
    inh_ext_noise_s_AMPA = inh_popA_init_s_AMPA_ext_noise(ones(inh_Nneurons, 1));

    pyr_WhoFired = [];
    pyr_SpikeTimes = []; %#ok<NASGU> % ignore this warning
    pyr_LastTimeEachFired = -pyr_tau_ref(ones(pyr_Nneurons, 1)); % the last spike of each neuron.

    inh_WhoFired = [];
    inh_SpikeTimes = []; %#ok<NASGU> % ignore this warning
    inh_LastTimeEachFired = -inh_tau_ref(ones(pyr_Nneurons, 1)); % the last spike of each neuron.

    % Connectivity vectors for recurrent exc->exc connections
    W_pyr2pyr_fft = structured_connectivity(excit_1d, J_plus_pyr2pyr, sigma_pyr2pyr);
    W_pyr2pyr_fft_2d = structured_connectivity_2d(sqrt(excit_2d), J_plus_high, sigma_high);
    W_pyr2inh_fft_2d = structured_connectivity_2d(sqrt(inhib_2d), J_plus_high_ei, sigma_high_ei);
    W_inh2pyr_fft_2d = structured_connectivity_2d(sqrt(excit_2d), J_plus_high_ei, sigma_high_ei);

% =========================================================================
% Recording files:
% =========================================================================

    fname_pyr = [root_dir, 'pyr_cell_', num2str(datanum), '.txt'];
    fname_inh = [root_dir, 'inh_cell_', num2str(datanum), '.txt'];
    fname_current = [root_dir, 'current_', num2str(datanum), '.txt'];
    
    if exist(fname_pyr,'file')
        delete(fname_pyr)
    end

    if exist(fname_inh,'file')
        delete(fname_inh)
    end
    
    f_pyr = fopen(fname_pyr, 'a', 'native');
    f_inh = fopen(fname_inh, 'a', 'native');
    f_current = fopen(fname_current, 'a', 'native');
    
    fprintf(f_current, ['t,', ...
        'pyr_min_low_leak,', 'pyr_min_low_NMDA,', 'pyr_min_low_GABA,', 'pyr_min_low_noise,', ...
        'pyr_min_high_leak,', 'pyr_min_high_NMDA,', 'pyr_min_high_GABA,', 'pyr_min_high_noise,', ...
        'min_low_recieve,', 'min_high_recieve,',  ...
        'pyr_mean_low_leak,', 'pyr_mean_low_NMDA,', 'pyr_mean_low_GABA,', 'pyr_mean_low_noise,', ...
        'pyr_mean_high_leak,', 'pyr_mean_high_NMDA,', 'pyr_mean_high_GABA,', 'pyr_mean_high_noise,', ...
        'mean_low_recieve,', 'mean_high_recieve,', ...
        'inh_mean_low_leak,', 'inh_mean_low_NMDA,', 'inh_mean_low_GABA,', 'inh_mean_low_noise,', ...
        'inh_mean_high_leak,', 'inh_mean_high_NMDA,', 'inh_mean_high_GABA,', 'inh_mean_high_noise,', '\n']);

    disp ('Setup for recoding data is done ....');
    
%% Part 3:  simulation loop
    for current_time = 0:dt:t_end %

        % Update indices of firing bank:
        pyr_popA_ind_now_in_bank = mod(pyr_popA_ind_now_in_bank, ...
            pyr_popA_delay_bank_size) + 1;
        inh_popA_ind_now_in_bank = mod(inh_popA_ind_now_in_bank, ...
            inh_popA_delay_bank_size) + 1;

        % Update gating variables of NMDA, AMPA and GABA
        pyr_x_NMDA = RK2_simple_linear_eq(pyr_x_NMDA, dt, -1/syn_tau_x_NMDA, 0);
        pyr_x_NMDA(pyr_WhoFired) = pyr_x_NMDA(pyr_WhoFired) + syn_alpha_x_NMDA;
        pyr_s_NMDA = RK2_4sNMDA(pyr_s_NMDA, dt, -1/syn_tau_s_NMDA, syn_alpha_s_NMDA*pyr_x_NMDA);
        
        pyr_s_AMPA = RK2_simple_linear_eq(pyr_s_AMPA, dt, -1/syn_tau_s_AMPA, 0);
        pyr_s_AMPA(pyr_WhoFired) = pyr_s_AMPA(pyr_WhoFired) + syn_alpha_s_AMPA;
        
        inh_s_GABA = RK2_simple_linear_eq(inh_s_GABA, dt, -1/syn_tau_s_GABA, 0);
        inh_s_GABA(inh_WhoFired) = inh_s_GABA(inh_WhoFired) + syn_alpha_s_GABA;
        

% =========================================================================
% Pyramidal cells update:
% =========================================================================
        
        % inject stimulus(input)
        pyr_I_applied = 0;
        k_time_period = logical(sum(current_time < pyr_input))...
            *logical(sum(current_time >= pyr_input))*...
            sum(current_time >= pyr_input);
        if k_time_period > 0
            pyr_I_applied = pyr_profile;
        end
           
        
        % i2e, target: inh_s_GABA
        pyr_I_GABA_group1 = popA_G_inh2pyr_GABA * sum(inh_s_GABA(inh_group1)) ...
            * (pyr_Vm_prev(pyr_group1) - syn_Ve_GABA);
        
        pyr_I_GABA_group2 = popA_G_inh2pyr_GABA * sum(inh_s_GABA(inh_group2)) ...
            * (pyr_Vm_prev(pyr_group2) - syn_Ve_GABA);

%         pyr_I_GABA_high = scaling * popA_G_inh2pyr_GABA * sum(inh_s_GABA(inh_high)) ...
%             * (pyr_Vm_prev(pyr_high) - syn_Ve_GABA);
         
        % s = reshape(1:16, 16, 1)
        %  reshape(repmat(reshape(repmat(s, 1, 2).', 8, 4), 2, 1), 8, 8)
        s_GABA_2d = repmat(reshape(repmat(inh_s_GABA(inh_high), 1, 2).', sqrt(excit_2d), sqrt(inhib_2d)), 2, 1);
        s_GABA_2d = 1/4 * reshape(s_GABA_2d, sqrt(excit_2d), sqrt(excit_2d));
        
        w_dot_s_GABA_2d = reshape(ifft2(W_inh2pyr_fft_2d .* fft2(s_GABA_2d)), excit_2d, 1);
        pyr_I_GABA_high = scaling * popA_G_inh2pyr_GABA * inhib_2d * w_dot_s_GABA_2d ...
            .* (pyr_Vm_prev(pyr_high) - syn_Ve_GABA);

        pyr_I_GABA = [pyr_I_GABA_group1; pyr_I_GABA_group2; pyr_I_GABA_high];

        % Update recurrent currents; thresholding etc.
        pyr_Vm_prev = pyr_Vm_new; % store voltage from the previous step
        
        
        % =================================================================
        % External AMPA current(noise)
        % =================================================================
        rand_vec = rand(pyr_Nneurons, 1) < dt * pyr_ext_noise_rate;
        pyr_ext_noise_s_AMPA = RK2_simple_linear_eq(pyr_ext_noise_s_AMPA, ...
            dt, -1/syn_tau_s_AMPA, syn_alpha_s_AMPA * rand_vec);
        
        pyr_I_noise_low = pyr_ext_noise_g_AMPA * pyr_ext_noise_s_AMPA(pyr_low) ...
            .* (pyr_Vm_prev(pyr_low) - syn_Ve_AMPA);
        pyr_I_noise_high = pyr_ext_noise_g_AMPA * pyr_ext_noise_s_AMPA(pyr_high) ...
            .* (pyr_Vm_prev(pyr_high) - syn_Ve_AMPA);
        
        pyr_I_noise = [pyr_I_noise_low; pyr_I_noise_high];   
        
        % leaky
        pyr_I_leak = pyr_g_leak * (pyr_Vm_prev - pyr_Vm_leak);
        
        
        % =================================================================
        % e2e, target: pyr_s_NMDA; Multiply weight matrix by s vector using FFT:
        % =================================================================
        w_dot_s_NMDA_group1 = ifft(W_pyr2pyr_fft .* fft(pyr_s_NMDA(pyr_group1)));
        pyr_I_NMDA_group1 = popA_G_pyr2pyr_NMDA * excit_1d * w_dot_s_NMDA_group1 ...
            .* (pyr_Vm_prev(pyr_group1) - syn_Ve_NMDA) ...
            ./ (1 + syn_Mg_concentration * exp(-0.062*pyr_Vm_prev(pyr_group1)) / 3.57);
        
        w_dot_s_NMDA_group2 = ifft(W_pyr2pyr_fft .* fft(pyr_s_NMDA(pyr_group2)));
        pyr_I_NMDA_group2 = popA_G_pyr2pyr_NMDA * excit_1d * w_dot_s_NMDA_group2 ...
            .* (pyr_Vm_prev(pyr_group2) - syn_Ve_NMDA) ...
            ./ (1 + syn_Mg_concentration * exp(-0.062*pyr_Vm_prev(pyr_group2)) / 3.57);
        
        s_NMDA_2d = reshape(pyr_s_NMDA(pyr_high), sqrt(excit_2d), sqrt(excit_2d));
        w_dot_s_NMDA_high = reshape(ifft2(W_pyr2pyr_fft_2d .* fft2(s_NMDA_2d)), excit_2d, 1);
        pyr_I_NMDA_high = scaling * popA_G_pyr2pyr_NMDA * excit_2d * w_dot_s_NMDA_high ...
            .* (pyr_Vm_prev(pyr_high) - syn_Ve_NMDA) ...
            ./ (1 + syn_Mg_concentration * exp(-0.062*pyr_Vm_prev(pyr_high)) / 3.57);

        pyr_I_NMDA = [pyr_I_NMDA_group1; pyr_I_NMDA_group2; pyr_I_NMDA_high];
        
        % =================================================================
        % connection between low and high brain area via AMPA
        % =================================================================
        s_AMPA_high_to_low = reshape(pyr_s_AMPA(pyr_high), sqrt(excit_2d), sqrt(excit_2d));
        s_AMPA_high_to_group1 = reshape(mean(s_AMPA_high_to_low, 1), sqrt(excit_2d), 1);
        s_AMPA_high_to_group2 = mean(s_AMPA_high_to_low, 2);
        s_AMPA_high_to_group1 = reshape(transpose(repmat(...
            s_AMPA_high_to_group1, 1, compress)), excit_1d, 1);
        s_AMPA_high_to_group2 = reshape(transpose(repmat(...
            s_AMPA_high_to_group2, 1, compress)), excit_1d, 1);
        pyr_I_high_to_group1 = G_high2low_AMPA * s_AMPA_high_to_group1 ...
            .* (pyr_Vm_prev(pyr_group1) - syn_Ve_AMPA);
        pyr_I_high_to_group2 = G_high2low_AMPA * s_AMPA_high_to_group2 ...
            .* (pyr_Vm_prev(pyr_group2) - syn_Ve_AMPA);
        
        s_AMPA_group1_to_high = pyr_s_AMPA(pyr_group1); % (n_1d �� 1)
        s_AMPA_group1_to_high = reshape(s_AMPA_group1_to_high, compress, sqrt(excit_2d)); % (2 �� sqrt(n_2d))
        s_AMPA_group1_to_high = transpose(s_AMPA_group1_to_high); % (sqrt(n_2d) �� 2)
        s_AMPA_group1_to_high = mean(s_AMPA_group1_to_high, 2); % (sqrt(n_2d) �� 1)
        s_AMPA_group1_to_high = repmat(s_AMPA_group1_to_high, sqrt(excit_2d), 1); % (n_2d �� 1)
        pyr_I_group1_to_high = G_low2high_AMPA * s_AMPA_group1_to_high ...
            .* (pyr_Vm_prev(pyr_high) - syn_Ve_AMPA);
        
        s_AMPA_group2_to_high = pyr_s_AMPA(pyr_group2); % (n_1d �� 1)
        s_AMPA_group2_to_high = reshape(s_AMPA_group2_to_high, compress, sqrt(excit_2d)); % (2 �� sqrt(n_2d))
        s_AMPA_group2_to_high = transpose(s_AMPA_group2_to_high); % (sqrt(n_2d) �� 2)
        s_AMPA_group2_to_high = mean(s_AMPA_group2_to_high, 2); % (sqrt(n_2d) �� 1)
        s_AMPA_group2_to_high = reshape(s_AMPA_group2_to_high, 1, sqrt(excit_2d)); % (1 �� sqrt(n_2d))
        s_AMPA_group2_to_high = repmat(s_AMPA_group2_to_high, sqrt(excit_2d), 1); % (sqrt(n_2d) �� sqrt(n_2d))
        s_AMPA_group2_to_high = reshape(s_AMPA_group2_to_high, excit_2d, 1); % (n_2d �� 1)
        pyr_I_group2_to_high = G_low2high_AMPA * s_AMPA_group2_to_high ...
            .* (pyr_Vm_prev(pyr_high) - syn_Ve_AMPA);
        
        pyr_I_low_to_high = pyr_I_group1_to_high + pyr_I_group2_to_high;
        
        pyr_I_connect = [pyr_I_high_to_group1; pyr_I_high_to_group2; pyr_I_low_to_high];
        
        
        % TOTAL CURRENT:
        pyr_I_total = - pyr_I_leak - pyr_I_NMDA - pyr_I_GABA ...
            - pyr_I_noise + pyr_I_applied - pyr_I_connect;

        % Membrate voltage:
        pyr_Vm_new = pyr_Vm_new + 1/pyr_Cm * dt * pyr_I_total;
        pyr_Vm_new((current_time - pyr_LastTimeEachFired) < pyr_tau_ref) = ...
            pyr_Vm_reset;
        pyr_WhoFiredNow = find(pyr_Vm_new > pyr_Vm_thresh);
        pyr_SpikeTimes = current_time + dt * ... 
            ((pyr_Vm_thresh - pyr_Vm_prev(pyr_WhoFiredNow)) ./ ... 
            (pyr_Vm_new(pyr_WhoFiredNow) - pyr_Vm_prev(pyr_WhoFiredNow)));
        pyr_Vm_new(pyr_WhoFiredNow) = pyr_Vm_reset;
        pyr_LastTimeEachFired(pyr_WhoFiredNow) = pyr_SpikeTimes;

        % Firing and transmission delays:
        pyr_WhoFired = find(pyr_popA_WhoFired_bank(:, pyr_popA_ind_now_in_bank));
        pyr_popA_WhoFired_bank(:, pyr_popA_ind_now_in_bank) = 0;

        non_circular_indices4storage = pyr_popA_ind_now_in_bank + ...
            pyr_popA_transmission_delay_t_steps(pyr_WhoFiredNow);

        circular_indices4storage = mod(non_circular_indices4storage-1,...
            pyr_popA_delay_bank_size) + 1;
        linear_indices4storage = pyr_WhoFiredNow + ...
            (circular_indices4storage - 1) * pyr_Nneurons;
        pyr_popA_WhoFired_bank(linear_indices4storage) = 1;

        % Store spikes
        n_fired = size(pyr_WhoFiredNow, 1);
        for k_fired = 1:n_fired
            fprintf(f_pyr, '%f,%f\n', pyr_WhoFiredNow(k_fired), pyr_SpikeTimes(k_fired));
        end

% =========================================================================
% Interneurons update:
% =========================================================================
        
        % =================================================================
        % e2i, target: pyr_s_NMDA
        % =================================================================
        inh_I_NMDA_group1 = popA_G_pyr2inh_NMDA * sum(pyr_s_NMDA(pyr_group1)) ...
            * (inh_Vm_prev(inh_group1) - syn_Ve_NMDA) ./ (1 + syn_Mg_concentration ...
            * exp(-0.062*inh_Vm_prev(inh_group1)) / 3.57);

        inh_I_NMDA_group2 = popA_G_pyr2inh_NMDA * sum(pyr_s_NMDA(pyr_group2)) ...
        * (inh_Vm_prev(inh_group2) - syn_Ve_NMDA) ./ (1 + syn_Mg_concentration ...
        * exp(-0.062*inh_Vm_prev(inh_group2)) / 3.57);
    
%         inh_I_NMDA_high = scaling * popA_G_pyr2inh_NMDA * sum(pyr_s_NMDA(pyr_high)) ...
%         * (inh_Vm_prev(inh_high) - syn_Ve_NMDA) ./ (1 + syn_Mg_concentration ...
%         * exp(-0.062*inh_Vm_prev(inh_high)) / 3.57);

        % reshape(sum(reshape(reshape(sum(reshape(q, 2, 32), 1), 4, 8).', 2, 16),1), 4, 4)
        s_NMDA_2d = sum(reshape(pyr_s_NMDA(pyr_high), 2, excit_2d/2), 1);
        s_NMDA_2d = sum(reshape(reshape(s_NMDA_2d, sqrt(inhib_2d), 2*sqrt(inhib_2d)).', 2, inhib_2d), 1);
        s_NMDA_2d = reshape(s_NMDA_2d, sqrt(inhib_2d), sqrt(inhib_2d));
        
        w_dot_s_NMDA_2d = reshape(ifft2(W_pyr2inh_fft_2d .* fft2(s_NMDA_2d)), inhib_2d, 1);
        inh_I_NMDA_high = scaling * popA_G_pyr2inh_NMDA * excit_2d * w_dot_s_NMDA_2d ...
            .* (inh_Vm_prev(inh_high) - syn_Ve_NMDA) ./ (1 + syn_Mg_concentration ...
            .* exp(-0.062*inh_Vm_prev(inh_high)) / 3.57);
   
    
        inh_I_NMDA = [inh_I_NMDA_group1; inh_I_NMDA_group2; inh_I_NMDA_high];
        
        % =================================================================
        % i2i, target: inh_s_GABA
        % =================================================================
        inh_I_GABA_group1 = popA_G_inh2inh_GABA * sum(inh_s_GABA(inh_group1)) ...
            .* (inh_Vm_prev(inh_group1) - syn_Ve_GABA);

        inh_I_GABA_group2 = popA_G_inh2inh_GABA * sum(inh_s_GABA(inh_group2)) ...
        .* (inh_Vm_prev(inh_group2) - syn_Ve_GABA);
    
        inh_I_GABA_high = scaling * popA_G_inh2inh_GABA * sum(inh_s_GABA(inh_high)) ...
        .* (inh_Vm_prev(inh_high) - syn_Ve_GABA);
        
        inh_I_GABA = [inh_I_GABA_group1; inh_I_GABA_group2; inh_I_GABA_high];

        % Update recurrent currents; thresholding etc.
        inh_Vm_prev = inh_Vm_new; % store voltage from the previous step
        
        % =================================================================
        % External AMPA current
        % =================================================================
        rand_vec = rand(inh_Nneurons,1) < dt * inh_ext_noise_rate;
        inh_ext_noise_s_AMPA = RK2_simple_linear_eq(inh_ext_noise_s_AMPA, ...
            dt, -1/syn_tau_s_AMPA, syn_alpha_s_AMPA*rand_vec);
        
        inh_I_noise_low = inh_ext_noise_g_AMPA * inh_ext_noise_s_AMPA(inh_low) ...
            .* (inh_Vm_prev(inh_low) - syn_Ve_AMPA);
        inh_I_noise_high = inh_ext_noise_g_AMPA * inh_ext_noise_s_AMPA(inh_high) ...
            .* (inh_Vm_prev(inh_high) - syn_Ve_AMPA);
        
        inh_I_noise = [inh_I_noise_low; inh_I_noise_high];
        inh_I_leak = inh_g_leak * (inh_Vm_prev - inh_Vm_leak);

        
        % TOTAL CURRENT:
        inh_I_total = -inh_I_leak - inh_I_NMDA - inh_I_GABA - inh_I_noise;

        
        % Membrate voltage:
        inh_Vm_new = inh_Vm_new + 1/inh_Cm * dt * inh_I_total;
        inh_Vm_new((current_time - inh_LastTimeEachFired) < inh_tau_ref) = ...
            inh_Vm_reset;
        inh_WhoFiredNow = find(inh_Vm_new > inh_Vm_thresh);
        inh_SpikeTimes = current_time+dt * ...
            ((inh_Vm_thresh - inh_Vm_prev(inh_WhoFiredNow)) ...
            ./ (inh_Vm_new(inh_WhoFiredNow) - inh_Vm_prev(inh_WhoFiredNow)));
        inh_Vm_new(inh_WhoFiredNow) = inh_Vm_reset;
        inh_LastTimeEachFired(inh_WhoFiredNow) = inh_SpikeTimes;    


        % Firing and transmission delays:
        inh_WhoFired = find(inh_popA_WhoFired_bank(:,inh_popA_ind_now_in_bank));
        inh_popA_WhoFired_bank(:,inh_popA_ind_now_in_bank) = 0;

        non_circular_indices4storage = inh_popA_ind_now_in_bank ...
            + inh_popA_transmission_delay_t_steps(inh_WhoFiredNow);

        circular_indices4storage = mod(non_circular_indices4storage - 1, ...
            inh_popA_delay_bank_size) + 1;
        linear_indices4storage = inh_WhoFiredNow ... 
            + (circular_indices4storage - 1) * inh_Nneurons;
        inh_popA_WhoFired_bank(linear_indices4storage) = 1;


        % Store spikes:
        n_fired = size(inh_WhoFiredNow, 1);
        for k_fired = 1:n_fired
            fprintf(f_inh, '%f,%f\n', inh_WhoFiredNow(k_fired), inh_SpikeTimes(k_fired));
        end
        
        % store current
        if sum(current_time == to_print)
            fprintf(f_current, '%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n', ...
                current_time, ...
                max(pyr_I_leak(pyr_low)), min(pyr_I_NMDA(pyr_low)), max(pyr_I_GABA(pyr_low)), min(pyr_I_noise(pyr_low)), ...
                max(pyr_I_leak(pyr_high)), min(pyr_I_NMDA(pyr_high)), max(pyr_I_GABA(pyr_high)), min(pyr_I_noise(pyr_high)), ...
                min(pyr_I_connect(pyr_low)), min(pyr_I_connect(pyr_high)),  ...
                mean(pyr_I_leak(pyr_low)), mean(pyr_I_NMDA(pyr_low)), mean(pyr_I_GABA(pyr_low)), mean(pyr_I_noise(pyr_low)), ...
                mean(pyr_I_leak(pyr_high)), mean(pyr_I_NMDA(pyr_high)), mean(pyr_I_GABA(pyr_high)), mean(pyr_I_noise(pyr_high)), ...
                mean(pyr_I_connect(pyr_low)), mean(pyr_I_connect(pyr_high)),  ...
                mean(inh_I_leak(inh_low)), mean(inh_I_NMDA(inh_low)), mean(inh_I_GABA(inh_low)), mean(inh_I_noise(inh_low)), ...
                mean(inh_I_leak(inh_high)), mean(inh_I_NMDA(inh_high)), mean(inh_I_GABA(inh_high)), mean(inh_I_noise(inh_high)));
        end
        
    end

    % Close All the recording files
    fclose(f_inh);
    fclose(f_pyr);

    % Draw figures
    %p = load(fname_pyr);
    %plot(p(:,2), p(:,1), '.k');

    
function stim = gaussian_input(num_neurons, features, width)
    neurons = 0: 360/num_neurons: 360;
    neurons = reshape(neurons(1: num_neurons), num_neurons, 1);
    stim = zeros(num_neurons, 1);
    for idx = 1:size(features)
        feature = features(idx);
        tuning = normpdf(neurons, feature, width);
        tuning = sqrt(2*pi) * width * tuning;
        stim = stim + tuning;
    end

    
function stim = gaussian_input_2d(num_neurons, features, width)
    neurons = 0: 360/num_neurons: 360;
    neurons = neurons(1: num_neurons);
    [x, y] = meshgrid(neurons, neurons);
    neurons_2d = [x(:), y(:)];
    stim = zeros(num_neurons^2, 1);
    for idx = 1:size(features)
        feature = features(idx, :);
        tuning = mvnpdf(neurons_2d, feature, [[width, 0]; [0, width]]);
        tuning = 2 * pi * width^2 * tuning;
        stim = stim + tuning;
    end
    stim = stim / max(stim);


function W_pyr2pyr_fft = structured_connectivity(pyr_Nneurons, J_plus_pyr2pyr, sigma_pyr2pyr)
    pref_dirs_pop = (0: 360/pyr_Nneurons: (360-360/pyr_Nneurons))';
    pref_dir_diff = min(pref_dirs_pop, 360-pref_dirs_pop);
    gauss_con = exp(-0.5 * pref_dir_diff.^2 / sigma_pyr2pyr^2);
    sigma_pyr2pyr = sigma_pyr2pyr / 360;
    tmp2 = sqrt(2*pi) * sigma_pyr2pyr * erf(.5/sqrt(2)/sigma_pyr2pyr);
    tmp1 = (1.-tmp2*J_plus_pyr2pyr) / (1.-tmp2);
    con_vec_not_normalized = tmp1 + (J_plus_pyr2pyr-tmp1) * gauss_con;
    con_vec = con_vec_not_normalized / sum(con_vec_not_normalized);
    W_pyr2pyr_fft = fft(con_vec);
    

function W_pyr2pyr_fft_2d = structured_connectivity_2d(N_excitatory, Jpos, sigma)
    tmp = sqrt(2*pi) * sigma;
    Jneg = (360 - tmp * Jpos) / (360 - tmp);
    [x, y] = meshgrid(0:N_excitatory-1, 0:N_excitatory-1); % x(n��n), y(n��n)
    neurons = [x(:), y(:)]; % (n^2 �� 2)
    neurons = reshape(sqrt(sum(neurons.^2, 2)), N_excitatory, N_excitatory); % (n �� n)
    dist = norm([N_excitatory, N_excitatory]);
    delta_theta = 360 * min(neurons, dist - neurons) / dist; % (n �� n)
    presyn_weight = Jneg + (Jpos - Jneg) * exp(-1/2 * delta_theta.^2 / sigma^2); % (n �� n)
    con_vec = presyn_weight / sum(presyn_weight(:));
    W_pyr2pyr_fft_2d = fft2(con_vec);
    
    
function gaussian_profile_vec = circular_gaussian(n_points,sigma,center_deg)
    % CALCULATE GAUSSIAN PROFILE
    dtheta_pop = 360 / n_points;
    pref_dirs_pop = abs((0: dtheta_pop: (360-dtheta_pop)) - center_deg)';
    pref_dir_diff = min(pref_dirs_pop, 360-pref_dirs_pop);
    gaussian_profile_vec = exp(-0.5*pref_dir_diff.^2 / sigma^2); % 这里的分母的1/(sqrt(2*sigma)*pi)呢？


function y_vec = RK2_4sNMDA(y_vec, delta_t, C_coeff, D_vec)
    % Integration step using Runga-Kutta order 2 method,
    y_vec_temp = y_vec + 0.5*delta_t*(C_coeff*y_vec + D_vec.*(1-y_vec));
    y_vec = y_vec + delta_t*(C_coeff*y_vec_temp + D_vec.*(1-y_vec_temp));
    
    
function y_vec = RK2_simple_linear_eq(y_vec, delta_t, deriv_coeff, delta_fun_vec)
    % Integration step using Runga-Kutta order 2 method
    y_vec = y_vec * (1 + delta_t * deriv_coeff + delta_t^2 * deriv_coeff^2 / 2)...
        + delta_fun_vec;

    
function theta_stim = get_rand_stimuli(n_stimuli, mini_diff)
    % Generating random cue array with minimum distance larger than mini_diff
    theta_stim = nan(n_stimuli, 1);
    theta_stim(1) = 0;
    max_theta = 360 - mini_diff;
    for nstim = 2:n_stimuli
        remain_dist = max_theta - theta_stim(nstim-1) - mini_diff;
        if remain_dist>=0
            theta_stim(nstim) = theta_stim(nstim-1) + mini_diff + rand()*remain_dist;
        else
            break;
        end
    end
    
    if sum(isnan(theta_stim))>0
        theta_stim = get_rand_stimuli(n_stimuli,mini_diff);
    end