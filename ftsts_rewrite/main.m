clear -all
clearvars

% start timer
    tic

% are we caching?
    cache = 1;

% Run Parameters.
    duration = 25 * 1000;   % (ms) duration of simulation
    step_size = 0.1;        % (ms) time step
    num_steps = duration / step_size;

% Neuron Parameters.
    mew_e = 20.8;       % mean baseline current
    sigma_e = 1;        % std
    mew_i = 18;
    sigma_i = 3;
    mew_c = 0;
    N_E = 1600;         % number of excitatory neurons
    N_I = 400;          % number of inhibitory neurons
    Ntot = N_E + N_I;

% Synaptic Parameters.
    Weight_0 = 1;
    J_E = 100;          % J_E = J_EI
    % TO MAKE FIGURE 3, 6, 7
    J_I = 260;          % J_I = J_IE
    % TO MAKE FIGURE 4
    % J_I = 75;         % J_I = J_IE
    N_i = 1;
    C_E = 0.3 * Ntot;   % N_I;
    C_I = 0.3 * Ntot;   % N_E;
    tau_LTP = 20;
    tau_LTD = 22;

% Make Random Synaptic Conncetions.
    epsilon_E = 0.1;    % connectivity
    epsilon_I = 0.1;    % connectivity
    
    [S_key_EI, num_synapses_EI] = make_synaptic_connections(N_I, N_E, epsilon_I);  % I → E
    [S_key_IE, num_synapses_IE] = make_synaptic_connections(N_E, N_I, epsilon_E);  % E → I
    
    W_IE = zeros(num_steps, 1);
    W_IE_std = zeros(num_steps, 1);

% Initial Conditions.
    vE0 = 14 * ones(1, N_E);
    vI0 = 14 * ones(1, N_I);
    S_EI0 = zeros(1, N_E);
    S_IE0 = zeros(1, N_I);
    X_IE0 = zeros(1, N_I);
    X_EI0 = zeros(1, N_E);
    Apost0 = zeros(1, num_synapses_IE);
    Apre0 = zeros(1, num_synapses_IE);
    W_IE0 = Weight_0 * ones(1, num_synapses_IE);   
    W_EI0 = Weight_0;
    leftover_S_EI = zeros(5 / step_size + 1, N_E);
    leftover_S_IE = zeros(5 / step_size + 1, N_I);
    ref_E = zeros(1, N_E);
    ref_I = zeros(1, N_I);
    spt_E0 = 0;
    spE_count0 = 0;
    phi0 = zeros(1, N_E);
    phif = zeros(1, N_E);
    sample_duration = 20;
    num_samples = duration / sample_duration;
    Synchrony = zeros(num_samples, 1);
    time_syn = zeros(num_samples, 1);
    num_steps_per_sample = sample_duration / step_size;
    spike_time_E = zeros(num_steps_per_sample, N_E);
    spE_count =  zeros(num_steps_per_sample, N_E);

    % TO MAKE FIGURE 7
    tau_E_m = 10;   % 8 + 4 * rand(1, N_E);
    tau_I_m = 10;   % 8 + 4 * rand(1, N_I);

% Generate General Stimulation Pattern.
    cross_100 = 1;
    comp_time = 0;
    V_stim = 1;
    T_stim = 1;
    x_neutral = 10;
    multiple = 1;
    t_pulse = T_stim * (x_neutral + multiple + 1);
    [Ue Ui] = pulsatile_input(multiple, V_stim, T_stim, x_neutral, duration, step_size);
    % sum(Ue)       % expect 0
    % sum(Ui)       % expect 0

% Save Presim State.
if cache
    disp('');
    disp('Saving presim state.');
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    filename = sprintf('./data/presim_state/m-%d-%d-%s.mat', N_E, N_I, timestamp)
    save(filename, ...
        % Run Parameters
        'duration', 'step_size', 'num_steps',
        % Neuron Parameters
        'mew_e', 'sigma_e', 'mew_i', 'sigma_i', 'mew_c', 'N_E', 'N_I', 'Ntot',
        % Synaptic Parameters
        'Weight_0', 'J_E', 'J_I', 'N_i', 'C_E', 'C_I', 'tau_LTP', 'tau_LTD',
        % Make Random Synaptic Connections
        'epsilon_E', 'epsilon_I', 'S_key_IE', 'S_key_EI', 'num_synapses_IE',
        'num_synapses_EI', 'W_IE', 'W_IE_std',
        % Initial Conditions
        'vE0', 'vI0', 'S_EI0', 'S_IE0', 'X_IE0', 'X_EI0', 'Apost0', 'Apre0',
        'W_IE0', 'W_EI0', 'leftover_S_EI', 'leftover_S_IE', 'ref_E', 'ref_I',
        'spt_E0', 'spE_count0', 'phi0', 'phif', 'sample_duration', 'num_samples',
        'Synchrony', 'time_syn', 'num_steps_per_sample', 'spike_time_E', 'spE_count',
        'tau_E_m', 'tau_I_m',
        % Generate General Stimulation Pattern
        'cross_100', 'comp_time', 'V_stim', 'T_stim', 'x_neutral', 'multiple',
        't_pulse', 'Ue', 'Ui',
        % Format version
        '-v7'
    )
end

% Run Simulation.
disp('');
disp('Running Simulation...');
for i = 1:num_samples
    % Display Progress.
    if i <= 1200 && mod(i, 100) == 0
        msg = sprintf('\tSample: %d/%d', i, num_samples);
        disp(msg);
    end
    if i > 1200 && mod(i, 10) == 0
        msg = sprintf('\tSample: %d/%d', i, num_samples);
        disp(msg);
    end

    % run parameters
    comp_time = (i - 1) * sample_duration;

    % TO MAKE FIGURES 3, 6, 7
    if mean(W_IE0) * J_I < 75
        cross_100 = 0;
    end

    % TO MAKE FIGURE 4
    % if mean(W_IE0)*J_I > 125
    %   cross_100 = 0;
    % end

    ON = 1 * (i * sample_duration >= 2000) * cross_100; % control input
    plast_on = 1 * (i * sample_duration >= 100);        % pasticity

    % indexes
    a = 1 + (i >= 2) * (i - 1) * num_steps_per_sample;  % sample start
    b = i * num_steps_per_sample;                       % sample end
    % todo: note sample_end below when renaming 'b' to 'sample_end'
    
    % if i <= 1200 && mod(i, 100) == 0
    %     disp(['Sample: ' num2str(i) '/' num2str(num_samples) '\t(' num2str(a) ':' num2str(b) ')'])
    % end
    % if i > 1200 && mod(i, 10) == 0
    %     disp(['Sample: ' num2str(i) '/' num2str(num_samples) '\t(' num2str(a) ':' num2str(b) ')'])
    % end

    % TO MAKE FIGURE 3
    Vstim = 100;
    ue = Vstim * Ue(1, a:b);
    ui = Vstim * Ui(1, a:b);
    
    % TO MAKE FIGURE 4
    % Vstim = 200;
    % ue = Vstim * Ui(1, a:b);
    % ui = Vstim * Ue(1, a:b);

    percent_V_stim = 1;
    [timem v_Em v_Im S_EIm S_IEm X_EIm X_IEm Apostm Aprem W_IEm spike_Em spike_Im ref_Em ref_Im synchronym spt_Em phif] = ode_neuron_model(plast_on,ON,vE0,vI0,S_EI0,S_IE0,X_EI0,X_IE0,Apost0,Apre0,W_IE0,W_EI0,mew_e,sigma_e,ue,ui,mew_i,sigma_i,J_E,J_I,C_E,C_I,tau_LTP,tau_LTD,step_size,sample_duration,N_E,N_I,S_key_EI,S_key_IE,leftover_S_EI,leftover_S_IE,ref_E,ref_I,tau_E_m,tau_I_m, percent_V_stim,comp_time,spt_E0,phif);

    % recorded variables
    time(a:b, :) = timem(1:num_steps_per_sample, :) + (i - 1) * sample_duration;
    W_IE(a:b, 1) = mean(W_IEm(1:num_steps_per_sample, :), 2);
    % spike_E(a:b, :) = spike_Em(1:num_steps_per_sample, :);
    % spike_I(a:b, :) = spike_Im(1:num_steps_per_sample, :);
    Synchrony(i) = synchronym;
    time_syn(i) = sample_duration * (i);
    spike_time_E(a:b, :) = spt_Em(1:num_steps_per_sample, :);

    % generate intial condition for next run
    sample_end = num_steps_per_sample; % todo: note when renaming 'b' to 'sample_end'
    vE0 = v_Em(sample_end, :);
    vI0 = v_Im(sample_end, :);
    S_EI0 = S_EIm(sample_end, :);
    S_IE0 = S_IEm(sample_end, :);
    X_EI0 = X_EIm(sample_end, :);
    X_IE0 = X_IEm(sample_end, :);
    Apost0 = Apostm(sample_end, :);
    Apre0 = Aprem(sample_end, :);
    W_IE0 = W_IEm(sample_end, :);
    W_EI0 = Weight_0;
    left_sample_end = sample_end - 5 / step_size;
    leftover_S_EI = S_EIm(left_sample_end:sample_end, :);
    leftover_S_IE = S_IEm(left_sample_end:sample_end, :);
    spt_E0 = spt_Em(sample_end, :);
end
disp('Simulation Complete.')

% Save Postsim State.
if cache
    disp('');
    disp('Saving postsim state.');
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    filename = sprintf('./data/postsim_state/m-%d-%d-%s.mat', N_E, N_I, timestamp)
    t = [0.1:step_size:duration];  % to check sanity
    save(filename, ...
        % Run Parameters
        'duration', 'step_size', 'num_steps',
        % Neuron Parameters
        'mew_e', 'sigma_e', 'mew_i', 'sigma_i', 'mew_c', 'N_E', 'N_I', 'Ntot',
        % Synaptic Parameters
        'Weight_0', 'J_E', 'J_I', 'N_i', 'C_E', 'C_I', 'tau_LTP', 'tau_LTD',
        % Make Random Synaptic Connections
        'epsilon_E', 'epsilon_I', 'S_key_IE', 'S_key_EI', 'num_synapses_IE',
        'num_synapses_EI', 'W_IE', 'W_IE_std',
        % Initial Conditions
        'vE0', 'vI0', 'S_EI0', 'S_IE0', 'X_IE0', 'X_EI0', 'Apost0', 'Apre0',
        'W_IE0', 'W_EI0', 'leftover_S_EI', 'leftover_S_IE', 'ref_E', 'ref_I',
        'spt_E0', 'spE_count0', 'phi0', 'phif', 'sample_duration', 'num_samples',
        'Synchrony', 'time_syn', 'num_steps_per_sample', 'spike_time_E', 'spE_count',
        'tau_E_m', 'tau_I_m',
        % Generate General Stimulation Pattern
        'cross_100', 'comp_time', 'V_stim', 'T_stim', 'x_neutral', 'multiple',
        't_pulse', 'Ue', 'Ui',
        % Sanity check and format version
        't',
        '-v7'
    )
end

% run time
minute = toc / 60


% --- TODO ZONE ---

% plots

% figure(1)
% plot(time / 1000, J_I * W_IE, 'k', 'LineWidth', 1.2)
% xlabel('Time (sec)')
% ylabel('Average Synpatic Weight')
% xlim([0 duration / 1000])
% ylim([0 300])

% Calculate Kuramoto Order Parameter.
% t = [0.1:step_size:duration];
% [RE] = kuramoto_syn(spike_time_E, t, step_size, duration, N_E);
% order_parameter = RE;
% save('order_parameter.mat', 'order_parameter', '-v7')

% figure(2)
% plot(t/1000,RE)
% xlim([0.2 0.8])
% ylim([0 1.1])

% calculate the rates

% dt = 100;
% [rate_E, rate_I, t_rate] = rate_calc(spike_E, spike_I, step_size, dt, N_E, N_I);

% figure(4)
% plot(t_rate/1000,1000*rate_E,'k',t_rate/1000,1000*rate_I,'r','LineWidth',1.2)
% ylabel('Average Firing Rate (sp/sec)')
% xlabel('Time (sec)')
% legend('Excitatory Neurons','Inhibitory Neurons')

% figure(3)
% plot(t,S_EI,'k',t,S_IE,'r')
% legend('I-to-E','E-to-I')
% xlabel('Time (sec)')
% ylabel('Average Synaptic Input')

% figure(4)
% plot(t/1000, spike_E, 'k.',t/1000, spike_I, 'r.')
% xlim([400/1000 500/1000])
% ylim([0.9 2000.1])
% ylabel('Neuron Index')
% xlabel('Time (sec)')

% tb = [200:0.1:600];
% tm = [5000:0.1:5400]; 
% te = [38000:0.1:38400];

% figure(3)
% subplot(3,1,1)
% a = 200/step_size;
% b = 600/step_size;

% plot(tb/1000, spike_E(a:b,:), 'k.',tb/1000, spike_I(a:b,:), 'r.')
% xlim([200/1000 600/1000])
% ylim([0.9 2000.1])
% ylabel('Neuron Index')
% xlabel('Time (sec)')

% c = 5000/step_size;
% d = 5400/step_size;
% subplot(3,1,2)
% plot(tm/1000, spike_E(c:d,:), 'k.',tm/1000, spike_I(c:d,:), 'r.')
% xlim([5000/1000 5400/1000])
% ylim([0.9 2000.1])
% ylabel('Neuron Index')
% xlabel('Time (sec)')

% e = 38000/step_size;
% f = 38400/step_size;
% subplot(3,1,3)
% plot(te/1000, spike_E(e:f,:), 'k.',te/1000, spike_I(e:f,:), 'r.')
% xlim([38000/1000 38400/1000])
% ylim([0.9 2000.1])
% ylabel('Neuron Index')
% xlabel('Time (sec)')
