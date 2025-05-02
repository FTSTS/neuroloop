"""
Code to plot data from the DBS simulations.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_pulsatile_input(Ue, Ui, duration, step_size):
    """
    Plot the Ue and Ui inputs.
    """

    num_steps = int(duration / step_size)
    t = [i * step_size for i in range(num_steps)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].plot(t, Ui, label='Ui', color='red')
    axes[0].set_title('FTSTS Waveform for inhibitory (I) neurons')
    axes[0].set_xlabel('time (s)')
    axes[0].set_ylabel('Vstim(t) (V)')
    axes[0].set_xlim([0, duration / 1000])
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()

    axes[1].plot(t, Ue, label='Ue', color='blue')
    axes[1].set_title('FTSTS Waveform for excitatory (E) neurons')
    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel('Vstim(t) (V)')
    axes[1].set_xlim([0, duration / 1000])
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_kop(t, re):
    plt.figure(figsize=(10, 5))
    plt.plot(t / 1000, re, label='Kuramoto Order Parameter', color='blue')
    plt.title('Time Series Kuramoto Order Parameter', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('KOP', fontsize=12)
    # plt.xlim([0.2, 0.8])
    plt.ylim([0, 1.1])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


def plot_avg_synaptic_weight(time, J_I, W_IE, duration):
    plt.figure(figsize=(10, 5))
    plt.plot(time / 1000, J_I * W_IE,
             label='Average Synaptic Weight', color='blue')
    plt.title('Time Series Average Synaptic Weight', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Average Synaptic Weight (IE)', fontsize=12)
    plt.xlim([0, duration / 1000])
    plt.ylim([0, 300])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


def plot_synchrony(syn):
    t = np.arange(len(syn))
    plt.figure(figsize=(10, 4))
    plt.plot(t, syn)
    plt.xlabel("Time")
    plt.ylabel("Synchrony")
    plt.tight_layout()
    plt.show()

# def plot_spike_patterns(spike_e, spike_i, step_size):
#     spike_e_T = spike_e.T
#     neuron_indices, time_indices = np.where(spike_e_T != 0)
#     spike_times = spike_e_T[neuron_indices, time_indices]
#     plt.figure(figsize=(10, 6))
#     plt.scatter(spike_times[1_000_000:1_001_000], neuron_indices[1_000_000:1_001_000], s=2)

#     # plt.figure(figsize=(12, 8))
#     # tb = np.arange(200, 600, step_size)
#     # tm = np.arange(5000, 5400, step_size)
#     # te = np.arange(38000, 38400, step_size)
#     # plt.figure(3)
#     # # E ?
#     # print("E")
#     # plt.subplot(3,1,1)
#     # a = int(200/step_size)
#     # b = int(600/step_size)
#     # plt.plot(tb/1000, spike_e[a:b, :], 'k.', tb/1000, spike_i[a:b, :], 'r.')
#     # plt.xlim([200/1000, 600/1000])
#     # plt.ylim([0.9, 2000.1])
#     # plt.ylabel('Neuron Index')
#     # plt.xlabel('Time (sec)')

#     # # F ?
#     # print("F")
#     # c = int(5000/step_size)
#     # d = int(5400/step_size)
#     # plt.subplot(3,1,2)
#     # plt.plot(tm/1000, spike_e[c:d, :], 'k.', tm/1000, spike_i[c:d, :], 'r.')
#     # plt.xlim([5000/1000, 5400/1000])
#     # plt.ylim([0.9, 2000.1])
#     # plt.ylabel('Neuron Index')
#     # plt.xlabel('Time (sec)')

#     # G ?
#     # print("G")
#     # e = int(38000/step_size)
#     # f = int(38400/step_size)
#     # plt.subplot(3,1,3)
#     # plt.plot(te/1000, spike_e[e:f, :], 'k.', te/1000, spike_i[e:f, :], 'r.')
#     # plt.xlim([38000/1000, 38400/1000])
#     # plt.ylim([0.9, 2000.1])
#     # plt.ylabel('Neuron Index')
#     # plt.xlabel('Time (sec)')
#     plt.show()


def plot_avg_synaptic_input(S_EI, S_IE, duration, step_size):
    num_steps = int(duration / step_size)
    t = [i * step_size for i in range(num_steps)]

    plt.figure(figsize=(10, 5))
    plt.plot(t, S_EI, 'k', t, S_IE, 'r')
    plt.title('Time Series Average Synaptic Input', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Average Synaptic Input', fontsize=12)
    plt.xlim([0, duration / 1000])
    plt.ylim([-0.5, 0.5])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(['I-to-E', 'E-to-I'])
    plt.show()

    # plt.figure(3)
    # plt.plot(t, S_EI, 'k', t, S_IE, 'r')
    # plt.legend(['I-to-E', 'E-to-I'])
    # plt.xlabel('Time (sec)')
    # plt.ylabel('Average Synaptic Input')


# def todo():

#     # t = np.arange(0.1, duration+step_size, step_size)
#     t = np.linspace(0.1,  # precision error with np.arange
#                     duration,
#                     int(round((duration - 0.1) / step_size)) + 1)

#     # calculate the rates

#     # dt = 100
#     # [rate_E, rate_I, t_rate] = rate_calc(spike_E, spike_I, step, dt, N_E, N_I);
#     #
#     # plt.figure(4)
#     # plt.plot(t_rate/1000, 1000*rate_E, 'k', t_rate/1000, 1000*rate_I, 'r', linewidth=1.2)
#     # plt.ylabel('Average Firing Rate (sp/sec)')
#     # plt.xlabel('Time (sec)')
#     # plt.legend(['Excitatory Neurons', 'Inhibitory Neurons'])

#     # plt.figure(4)
#     # plt.plot(t/1000, spike_E, 'k.', t/1000, spike_I, 'r.')
#     # plt.xlim([400/1000, 500/1000])
#     # plt.ylim([0.9, 2000.1])
#     # plt.ylabel('Neuron Index')
#     # plt.xlabel('Time (sec)')
