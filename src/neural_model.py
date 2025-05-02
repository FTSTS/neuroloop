"""
Simulation of Deep Brain Stimulation (DBS).

Uses the Forced Temporal Spike-Time Stimulation (FTSTS) DBS strategy.
"""
# pylint: disable=invalid-name

import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
from scipy.io import savemat
from pulsatile_input import pulsatile_input
from make_synaptic_connections import make_synaptic_connections


INHIBITION_THRESHOLD = 75  # (mV) threshold for inhibition
STIMULATION_ONSET_TIME_RATIO = 0.08  # begin after 8% of the simulation time
PLASTICITY_ONSET_TIME_RATIO = 0.004  # begin after 0.4% of the simulation time
SAMPLE_RATIO = 0.0008  # sample window is 0.08% of the simulation time

V_REST = 0  # (mV) resting potential
V_THRESHOLD = 20  # (mV) threshold potential for action potential
V_RESET = 14  # (mV) reset potential after action potential
REFRACTORY = 2  # (ms) refractory period after action potential
MAX_WEIGHT_IE = 10  # (mV) maximum synaptic weight for E->I connections
MAX_WEIGHT_EI = 290  # (mV) maximum synaptic weight for I->E connections


class NeuralModel:
    """
    Neural Network Model for Simulating Deep Brain Stimulation (DBS).

    Excitatory-inhibitory network model with synaptic plasticity.
    """

    def __init__(
            self,
            num_e=1600,  # number of excitatory neurons
            num_i=400,  # number of inhibitory neuronsƒ
            duration=25_000,  # (ms) duration of simulation
            # todo: kwargs for other parameters
            seed=None,  # random seed for reproducibility
    ):
        if seed is not None:
            np.random.seed(seed)

        # Run Parameters.
        self.duration = duration
        self.step_size = 0.1  # (ms)
        self.num_steps = int(duration / self.step_size)
        self.sample_duration = 20  # (ms)
        # self.sample_duration = int(duration * SAMPLE_RATIO)
        assert self.sample_duration > 0
        self.num_samples = int(duration / self.sample_duration)
        self.num_steps_per_sample = int(self.sample_duration / self.step_size)

        # Neuron Parameters.
        self.mew_e = 20.8
        self.sigma_e = 1
        self.mew_i = 18
        self.sigma_i = 3
        self.mew_c = 0
        self.num_e = num_e
        self.num_i = num_i
        self.num_neurons = self.num_e + self.num_i

        # Synaptic Parameters.
        self.weight_0 = 1
        self.j_e = 100  # synaptic strength (J_E = J_EI)
        self.j_i = 260  # synaptic strength (J_I = J_IE)
        self.n_i = 1
        self.c_e = 0.3 * self.num_neurons  # N_I;
        self.c_i = 0.3 * self.num_neurons  # N_E;
        self.tau_ltp = 20  # long-term potentiation time constant
        self.tau_ltd = 22  # long-term depression time constant

        # Make Random Synaptic Connections.
        self.epsilon_e = 0.1
        self.epsilon_i = 0.1
        self.s_key_ei, self.num_synapses_ei = make_synaptic_connections(  # I -> E
            num_pre=self.num_i,
            num_post=self.num_e,
            epsilon=self.epsilon_i,
        )
        self.s_key_ie, self.num_synapses_ie = make_synaptic_connections(  # E -> I
            num_pre=self.num_e,
            num_post=self.num_i,
            epsilon=self.epsilon_e,
        )
        self.w_ie = np.zeros((self.num_steps, 1))
        self.w_ie_std = np.zeros((int(self.num_steps), 1))

        # Initial Conditions (todo: state?).
        self.v_e0 = 14 * np.ones((1, self.num_e))
        self.v_i0 = 14 * np.ones((1, self.num_i))
        self.s_ei0 = np.zeros((1, self.num_e))
        self.s_ie0 = np.zeros((1, self.num_i))
        self.x_ei0 = np.zeros((1, self.num_e))
        self.x_ie0 = np.zeros((1, self.num_i))
        self.apost0 = np.zeros((1, int(self.num_synapses_ie)))
        self.apre0 = np.zeros((1, int(self.num_synapses_ie)))
        self.w_ie0 = self.weight_0 * np.ones((1, int(self.num_synapses_ie)))
        self.w_ei0 = self.weight_0
        self.leftover_s_ei = np.zeros((
            int(5 / self.step_size) + 1,
            self.num_e
        ))
        self.leftover_s_ie = np.zeros((
            int(5 / self.step_size) + 1, self.num_i
        ))
        self.ref_e = np.zeros((1, self.num_e))
        self.ref_i = np.zeros((1, self.num_i))
        self.spt_e0 = 0
        self.sp_e_count0 = 0
        self.phi0 = np.zeros((1, self.num_e))
        self.phif = np.zeros((1, self.num_e))
        self.synchrony = np.zeros((int(self.num_samples), 1))
        self.time_syn = np.zeros((int(self.num_samples), 1))
        self.spike_time_e = np.zeros((self.num_steps, self.num_e))
        self.sp_e_count = np.zeros((
            int(self.num_steps_per_sample),
            self.num_e
        ))
        self.tau_e_m = np.full((1, self.num_e), 10)
        self.tau_i_m = np.full((1, self.num_i), 10)

        # Generate General Stimulation Pattern.
        self.cross_100 = 1
        self.comp_time = 0
        self.v_stim = 1  # (mV) stimulation amplitude (todo: rename: Ustim)
        self.t_stim = 1  # (ms)
        self.x_neutral = 10  # (ms) (todo: rename: t_neutral)
        self.multiple = 1
        self.t_pulse = self.t_stim * (self.x_neutral + self.multiple + 1)
        self.u_e, self.u_i = pulsatile_input(  # todo: rename Ve, Vi?
            multi=self.multiple,
            v_stim=self.v_stim,
            t_stim=self.t_stim,
            x=self.x_neutral,
            duration=self.duration,
            step_size=self.step_size
        )
        self.u_e = self.u_e.reshape(1, -1)  # (N,) -> (1, N)
        self.u_i = self.u_i.reshape(1, -1)  # (N,) -> (1, N)
        self.time_array = np.zeros((self.num_steps, 1))

    def _dVdt(self, v, z, mu, stim, sigma, tau_m, x):
        """todo"""
        # todo: link to paper/equation
        # todo: include equation in docstring

        # τm dV(t)/dt = -V(t) + Z(t) μ + (σ * sqrt(τm) * X(t)) + Vstim(t)

        return (
            V_REST - v + z + mu + sigma * np.sqrt(tau_m) * x + stim
        ) / tau_m

    def step(self, ue, ui, plast_on, stim_on, percent_V_stim=1):
        """
        ODE Neuron Model.

        Function to simulate a network of excitatory and inhibitory neurons
        with synaptic plasticity.
        """

        num_steps = self.num_steps_per_sample

        # Neuron Parameters.
        tau_d = 1
        tau_r = 1
        x_e = np.random.randn(num_steps + 1, self.num_e)  # white noise
        x_i = np.random.randn(num_steps + 1, self.num_i)  # white noise

        # Synaptic Parameters.
        WEI = self.w_ei0
        syn_delay = 5  # (ms)

        # Plasticity Parameters.
        dApre_0 = 0.005*1
        dApost_0 = dApre_0*1
        Apre_i = 0
        Apost_i = 0
        a_LTD = -1 * plast_on * 1.1
        a_LTP = 1 * plast_on * 1
        eta = 0.25

        # Initialize State Vectors.
        # todo: can reduce memory with curr, next instead of full arrays
        # todo: maybe not... np.mean(v_e, axis=1)
        v_e = np.zeros((num_steps + 1, self.num_e))
        v_i = np.zeros((num_steps + 1, self.num_i))
        s_ei = np.zeros((num_steps + 1, self.num_e))
        s_ie = np.zeros((num_steps + 1, self.num_i))
        x_ei = np.zeros((num_steps + 1, self.num_e))
        x_ie = np.zeros((num_steps + 1, self.num_i))
        apost = np.zeros((num_steps + 1, self.num_synapses_ie))
        apre = np.zeros((num_steps + 1, self.num_synapses_ie))
        w_ie = np.zeros((num_steps + 1, self.num_synapses_ie))
        step_times = np.zeros((num_steps + 1, 1))
        spike_e = np.zeros((num_steps + 1, self.num_e))
        spike_i = np.zeros((num_steps + 1, self.num_i))
        # spt_E = np.zeros((1, self.num_e))
        spt_e = np.zeros((num_steps + 1, self.num_e))

        v_e[0, :] = self.v_e0
        v_i[0, :] = self.v_i0
        s_ei[0, :] = self.s_ei0
        s_ie[0, :] = self.s_ie0
        x_ei[0, :] = self.x_ei0
        x_ie[0, :] = self.x_ie0
        apost[0, :] = self.apost0
        apre[0, :] = self.apre0
        w_ie[0, :] = self.w_ie0
        spt_e[0, :] = self.spt_e0

        # Stimulation Parameters.
        # number of E neurons stimulating
        stim_percent_E = np.zeros(self.num_e)
        a = 0
        b = int(np.floor(percent_V_stim * self.num_e))
        stim_percent_E[a:b] = 1  # TO MAKE FIGURES 3,4,7
        # stim_percent_E[a:b] = 1 + 0.1*np.random.randn(b)  # TO MAKE FIGURE 6

        # number of I neurons stimulating
        stim_percent_I = np.zeros(self.num_i)
        a = 0
        b = int(np.floor(percent_V_stim * self.num_i))
        stim_percent_I[a:b] = 1  # FIGURES 3,4,7
        # stim_percent_I[a:b] = 1 + 0.1*np.random.randn(b)  # FIGURE 6

        spike_I_time = np.zeros((num_steps + 1, self.num_i))
        delay_index = int(syn_delay / self.step_size) - 1

        for t in range(num_steps):
            step_times[t + 1, 0] = step_times[t, 0] + self.step_size

            # Excitatory-Inhibitory Network Model Updates.
            # Voltage (membrane) potentials.
            # (eq1&2): τm dV(t)/dt = -V(t) + Z(t) μ + (σ * sqrt(τm) * X(t)) + Vstim(t)
            # FIGURES 3,4,7
            dv_Edt = self._dVdt(  # equation (1) in paper
                v=v_e[t, :],
                z=self.j_e / self.c_e * (  # equation (3) in paper
                    s_ei[t - delay_index, :] if t > delay_index
                    else self.leftover_s_ei[t, :]
                ),
                mu=self.mew_e,
                stim=stim_on * ue[0, t],
                sigma=self.sigma_e,
                tau_m=self.tau_e_m[0, :],
                x=x_e[t, :],
            )
            dv_Idt = self._dVdt(  # equation (2) in paper
                v=v_i[t, :],
                z=self.j_i / self.c_i * (  # equation (3) in paper
                    s_ie[t - delay_index, :] if t > delay_index
                    else self.leftover_s_ie[t, :]
                ),
                mu=self.mew_i,
                stim=stim_on * ui[0, t],
                sigma=self.sigma_i,
                tau_m=self.tau_i_m[0, :],
                x=x_i[t, :],
            )
            # FIGURE 6
            # dv_Edt[i, :] = (vrest - v_E[i, :] + J_E/C_E * S_EI[i-delay_index, :] + mew_e + ON1 * ue[0, i] + ON1 * 10 * np.random.randn(1, N_E) + sigma_e * (tau_E_m[0, :]**0.5) * whitenoise_E[i, :]) / tau_E_m[0, :]
            # dv_Edt[i, :] = (vrest - v_E[i, :] + J_E/C_E * leftover_S_EI[i, :] + mew_e + ON1 * ue[0, i] + ON1 * 10 * np.random.randn(1, N_E) + sigma_e * (tau_E_m[0, :]**0.5) * whitenoise_E[i, :]) / tau_E_m[0, :]
            # dv_Idt[i, :] = (vrest - v_I[i, :] + J_I/C_I * S_IE[i-delay_index, :] + mew_i + ON1 * ui[0, i] + ON1 * 10 * np.random.randn(1, N_I) + sigma_i * (tau_I_m[0, :]**0.5) * whitenoise_I[i, :]) / tau_I_m[0, :]
            # dv_Idt[i, :] = (vrest - v_I[i, :] + J_I/C_I * leftover_S_IE[i, :] + mew_i + ON1 * ui[0, i] + ON1 * 10 * np.random.randn(1, N_I) + sigma_i * (tau_I_m[0, :]**0.5) * whitenoise_I[i, :]) / tau_I_m[0, :]
            v_e[t + 1, :] = v_e[t, :] + self.step_size * dv_Edt
            v_i[t + 1, :] = v_i[t, :] + self.step_size * dv_Idt

            # Conductance Updates.
            # (eq5): τd dS(t)/dt = -S(t) + X(t)
            dS_EIdt = (  # equation (5) in paper
                -s_ei[t, :] + x_ei[t, :]
            ) / tau_d
            dS_IEdt = (  # equation (5) in paper
                -s_ie[t, :] + x_ie[t, :]
            ) / tau_d
            s_ei[t + 1, :] = s_ei[t, :] + self.step_size * dS_EIdt
            s_ie[t + 1, :] = s_ie[t, :] + self.step_size * dS_IEdt

            # (eq6): τr dX(t)/dt = -X(t) + W(t) * δ(t - tpre + tdelay)
            dX_EIdt = (  # equation (6) in paper
                -x_ei[t, :]  # + W(t) * δ(t - tpre + tdelay) in spike check
            ) / tau_r
            dX_IEdt = (  # equation (6) in paper
                -x_ie[t, :]  # + W(t) * δ(t - tpre + tdelay) in spike check
            ) / tau_r
            x_ei[t + 1, :] = x_ei[t, :] + self.step_size * dX_EIdt
            x_ie[t + 1, :] = x_ie[t, :] + self.step_size * dX_IEdt

            # Spike-Timing Dependence Plasticity (STDP) Updates.
            # (eq7): W(t + Δt) = W(t) + ΔW(t)
            # (eq8): ΔW(t) = η * aLTP * Apost(t) if tpre - tpost < 0
            # (eq9): ΔW(t) = η * aLTD * Apre(t) if tpre - tpost > 0
            w_ie[t + 1, :] = (  # equation (7) in paper
                w_ie[t, :]  # + ΔW(t) in spike check
            )

            # (eq10): τLTP dApost/dt = -Apost + A0 * δ(t - tpost)
            dApostdt = (
                -apost[t, :]  # + A0 * δ(t - tpost) in spike check
            ) / self.tau_ltd
            apost[t + 1, :] = apost[t, :] + self.step_size * dApostdt

            # (eq11): τLTD dApre/dt = -Apre + A0 * δ(t - tpre)
            dApredt = (
                -apre[t, :]  # + A0 * δ(t - tpre) in spike check
            ) / self.tau_ltp
            apre[t + 1, :] = apre[t, :] + self.step_size * dApredt

            # Refractory Updates.
            self.ref_e[0, :] -= self.step_size
            self.ref_i[0, :] -= self.step_size

            # Calculate Kuramoto Order (but not really).
            spt_e[t + 1, :] = spt_e[t, :]
            # phif[i+1+int(comp_time/step), :] = 2*np.pi * (time[i+1, 0] + comp_time - spt_E[i, :])

            # Check for Spikes.
            for k in range(self.num_e):  # excitatory
                if v_e[t, k] < V_THRESHOLD <= v_e[t + 1, k]:
                    v_e[t + 1, k] = V_RESET
                    self.ref_e[0, k] = REFRACTORY
                    spike_e[t + 1, k] = k + 1
                    spt_e[t + 1, k] = step_times[t + 1, 0] + self.comp_time
                    for j in range(self.num_i):  # E to I
                        if self.s_key_ie[k, j] != 0:
                            syn_idx = int(self.s_key_ie[k, j]) - 1
                            x_ie[t + 1, j] = (  # + W(t) * δ(t - tpre + tdelay)
                                x_ie[t, j] + w_ie[t, syn_idx]
                            )
                            # plasticity update - "on_pre"
                            apre[t + 1, syn_idx] = (  # + A0 * δ(t - tpre)
                                apre[t, syn_idx] + dApre_0
                            )
                            w_ie[t + 1, syn_idx] = (  # equation (9) in paper
                                w_ie[t, syn_idx]
                                + eta * a_LTD * apost[t, syn_idx]
                            )
                            # max synaptic weight check
                            if (self.j_i * w_ie[t + 1, syn_idx]) < MAX_WEIGHT_IE:
                                w_ie[t + 1, syn_idx] = MAX_WEIGHT_IE / self.j_i
                elif self.ref_e[0, k] >= 0:  # in refractory period
                    v_e[t + 1, k] = V_RESET
                elif v_e[t + 1, k] < V_REST:
                    v_e[t + 1, k] = V_REST

            for k in range(self.num_i):  # inhibitory
                if v_i[t, k] < V_THRESHOLD <= v_i[t + 1, k]:
                    v_i[t + 1, k] = V_RESET
                    self.ref_i[0, k] = REFRACTORY
                    spike_i[t + 1, k] = k + 1 + self.num_e
                    spike_I_time[t + 1, k] = step_times[t + 1, 0]

                    for j in range(self.num_e):  # I to E
                        if self.s_key_ei[k, j] != 0:
                            syn_idx = int(self.s_key_ei[k, j]) - 1
                            x_ei[t + 1, j] = x_ei[t, j] - WEI
                        # plasticity update - "on_post"
                        if self.s_key_ie[j, k] != 0:
                            syn_idx = int(self.s_key_ie[j, k]) - 1
                            apost[t + 1, syn_idx] = apost[t, syn_idx] + dApost_0
                            w_ie[t + 1, syn_idx] = (  # equation (8) in paper
                                w_ie[t, syn_idx]
                                + eta * a_LTP * apre[t, syn_idx]
                            )
                            # max synaptic weight check
                            if (self.j_i * w_ie[t + 1, syn_idx]) > MAX_WEIGHT_EI:
                                w_ie[t + 1, syn_idx] = MAX_WEIGHT_EI / self.j_i
                elif self.ref_i[0, k] >= 0:
                    # check if in refractory period
                    v_i[t + 1, k] = V_RESET
                elif v_i[t + 1, k] < V_REST:
                    v_i[t + 1, k] = V_REST

        # Calculate Synchrony.
        # todo: what synchrony measurement is this?
        # todo: different from Kuramoto?
        N = self.num_e  # + self.num_i;
        Vcomb = np.zeros((num_steps + 1, N))
        Vcomb[:, 0:self.num_e] = v_e
        V1 = np.mean(Vcomb, axis=1)

        # variance of average voltage over whole run
        # sigma_v^2 = <(V(t)^2>t -[<V(t)>]^2
        sigma_squ_v = np.mean(V1 ** 2) - (np.mean(V1)) ** 2

        # variance of voltage at each time step
        sigma_vi = np.zeros(N)
        sum_sig = 0
        for j in range(N):
            sigma_vi[j] = (
                np.mean(Vcomb[:, j] ** 2)
                - (np.mean(Vcomb[:, j])) ** 2
            )
            sum_sig = sum_sig + sigma_vi[j]

        syn_squ = sigma_squ_v / (sum_sig / N)
        synchrony = float(np.sqrt(syn_squ))

        assert step_times.shape == (num_steps + 1, 1)
        assert v_e.shape == (num_steps + 1, self.num_e)
        assert v_i.shape == (num_steps + 1, self.num_i)
        assert s_ei.shape == (num_steps + 1, self.num_e)
        assert s_ie.shape == (num_steps + 1, self.num_i)
        assert x_ei.shape == (num_steps + 1, self.num_e)
        assert x_ie.shape == (num_steps + 1, self.num_i)
        assert apost.shape == (num_steps + 1, self.num_synapses_ie)
        assert apre.shape == (num_steps + 1, self.num_synapses_ie)
        assert w_ie.shape == (num_steps + 1, self.num_synapses_ie)
        assert spike_e.shape == (num_steps + 1, self.num_e)
        assert spike_i.shape == (num_steps + 1, self.num_i)
        assert spt_e.shape == (num_steps + 1, self.num_e)
        assert self.ref_e.shape == (1, self.num_e)
        assert self.ref_i.shape == (1, self.num_i)
        assert self.phif.shape == (1, self.num_e)

        return (
            step_times,
            v_e,
            v_i,
            s_ei,
            s_ie,
            x_ei,
            x_ie,
            apost,
            apre,
            w_ie,
            spike_e,
            spike_i,
            self.ref_e,
            self.ref_i,
            synchrony,
            spt_e,
            self.phif,
        )

    def save_state(self, filename: str) -> None:
        """
        Save the current state of the simulation to a .mat file.
        """

        data = {
            # Run Parameters
            'duration': self.duration,
            'step_size': self.step_size,
            'num_steps': self.num_steps,
            'sample_duration': self.sample_duration,
            'num_samples': self.num_samples,
            'num_steps_per_sample': self.num_steps_per_sample,

            # Neuron Parameters
            'mew_e': self.mew_e,
            'sigma_e': self.sigma_e,
            'mew_i': self.mew_i,
            'sigma_i': self.sigma_i,
            'mew_c': self.mew_c,
            'num_e': self.num_e,
            'num_i': self.num_i,
            'num_neurons': self.num_neurons,

            # Synaptic Parameters
            'weight_0': self.weight_0,
            'j_e': self.j_e,
            'j_i': self.j_i,
            'n_i': self.n_i,
            'c_e': self.c_e,
            'c_i': self.c_i,
            'tau_ltp': self.tau_ltp,
            'tau_ltd': self.tau_ltd,

            # Make Random Synaptic Connections
            'epsilon_e': self.epsilon_e,
            'epsilon_i': self.epsilon_i,
            's_key_ie': self.s_key_ie,
            's_key_ei': self.s_key_ei,
            'num_synapses_ie': self.num_synapses_ie,
            'num_synapses_ei': self.num_synapses_ei,
            'w_ie': self.w_ie,
            'w_ie_std': self.w_ie_std,

            # Initial Conditions
            'v_e0': self.v_e0,
            'v_i0': self.v_i0,
            's_ei0': self.s_ei0,
            's_ie0': self.s_ie0,
            'x_ie0': self.x_ie0,
            'x_ei0': self.x_ei0,
            'apost0': self.apost0,
            'apre0': self.apre0,
            'w_ie0': self.w_ie0,
            'w_ei0': self.w_ei0,
            'leftover_s_ei': self.leftover_s_ei,
            'leftover_s_ie': self.leftover_s_ie,
            'ref_e': self.ref_e,
            'ref_i': self.ref_i,
            'spt_e0': self.spt_e0,
            'spe_count0': self.sp_e_count0,
            'phi0': self.phi0,
            'phif': self.phif,
            'synchrony': self.synchrony,
            'time_syn': self.time_syn,
            'spike_time_e': self.spike_time_e,
            'sp_e_count': self.sp_e_count,
            'tau_e_m': self.tau_e_m,
            'tau_i_m': self.tau_i_m,

            # Generate General Stimulation Pattern
            'cross_100': self.cross_100,
            'comp_time': self.comp_time,
            'v_stim': self.v_stim,
            't_stim': self.t_stim,
            'x_neutral': self.x_neutral,
            'multiple': self.multiple,
            't_pulse': self.t_pulse,
            'u_e': self.u_e,
            'u_i': self.u_i,

            # Other
            'time_array': self.time_array,
        }
        savemat(filename, data)
        print(f"State saved to {filename}")

    def run_simulation(self, cache: bool = False):
        """
        Run a round of open-loop FTSTS simulation.
        """

        if cache:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = f"./data/presim_state/py-{self.num_e}-{self.num_i}-{timestamp}.mat"
            print(f"Saving presim state to {path}")
            self.save_state(path)

        stim_onset_time = int(self.duration * STIMULATION_ONSET_TIME_RATIO)
        plast_onset_time = int(self.duration * PLASTICITY_ONSET_TIME_RATIO)

        tic = time.time()

        for i in tqdm(range(1, self.num_samples + 1), "Simulating", unit="sample"):
            self.comp_time = (i - 1) * self.sample_duration

            # Calculate Average Effective Inhibition.
            if np.mean(self.w_ie0) * self.j_i < INHIBITION_THRESHOLD:
                self.cross_100 = 0

            # Toggle Stimulation and Plasticity.
            stim_on = ((i * self.sample_duration) >=
                       stim_onset_time) * self.cross_100
            plast_on = (i * self.sample_duration) >= plast_onset_time
            assert stim_on in (0, 1)
            assert plast_on in (0, 1)

            # Define the Sample Window.
            sample_start = (i >= 2) * (i - 1) * self.num_steps_per_sample
            sample_end = i * self.num_steps_per_sample
            assert sample_end - sample_start == self.num_steps_per_sample

            # Define the Stimulation Input.
            vstim = 100  # mV (todo: rename: Ustim?)
            ue = vstim * self.u_e[:, sample_start:sample_end]
            ui = vstim * self.u_i[:, sample_start:sample_end]
            assert ue.shape[1] == self.num_steps_per_sample
            assert ui.shape[1] == self.num_steps_per_sample

            # Run the ODE Neuron Model.
            (
                step_times,
                v_e,
                v_i,
                s_ei,
                s_ie,
                x_ei,
                x_ie,
                apost,
                apre,
                w_ie,
                spike_e,
                spike_i,
                ref_e,
                ref_i,
                synchrony,
                spt_e,
                phif
            ) = self.step(
                ue=ue,
                ui=ui,
                plast_on=plast_on,
                stim_on=stim_on,
                percent_V_stim=1
            )

            # Recorded Variables.
            self.time_array[sample_start:sample_end, 0] = (
                step_times[0:-1, 0] + (i - 1) * self.sample_duration
            )
            self.w_ie[sample_start:sample_end, 0] = np.mean(
                w_ie[0:-1, :],
                axis=1
            )
            # self.spike_e[sample_start:sample_end, :] = spike_e_m[0:num_steps_per_sample, :]
            # self.spike_i[sample_start:sample_end, :] = spike_i_m[0:num_steps_per_sample, :]
            self.synchrony[i - 1, 0] = synchrony
            self.time_syn[i - 1, 0] = self.sample_duration * (i)
            self.spike_time_e[sample_start:sample_end, :] = spt_e[0:-1, :]

            # Update Initial Conditions (for next run).
            self.v_e0 = v_e[-1, :].reshape(1, -1)
            self.v_i0 = v_i[-1, :].reshape(1, -1)
            self.s_ei0 = s_ei[-1, :].reshape(1, -1)
            self.s_ie0 = s_ie[-1, :].reshape(1, -1)
            self.x_ei0 = x_ei[-1, :].reshape(1, -1)
            self.x_ie0 = x_ie[-1, :].reshape(1, -1)
            self.apost0 = apost[-1, :].reshape(1, -1)
            self.apre0 = apre[-1, :].reshape(1, -1)
            self.w_ie0 = w_ie[-1, :].reshape(1, -1)
            self.w_ei0 = self.weight_0
            offset = self.num_steps_per_sample - int(5 / self.step_size)
            self.leftover_s_ei = s_ei[offset:-1, :]
            self.leftover_s_ie = s_ie[offset:-1, :]
            self.spt_e0 = spt_e[-1, :]

        minute = (time.time() - tic) / 60
        print('Simulation complete.')
        print("Run time (minutes):", minute)

        if cache:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = f"./data/postsim_state/py-{self.num_e}-{self.num_i}-{timestamp}.mat"
            print(f"Saving postsim state to {path}")
            self.save_state(path)

        return (  # for example run
            self.spike_time_e,
            self.step_size,
            self.duration,
            self.num_e,
            self.j_i,
            self.w_ie,
            self.synchrony,
        )


def main():
    """Example run"""

    neural_model = NeuralModel(
        num_e=160,  # number of excitatory neurons
        num_i=40,  # number of inhibitory neurons
        duration=5000,  # (ms) duration of simulation
        seed=42,
    )

    data = neural_model.run_simulation()
    sptime, step_size, duration, ne, J_I, W_IE, synchrony = data

    t = np.linspace(0.1,
                    duration,
                    int(round((duration - 0.1) / step_size)) + 1)
    t = np.ascontiguousarray(t, dtype=np.float64)

    # Compute Neuron Synchronization.
    from kuramoto_syn import kuramoto_syn
    re = kuramoto_syn(
        sptime=sptime,
        t=t,
        step_size=step_size,
        duration=duration,
        num_neurons=ne,
        fast=True,
    )

    from plotting import plot_kop, plot_avg_synaptic_weight, plot_synchrony
    plot_kop(t, re)
    plot_avg_synaptic_weight(t, J_I, W_IE, duration)
    plot_synchrony(synchrony)


if __name__ == "__main__":
    main()
