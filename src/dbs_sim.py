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
from kuramoto_syn import kuramoto_syn
from pulsatile_input import pulsatile_input
from ode_neuron_model import ode_neuron_model
from make_synaptic_connections import make_synaptic_connections
from plotting import plot_kop, plot_avg_synaptic_weight, plot_synchrony


INHIBITION_THRESHOLD = 75  # (mV) threshold for inhibition
STIMULATION_ONSET_TIME_RATIO = 0.08  # begin after 8% of the simulation time
PLASTICITY_ONSET_TIME_RATIO = 0.004  # begin after 0.4% of the simulation time
# simulation duration = 25,000 --> sample_duration = 20
# 25,000 * 0.0008 = 20


def dbs_simulation(
        duration=25_000,  # (ms) duration of simulation
        N_E=1600,
        N_I=400,
        seed=None,
        cache=False,
):
    """
    Sets up and runs the DBS simulation.

    If `cache` is True, simulation state from before and after the simulation
    will be saved.
    """

    if seed:
        np.random.seed(seed)

    # Make ode folder
    # ode_path = f"./data/ode/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    # os.mkdir(ode_path)

    tic = time.time()

    # Run Parameters.
    step_size = 0.1  # (ms)
    num_steps = int(duration / step_size)
    stim_onset_time = int(duration * STIMULATION_ONSET_TIME_RATIO)
    plast_onset_time = int(duration * PLASTICITY_ONSET_TIME_RATIO)

    # Neuron Parameters.
    mew_e = 20.8
    sigma_e = 1
    mew_i = 18
    sigma_i = 3
    mew_c = 0
    Ntot = N_E + N_I

    # Synaptic Parameters.
    Weight_0 = 1
    J_E = 100  # synaptic strength (J_E = J_EI)
    J_I = 260  # synaptic strength (J_I = J_IE)
    N_i = 1  # copilot: number of synapses per neuron?
    C_E = 0.3 * Ntot  # N_I;
    C_I = 0.3 * Ntot  # N_E;
    tau_LTP = 20  # long-term potentiation time constant
    tau_LTD = 22  # long-term depression time constant

    # Make Random Synaptic Conncetions.
    epsilon_E = 0.1  # connectivity
    epsilon_I = 0.1  # connectivity

    S_key_EI, num_synapses_EI = make_synaptic_connections(  # I -> E
        num_pre=N_I,
        num_post=N_E,
        epsilon=epsilon_I,
    )
    S_key_IE, num_synapses_IE = make_synaptic_connections(  # E -> I
        num_pre=N_E,
        num_post=N_I,
        epsilon=epsilon_E,
    )

    W_IE = np.zeros((num_steps, 1))
    W_IE_std = np.zeros((int(num_steps), 1))

    # Initial Conditions.
    vE0 = 14 * np.ones((1, N_E))
    vI0 = 14 * np.ones((1, N_I))
    S_EI0 = np.zeros((1, N_E))
    S_IE0 = np.zeros((1, N_I))
    X_EI0 = np.zeros((1, N_E))
    X_IE0 = np.zeros((1, N_I))
    Apost0 = np.zeros((1, int(num_synapses_IE)))
    Apre0 = np.zeros((1, int(num_synapses_IE)))
    W_IE0 = Weight_0 * np.ones((1, int(num_synapses_IE)))
    W_EI0 = Weight_0
    leftover_S_EI = np.zeros((int(5/step_size) + 1, N_E))
    leftover_S_IE = np.zeros((int(5/step_size) + 1, N_I))
    ref_E = np.zeros((1, N_E))
    ref_I = np.zeros((1, N_I))
    spt_E0 = 0
    spE_count0 = 0
    phi0 = np.zeros((1, N_E))
    phif = np.zeros((1, N_E))

    sample_duration = 20
    num_samples = int(duration / sample_duration)
    Synchrony = np.zeros((int(num_samples), 1))
    time_syn = np.zeros((int(num_samples), 1))
    num_steps_per_sample = int(sample_duration / step_size)
    spike_time_E = np.zeros((num_steps, N_E))
    spE_count = np.zeros((int(num_steps_per_sample), N_E))

    # TO MAKE FIGURE 7
    tau_E_m = np.full((1, N_E), 10)
    tau_I_m = np.full((1, N_I), 10)

    # Generate General Stimulation Pattern
    cross_100 = 1
    comp_time = 0
    V_stim = 1  # (mV) stimulation amplitude (todo: rename: Ustim)
    T_stim = 1  # (ms)
    x_neutral = 10  # (ms) (todo: rename: t_neutral)
    multiple = 1
    t_pulse = T_stim * (x_neutral + multiple + 1)
    Ue, Ui = pulsatile_input(  # todo: rename Ve, Vi?
        multi=multiple,
        v_stim=V_stim,
        t_stim=T_stim,
        x=x_neutral,
        duration=duration,
        step_size=step_size
    )

    Ue = Ue.reshape(1, -1)  # (N,) -> (1, N)
    Ui = Ui.reshape(1, -1)  # (N,) -> (1, N)

    time_array = np.zeros((num_steps, 1))

    # Save Presim State.
    if cache:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"./data/presim_state/py-{N_E}-{N_I}-{timestamp}.mat"
        print(f"Saving presim state to {filename}")
        savemat(filename, {
            # Run Parameters
            'duration': duration, 'step_size': step_size, 'num_steps': num_steps,

            # Neuron Parameters
            'mew_e': mew_e, 'sigma_e': sigma_e, 'mew_i': mew_i, 'sigma_i': sigma_i,
            'mew_c': mew_c, 'N_E': N_E, 'N_I': N_I, 'Ntot': Ntot,

            # Synaptic Parameters
            'Weight_0': Weight_0, 'J_E': J_E, 'J_I': J_I, 'N_i': N_i,
            'C_E': C_E, 'C_I': C_I, 'tau_LTP': tau_LTP, 'tau_LTD': tau_LTD,

            # Make Random Synaptic Connections
            'epsilon_E': epsilon_E, 'epsilon_I': epsilon_I,
            'S_key_IE': S_key_IE, 'S_key_EI': S_key_EI,
            'num_synapses_IE': num_synapses_IE, 'num_synapses_EI': num_synapses_EI,
            'W_IE': W_IE, 'W_IE_std': W_IE_std,

            # Initial Conditions
            'vE0': vE0, 'vI0': vI0, 'S_EI0': S_EI0, 'S_IE0': S_IE0,
            'X_IE0': X_IE0, 'X_EI0': X_EI0, 'Apost0': Apost0, 'Apre0': Apre0,
            'W_IE0': W_IE0, 'W_EI0': W_EI0,
            'leftover_S_EI': leftover_S_EI, 'leftover_S_IE': leftover_S_IE,
            'ref_E': ref_E, 'ref_I': ref_I, 'spt_E0': spt_E0,
            'spE_count0': spE_count0, 'phi0': phi0, 'phif': phif,
            'sample_duration': sample_duration, 'num_samples': num_samples,
            'Synchrony': Synchrony, 'time_syn': time_syn,
            'num_steps_per_sample': num_steps_per_sample,
            'spike_time_E': spike_time_E, 'spE_count': spE_count,
            'tau_E_m': tau_E_m, 'tau_I_m': tau_I_m,

            # Generate General Stimulation Pattern
            'cross_100': cross_100, 'comp_time': comp_time, 'V_stim': V_stim,
            'T_stim': T_stim, 'x_neutral': x_neutral, 'multiple': multiple,
            't_pulse': t_pulse, 'Ue': Ue, 'Ui': Ui,

            # Other
            'time_array': time_array,
        })

    # Run Simulation.
    for i in tqdm(range(1, num_samples + 1), "Simulating", unit="sample"):

        comp_time = (i - 1) * sample_duration

        # Calculate Average Effective Inhibitition.
        if np.mean(W_IE0) * J_I < INHIBITION_THRESHOLD:
            cross_100 = 0

        # Toggle Stimulation and Plasticity.
        stim_on = ((i * sample_duration) >= stim_onset_time) * cross_100
        plast_on = (i * sample_duration) >= plast_onset_time
        assert stim_on in (0, 1)
        assert plast_on in (0, 1)

        # Define the Sample Window.
        sample_start = (i >= 2) * (i - 1) * num_steps_per_sample
        sample_end = i * num_steps_per_sample
        assert sample_end - sample_start == num_steps_per_sample
        # sample_window = f"{sample_start}:{sample_end - 1}"
        # if i <= 1200 and i % 100 == 0:
        #     print(f"\tSample: {i} / {num_samples}\t({sample_window})")
        # elif i > 1200 and i % 10 == 0:
        #     print(f"\tSample: {i} / {num_samples}\t({sample_window})")

        # Define the Stimulation Input.
        Vstim = 100  # mV (todo: rename: Ustim?)
        ue = Vstim * Ue[:, sample_start:sample_end]
        ui = Vstim * Ui[:, sample_start:sample_end]
        assert ue.shape[1] == num_steps_per_sample
        assert ui.shape[1] == num_steps_per_sample

        # Run the ODE Neuron Model.
        percent_V_stim = 1
        (
            time_m,
            vE_m,
            vI_m,
            S_EI_m,
            S_IE_m,
            X_EI_m,
            X_IE_m,
            Apost_m,
            Apre_m,
            W_IE_m,
            spike_E_m,
            spike_I_m,
            ref_E_m,
            ref_I_m,
            synchrony_m,
            spt_E_m,
            phif
        ) = ode_neuron_model(
            plast_on=plast_on,
            ON=stim_on,
            vE0=vE0,
            vI0=vI0,
            S_EI0=S_EI0,
            S_IE0=S_IE0,
            X_EI0=X_EI0,
            X_IE0=X_IE0,
            Apost0=Apost0,
            Apre0=Apre0,
            W_IE0=W_IE0,
            W_EI0=W_EI0,
            mew_e=mew_e,
            sigma_e=sigma_e,
            ue=ue,
            ui=ui,
            mew_i=mew_i,
            sigma_i=sigma_i,
            J_E=J_E,
            J_I=J_I,
            C_E=C_E,
            C_I=C_I,
            tau_LTP=tau_LTP,
            tau_LTD=tau_LTD,
            step_size=step_size,
            sample_duration=sample_duration,
            N_E=N_E,
            N_I=N_I,
            S_key_EI=S_key_EI,
            S_key_IE=S_key_IE,
            leftover_S_EI=leftover_S_EI,
            leftover_S_IE=leftover_S_IE,
            ref_E=ref_E,
            ref_I=ref_I,
            tau_E_m=tau_E_m,
            tau_I_m=tau_I_m,
            percent_V_stim=percent_V_stim,
            comp_time=comp_time,
            spt_E0=spt_E0,
            phif=phif,
        )

        # recorded variables
        time_array[sample_start:sample_end, 0] = (
            time_m[0:-1, 0] + (i - 1) * sample_duration
        )
        W_IE[sample_start:sample_end, 0] = np.mean(W_IE_m[0:-1, :], axis=1)
        # spike_E[sample_start:sample_end, :] = spike_E_m[0:num_steps_per_sample, :]
        # spike_I[sample_start:sample_end, :] = spike_I_m[0:num_steps_per_sample, :]
        Synchrony[i - 1, 0] = synchrony_m
        time_syn[i - 1, 0] = sample_duration * (i)
        spike_time_E[sample_start:sample_end, :] = spt_E_m[0:-1, :]

        # Update Intial Conditions (for next run).
        vE0 = vE_m[-1, :].reshape(1, -1)
        vI0 = vI_m[-1, :].reshape(1, -1)
        S_EI0 = S_EI_m[-1, :].reshape(1, -1)
        S_IE0 = S_IE_m[-1, :].reshape(1, -1)
        X_EI0 = X_EI_m[-1, :].reshape(1, -1)
        X_IE0 = X_IE_m[-1, :].reshape(1, -1)
        Apost0 = Apost_m[-1, :].reshape(1, -1)
        Apre0 = Apre_m[-1, :].reshape(1, -1)
        W_IE0 = W_IE_m[-1, :].reshape(1, -1)
        W_EI0 = Weight_0
        left_sample_end = num_steps_per_sample - int(5 / step_size)
        leftover_S_EI = S_EI_m[left_sample_end:-1, :]
        leftover_S_IE = S_IE_m[left_sample_end:-1, :]
        spt_E0 = spt_E_m[-1, :]

    minute = (time.time() - tic) / 60
    print('Simulation complete.')
    print("Run time (minutes):", minute)

    # Save Postsim State.
    if cache:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"./data/postsim_state/py-{N_E}-{N_I}-{timestamp}.mat"
        print(f"Saving postsim state to {filename}")
        savemat(filename, {
            # Run Parameters
            'duration': duration, 'step_size': step_size, 'num_steps': num_steps,

            # Neuron Parameters
            'mew_e': mew_e, 'sigma_e': sigma_e, 'mew_i': mew_i, 'sigma_i': sigma_i,
            'mew_c': mew_c, 'N_E': N_E, 'N_I': N_I, 'Ntot': Ntot,

            # Synaptic Parameters
            'Weight_0': Weight_0, 'J_E': J_E, 'J_I': J_I, 'N_i': N_i,
            'C_E': C_E, 'C_I': C_I, 'tau_LTP': tau_LTP, 'tau_LTD': tau_LTD,

            # Make Random Synaptic Connections
            'epsilon_E': epsilon_E, 'epsilon_I': epsilon_I,
            'S_key_IE': S_key_IE, 'S_key_EI': S_key_EI,
            'num_synapses_IE': num_synapses_IE, 'num_synapses_EI': num_synapses_EI,
            'W_IE': W_IE, 'W_IE_std': W_IE_std,

            # Initial Conditions
            'vE0': vE0, 'vI0': vI0, 'S_EI0': S_EI0, 'S_IE0': S_IE0,
            'X_IE0': X_IE0, 'X_EI0': X_EI0, 'Apost0': Apost0, 'Apre0': Apre0,
            'W_IE0': W_IE0, 'W_EI0': W_EI0,
            'leftover_S_EI': leftover_S_EI, 'leftover_S_IE': leftover_S_IE,
            'ref_E': ref_E, 'ref_I': ref_I, 'spt_E0': spt_E0,
            'spE_count0': spE_count0, 'phi0': phi0, 'phif': phif,
            'sample_duration': sample_duration, 'num_samples': num_samples,
            'Synchrony': Synchrony, 'time_syn': time_syn,
            'num_steps_per_sample': num_steps_per_sample,
            'spike_time_E': spike_time_E, 'spE_count': spE_count,
            'tau_E_m': tau_E_m, 'tau_I_m': tau_I_m,

            # Generate General Stimulation Pattern
            'cross_100': cross_100, 'comp_time': comp_time, 'V_stim': V_stim,
            'T_stim': T_stim, 'x_neutral': x_neutral, 'multiple': multiple,
            't_pulse': t_pulse, 'Ue': Ue, 'Ui': Ui,

            # Other
            'time_array': time_array,
        })

    return (  # for example run
        spike_time_E,
        step_size,
        duration,
        N_E,
        J_I,
        W_IE,
        Synchrony,
    )


def main():
    """Example run"""

    data = dbs_simulation(
        duration=5000,
        N_E=160,
        N_I=40,
        seed=42,
        cache=False,
    )

    sptime, step_size, duration, ne, J_I, W_IE, synchrony = data

    t = np.linspace(
        0.1,
        duration,
        int(round((duration - 0.1) / step_size)) + 1
    )
    t = np.ascontiguousarray(t, dtype=np.float64)

    # Compute Neuron Synchronization.
    re = kuramoto_syn(
        sptime=sptime,
        t=t,
        step_size=step_size,
        duration=duration,
        num_neurons=ne,
        fast=True,
    )

    plot_kop(t, re)
    plot_avg_synaptic_weight(t, J_I, W_IE, duration)
    plot_synchrony(synchrony)


if __name__ == "__main__":
    main()
