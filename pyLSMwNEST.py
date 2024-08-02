import nest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class LiquidStateMachineNEST:
    def __init__(self, n_exc=800, n_inh=200, connectivity=0.1):
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.connectivity = connectivity
        self.nodes_E = nest.Create('iaf_psc_alpha', n_exc)  # Excitatory neurons
        self.nodes_I = nest.Create('iaf_psc_alpha', n_inh)  # Inhibitory neurons
        self._initialize_reservoir()

    def _initialize_reservoir(self):
        nest.CopyModel('static_synapse', 'excitatory', {'weight': 2.5, 'delay': 1.5})
        nest.CopyModel('static_synapse', 'inhibitory', {'weight': -2.5, 'delay': 1.5})

        # Connect neurons
        nest.Connect(self.nodes_E, self.nodes_E, syn_spec='excitatory', 
                     conn_spec={'rule': 'fixed_indegree', 'indegree': int(self.connectivity * self.n_exc)})
        nest.Connect(self.nodes_E, self.nodes_I, syn_spec='excitatory', 
                     conn_spec={'rule': 'fixed_indegree', 'indegree': int(self.connectivity * self.n_exc)})
        nest.Connect(self.nodes_I, self.nodes_E, syn_spec='inhibitory', 
                     conn_spec={'rule': 'fixed_indegree', 'indegree': int(self.connectivity * self.n_inh)})
        nest.Connect(self.nodes_I, self.nodes_I, syn_spec='inhibitory', 
                     conn_spec={'rule': 'fixed_indegree', 'indegree': int(self.connectivity * self.n_inh)})

    def inject_current(self, current_amplitudes):
        # Create step current generators for input current
        times_amplitudes = np.arange(0.1, len(current_amplitudes) * 0.1, 0.1)
        print("len(current_amplitudes):", len(current_amplitudes))
        print("len(times_amplitudes):", len(times_amplitudes))
        generators = nest.Create('step_current_generator', len(current_amplitudes), 
                                 params={'amplitude_values': current_amplitudes[1:], 
                                         'amplitude_times': times_amplitudes})
        nest.Connect(generators, self.nodes_E[:len(current_amplitudes)])

    def get_spike_data(self, sim_time):
        spike_detector = nest.Create('spike_detector')
        nest.Connect(self.nodes_E, spike_detector)
        nest.Simulate(sim_time)
        spike_data = nest.GetStatus(spike_detector, 'events')[0]
        return spike_data

    def get_reservoir_states(self, sim_time):
        spikes = self.get_spike_data(sim_time)
        # Create a binary matrix of spikes
        spike_matrix = np.zeros((self.n_exc, int(sim_time)))
        for neuron_id, time in zip(spikes['senders'], spikes['times']):
            spike_matrix[neuron_id - 1, int(time) - 1] = 1
        return spike_matrix.T  # Transpose to have time steps as rows

# Example usage
if __name__ == '__main__':
    nest.ResetKernel()
    nest.SetKernelStatus({'print_time': True, 'local_num_threads': 8})
    lsm = LiquidStateMachineNEST(n_exc=800, n_inh=200)

    # Generate a sample sinusoidal signal as input
    sim_time = 1000  # in ms
    t = np.linspace(0, 2 * np.pi, sim_time)
    input_signal = np.sin(t)

    # Scale the input signal
    scaler = StandardScaler()
    input_signal_scaled = scaler.fit_transform(input_signal.reshape(-1, 1)).flatten()

    # Inject scaled input signal into the reservoir
    lsm.inject_current(input_signal_scaled)

    # Run the simulation
    reservoir_states = lsm.get_reservoir_states(sim_time)

    # Split data into training and testing sets
    train_size = int(sim_time * 0.8)
    train_states = reservoir_states[:train_size]
    test_states = reservoir_states[train_size:]
    train_targets = input_signal[1:train_size + 1]
    test_targets = input_signal[train_size + 1:]

    # Train a readout layer using Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(train_states, train_targets)

    # Predict the test data
    predictions = ridge.predict(test_states)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(test_targets)), test_targets, label='True Signal')
    plt.plot(range(len(predictions)), predictions, label='Predicted Signal')
    plt.xlabel('Time steps')
    plt.ylabel('Amplitude')
    plt.title('Time Series Prediction with LSM')
    plt.legend()
    plt.show()
