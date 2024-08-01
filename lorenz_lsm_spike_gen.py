import nest
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from lsm.nest import LSM
import nest.raster_plot as raster_plot
from lsm.utils import poisson_generator
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

# Define parameters
n_exc = 1000  # Number of excitatory neurons
n_inh = 250   # Number of inhibitory neurons
n_rec = 500   # Number of recurrent neurons to record

sim_time = 200000 # Simulation time in ms
stim_interval = 300  # Interval between stimuli
stim_length = 50  # Length of stimuli
stim_rate = 200  # Stimulus rate in Hz
readout_delay = 10 # Delay before readout

# Generate Lorenz Attractor Data
def generate_lorenz_data(num_steps, dt=0.1, sigma=10.0, rho=28.0, beta=8.0/3.0, save_path='lorenz_data.csv', regenerate=False):
    if not regenerate and os.path.exists(save_path):
        print("Loading Lorenz Attractor Data from CSV...")
        data = pd.read_csv(save_path)
        print("Data loaded from CSV.")
        return data.values
    
    def lorenz(t, state):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]

    t_span = (0, num_steps * dt)
    t_eval = np.arange(0, num_steps * dt, dt)
    initial_state = [1.0, 1.0, 1.0]
    sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval, method='RK45')

    print("Lorenz Attractor Data Shape:", sol.y.T.shape)  # Debugging print

    # Save the data to a CSV file
    data = pd.DataFrame(sol.y.T, columns=['x', 'y', 'z'])
    data.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")

    return sol.y.T  # Transpose to get shape (num_steps, 3)

def generate_spike_times_lorenz(data, stim_times, gen_burst):
    """
    Generate spike times based on the Lorenz X component.

    Parameters:
    - data: array-like, values of the Lorenz X component.
    - stim_times: array-like, times at which stimuli are given.
    - gen_burst: function, generates a burst of spikes around a given time.
    - scale_factor: float, factor to scale the spike times.

    Returns:
    - inp_spikes: list of arrays, each array contains the spike times for an input neuron.
    """

    inp_spikes = []

    for value, t in zip(data, stim_times):
        spike_count = int(value)
        if spike_count > 1e-6:
            spikes = np.concatenate([t + gen_burst() for _ in range(spike_count)])
            
            # Scale and adjust spike times
            spikes *= 10
            spikes = spikes.round() + 1.0
            spikes = spikes / 10.0

            spikes = np.sort(spikes)

            inp_spikes.append(spikes)
        else:
            inp_spikes.append(np.array([]))

    return inp_spikes

def inject_spikes(inp_spikes, neuron_targets):
    spike_generators = nest.Create("spike_generator", len(inp_spikes))
    for sg, sp in zip(spike_generators, inp_spikes):
        nest.SetStatus([sg], {'spike_times': sp})
    C_inp = 100  # int(N_E / 20)  # number of outgoing input synapses per input neuron
    delay = dict(distribution='normal_clipped', mu=10., sigma=20., low=3., high=200.)
    nest.Connect(spike_generators, neuron_targets,
                 {'rule': 'fixed_outdegree',
                  'outdegree': C_inp},
                 {'synapse_model': 'static_synapse',
                  'delay': delay,
                  'weight': {'distribution': 'uniform',
                             'low': 2.5 * 10 * 5.0,
                             'high': 7.5 * 10 * 5.0}
                  })
    
def plot_raster(lsm):
    nest.raster_plot.from_device(lsm._rec_detector, hist=False, title="Raster Plot of Recurrent Neurons")
    plt.show()

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)

# Main function
def main():
    nest.SetKernelStatus({'print_time': True, 'local_num_threads': 8})

    print("Setting up LSM...")
    stim_times = np.arange(stim_interval, sim_time - stim_length - readout_delay, stim_interval)
    readout_times = stim_times + stim_length + readout_delay

    # Generate Lorenz attractor data
    print("Generating Lorenz Attractor Data...")
    lorenz_data = generate_lorenz_data(sim_time)  #  time steps

    lorenz_data_transposed = lorenz_data.T

    x , y, z = lorenz_data_transposed

    x_normalized = StandardScaler().fit_transform(x.reshape(-1, 1)).reshape(-1)

    def gen_stimulus_pattern(): return poisson_generator(stim_rate, t_stop=stim_length)

    inp_spikes = generate_spike_times_lorenz(x_normalized, stim_times, gen_burst=gen_stimulus_pattern)

    print("input spikes value:", inp_spikes)
    print("input spikes shape:", len(inp_spikes))


    lsm = LSM(n_exc=n_exc, n_inh=n_inh, n_rec=n_rec)

    inject_spikes(inp_spikes, lsm.inp_nodes)

    # Simulate
    nest.Simulate(readout_times[-1] + 1000)

    print("len of readout times:", len(readout_times))
    print("len of x_normalized:", len(x_normalized))
    targets = x_normalized[:len(readout_times)]
    
    print("getting states")
    states = lsm.get_states(readout_times, tau=20)
    # print("States shape:", states.shape)  # Debugging print

    plot_raster(lsm)

    # Add a constant component to states for bias
    states = np.hstack([states, np.ones((np.size(states, 0), 1))])

    n_examples = np.size(targets, 0)
    n_examples_train = int(n_examples * 0.8)

    train_states, test_states = states[:n_examples_train, :], states[n_examples_train:, :]
    train_targets, test_targets = targets[:n_examples_train], targets[n_examples_train:]

    # Debugging print statements
    print("train_states.shape:", train_states.shape)
    print("test_states.shape:", test_states.shape)
    print("train_targets.shape:", train_targets.shape)
    print("test_targets.shape:", test_targets.shape)

    # Plot the first 80% of the Lorenz data and the train targets
    # plt.figure(figsize=(12, 6)) 
    # plt.plot(range(len(targets)), targets, 'r--', label='targets X')
    # plt.plot(range(len(train_targets)), train_targets, 'b--', label='Train Targets X')
    # plt.xlabel('Example')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.title('First 80% of Lorenz Data vs. Train Targets')
    # plt.show()

    readout_weights = lsm.compute_readout_weights(train_states, train_targets, reg_fact=5.0)

    def classify(prediction):
        return (prediction >= 0.5).astype(int)
    
    train_prediction = lsm.compute_prediction(train_states, readout_weights)
    train_results = classify(train_prediction)

    test_prediction = lsm.compute_prediction(test_states, readout_weights)
    test_results = classify(test_prediction)

    readout = Ridge(alpha=5.0)
    readout.fit(train_states, train_targets)
    train_prediction_ridge = readout.predict(train_states)
    test_prediction_ridge = readout.predict(test_states)

    print("simulation time: {}ms".format(sim_time))
    print("number of stimuli: {}".format(len(stim_times)))
    print("size of each state: {}".format(np.size(states, 1)))

    print("---------------------------------------")

    def eval_prediction(prediction, targets, label):
        n_fails = sum(abs(prediction - targets))
        n_total = len(targets)
        print("mismatched {} examples: {:.1f}/{:.1f} [{:.1f}%]".format(label, n_fails, n_total, float(n_fails) / n_total * 100))

    eval_prediction(train_results, train_targets, "training")
    eval_prediction(test_results, test_targets, "test")
    eval_prediction(train_prediction_ridge, train_targets, "training ridge")
    eval_prediction(test_prediction_ridge, test_targets, "test ridge")
    print("---------------------------------------")

    def eval_prediction(prediction, targets, label):
        try:
            mse = np.mean((prediction - targets) ** 2)
            print(f"{label} MSE: {mse}")
        except Exception as e:
            print(f"Error: {e}")

    eval_prediction(train_prediction, train_targets, "Training")
    eval_prediction(test_prediction, test_targets, "Test")
    eval_prediction(train_prediction_ridge, train_targets, "Training Ridge")
    eval_prediction(test_prediction_ridge, test_targets, "Test Ridge")

    

    # Visualize the predictions
    plt.figure(figsize=(12, 6))
    plt.plot(test_targets, 'b-', label='Targets')
    plt.plot(test_prediction, 'r--', label='Predictions')
    plt.plot(test_prediction_ridge, 'g--', label='Predictions Ridge')
    plt.xlabel('Example')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Test Set Predictions vs. Targets')
    plt.show()

    # # Plot the Lorenz attractor time series
    # plt.figure(figsize=(12, 6)) 
    # plt.subplot(3, 1, 1)
    # plt.plot(lorenz_data_normalized[:, 0], label='True X')
    # plt.ylabel('X')
    # plt.legend()
    # plt.subplot(3, 1, 2)
    # plt.plot(lorenz_data_normalized[:, 1], label='True Y')
    # plt.ylabel('Y')
    # plt.legend()
    # plt.subplot(3, 1, 3)
    # plt.plot(lorenz_data_normalized[:, 2], label='True Z')
    # plt.xlabel('Time Step')
    # plt.ylabel('Z')
    # plt.legend()
    # plt.suptitle('Lorenz Attractor Time Series')
    # plt.show()

if __name__ == "__main__":
    main()