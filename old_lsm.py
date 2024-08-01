import nest
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from lsm.nest import LSM
import nest.raster_plot as raster_plot
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import os

# Define parameters
n_exc = 1000  # Number of excitatory neurons
n_inh = 250   # Number of inhibitory neurons
n_rec = 500   # Number of recurrent neurons to record

sim_time = 20000 # Simulation time in ms
stim_interval = 300  # Interval between stimuli

# Generate Lorenz Attractor Data
def generate_lorenz_data(num_steps, dt=0.01, sigma=10.0, rho=28.0, beta=8.0/3.0, save_path='lorenz_data.csv', regenerate=False):
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

def inject_currents(data, neuron_targets):

    def generate_delay_normal_clipped(mu=10., sigma=20., low=3., high=200.):
        delay = np.random.normal(mu, sigma)
        delay = max(min(delay, high), low)
        return delay

    times = [float(i+1) for i in range(len(data))]
    generators = nest.Create('step_current_generator',params={"amplitude_values":data, "amplitude_times":times})
    C_inp = 100 # int(N_E / 20)  # number of outgoing input synapses per input neuron
    nest.Connect(generators, neuron_targets,
                {'rule': 'fixed_outdegree', 'outdegree': C_inp},
                {"synapse_model": "static_synapse",
                "delay": generate_delay_normal_clipped(),
                "weight":  {'distribution': 'uniform',
                             'low': 2.5 * 10 * 5.0,
                             'high': 7.5 * 10 * 5.0}})

def visualize_networks(self, M):
        # extract position information, transpose to list of x, y and z positions
        xpos, ypos, zpos = zip(*nest.GetPosition(M))
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection="3d")
        ax1.scatter(xpos, ypos, zpos, s=15, facecolor="b")
        plt.show()

def plot_raster(lsm):
    # Increase the figure size and DPI for better resolution
    # plt.figure(figsize=(12, 8), dpi=100)

    # Generate the raster plot
    nest.raster_plot.from_device(lsm._rec_detector, hist=False, title="Raster Plot of Recurrent Neurons")

    # Show the plot
    plt.show()

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)
# Main function
def main():
    nest.SetKernelStatus({'print_time': True, 'local_num_threads': 8})  # Set resolution to 0.1 ms

    # Generate Lorenz attractor data
    lorenz_data = generate_lorenz_data(sim_time)  #  time steps

    lorenz_data_transposed = lorenz_data.T

    x , y, z = lorenz_data_transposed

    scaler = StandardScaler()
    x_normalized = scaler.fit_transform(x.reshape(-1, 1)).reshape(-1)
    y_normalized = scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)
    z_normalized = scaler.fit_transform(z.reshape(-1, 1)).reshape(-1)

    lorenz_data_normalized = np.vstack((x_normalized,y_normalized,z_normalized)).T

    # Initialize the LSM
    lsm = LSM(n_exc=n_exc, n_inh=n_inh, n_rec=n_rec)

    # Inject currents into the LSM using only the X component
    num_neurons = len(lsm.exc_nodes)
    neurons_per_component = num_neurons // 3

    inject_currents(lorenz_data_normalized[:,0], lsm.inp_nodes[:neurons_per_component])
    print("Currents injected successfully with X values only!")  # Debugging print
    inject_currents(lorenz_data_normalized[:,1], lsm.inp_nodes[neurons_per_component:2*neurons_per_component])
    print("Currents injected successfully with Y values only!")  # Debugging print
    inject_currents(lorenz_data_normalized[:,2], lsm.inp_nodes[2*neurons_per_component:])
    print("Currents injected successfully with Z values only!")  # Debugging print

    # Visualize the reservoir
    # visualize_reservoir(lsm.exc_nodes, lsm.inh_nodes)
    # visualize_networks(lsm, lsm.rec_nodes)

    # Simulate
    nest.Simulate(sim_time)

    readout_times = np.arange(0, sim_time, 0.1)
    targets = lorenz_data_normalized[:len(readout_times), 0]
    
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

    

    plt.figure(figsize=(12, 6)) 
    plt.plot(range(len(lorenz_data)), lorenz_data[:, 0], 'g-', label='True X (first 80%)')
    plt.plot(range(len(targets)), targets, 'r--', label='targets X')
    plt.plot(range(len(train_targets)), train_targets, 'b--', label='Train Targets X')
    plt.xlabel('Example')
    plt.ylabel('Value')
    plt.legend()
    plt.title('First 80% of Lorenz Data vs. Train Targets')
    plt.show()

    readout_weights = lsm.compute_readout_weights(train_states, train_targets, reg_fact=1.0)
    train_prediction = lsm.compute_prediction(train_states, readout_weights)
    test_prediction = lsm.compute_prediction(test_states, readout_weights)

    print("simulation time: {}ms".format(sim_time))
    print("number of stimuli: {}".format(len(readout_times)))
    print("size of each state: {}".format(np.size(states, 1)))

    print("---------------------------------------")

    def eval_prediction(prediction, targets, label):
        try:
            mse = np.mean((prediction - targets) ** 2)
            print(f"{label} MSE: {mse}")
        except Exception as e:
            print(f"Error: {e}")

    eval_prediction(train_prediction, train_targets, "Training")
    eval_prediction(test_prediction, test_targets, "Test")

    # Plot the Lorenz attractor time series
    plt.figure(figsize=(12, 6)) 
    plt.subplot(3, 1, 1)
    plt.plot(lorenz_data[16000:, 0], label='True X')
    # plt.plot(range(1000), test_targets, 'r--', label='Targets X')
    # plt.plot(range(4000), test_prediction, 'b--', label='Predictions X')
    plt.plot()
    plt.ylabel('X')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(lorenz_data[16000:, 1], label='True Y')
    plt.ylabel('Y')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(lorenz_data[16000:, 2], label='True Z')
    plt.xlabel('Time Step')
    plt.ylabel('Z')
    plt.legend()
    plt.suptitle('Lorenz Attractor Time Series')
    plt.show()

if __name__ == "__main__":
    main()
