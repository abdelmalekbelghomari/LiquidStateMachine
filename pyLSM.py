import numpy as np

class LiquidStateMachine:
    def __init__(self, input_dim, reservoir_size, excitatory_ratio= 0.8, spectral_radius=0.95, sparsity=0.1, v_th=1.0, tau=20.0, dt=1.0):
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.excitatory_ratio = excitatory_ratio
        self.num_excitatory = int(self.reservoir_size * self.excitatory_ratio)
        self.num_inhibitory = self.reservoir_size - self.num_excitatory
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.v_th = v_th
        self.tau = tau
        self.dt = dt
        self.W_in_standard_deviation = 1.0/np.sqrt(input_dim)
        self.W_res_standard_deviation = 1.0/np.sqrt(reservoir_size)
        self.W_in = np.random.normal(0, self.W_in_standard_deviation, (self.reservoir_size, self.input_dim))
        self.W_res = self._initialize_reservoir()
        self.v = np.zeros(self.reservoir_size)
        self.spikes = np.zeros(self.reservoir_size)

    def _initialize_reservoir(self):
        W = np.random.normal(0, self.W_res_standard_deviation, (self.reservoir_size, self.reservoir_size))
        W[np.random.rand(*W.shape) < self.sparsity] = 0

        # Excitatory connections
        W[:self.num_excitatory,:] *= np.abs(W[:self.num_excitatory,:])
        # Inhibitory connections
        W[self.num_excitatory:,:] *= -np.abs(W[self.num_excitatory:,:])

        radius = np.max(np.abs(np.linalg.eigvals(W)))
        W *= self.spectral_radius / radius
        return W

    def step(self, input_vector):
        if self.input_dim == 1:
            input_vector = input_vector.reshape(-1)
        I = np.dot(self.W_in, input_vector) + np.dot(self.W_res, self.spikes)
        dv = (I - self.v) / self.tau
        self.v += dv * self.dt
        self.spikes = (self.v >= self.v_th).astype(float)
        self.v[self.spikes == 1] = 0
        return self.spikes

    def get_states(self, inputs):
        states = []
        for input_vector in inputs:
            states.append(self.step(input_vector))
        return np.array(states)
    
    def calculate_lyapunov_exponent(self, states):
        # Calculate the largest Lyapunov exponent for the given states
        n = len(states)
        divergence = np.zeros(n-1)
        for i in range(1, n):
            divergence[i-1] = np.linalg.norm(states[i] - states[i-1])
        divergence = np.log(divergence)
        return np.mean(divergence)

    def test_esp(self, states):
        lyapunov_exponent = self.calculate_lyapunov_exponent(states)
        print(f"Largest Lyapunov Exponent: {lyapunov_exponent}")
        if lyapunov_exponent < 0:
            print("ESP is likely satisfied (stable system).")
        else:
            print("ESP is likely violated (chaotic system).")