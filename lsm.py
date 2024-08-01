import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

np.random.seed(seed=0)

# Lorenz equation data generation
class Lorenz:
    # parameter settings
    def __init__(self, sigma, r, b):
        self.sigma = sigma
        self.r = r
        self.b = b

    def f1(self, t, x, y, z):
        return -self.sigma*x + self.sigma*y

    def f2(self, t, x, y, z):
        return -x*z + self.r*x - y

    def f3(self, t, x, y, z):
        return x*y - self.b*z

    def Lorenz(self, t, X):
        next_X = [self.f1(t, X[0], X[1], X[2]), 
                  self.f2(t, X[0], X[1], X[2]), 
                  self.f3(t, X[0], X[1], X[2])]
        return np.array(next_X)

    # 4th-order Runge-Kutta
    def Runge_Kutta(self, x0, T, dt):
        X = x0
        t = 0
        data = []
        while t < T:
            k_1 = self.Lorenz(t, X)
            k_2 = self.Lorenz(t + dt/2, X + dt*k_1/2)
            k_3 = self.Lorenz(t + dt/2, X + dt*k_2/2)
            k_4 = self.Lorenz(t + dt, X + dt*k_3)
            next_X = X + dt/6*(k_1 + 2*k_2 + 2*k_3 + k_4)
            data.append(next_X)
            X = next_X
            t = t + dt
        return np.array(data)

# Liquid State Machine class definitions（LIF model）
class LiquidStateMachine:
    def __init__(self, input_dim, reservoir_size, spectral_radius=0.95, sparsity=0.9, v_th=1.0, tau=20.0, dt=1.0):
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.v_th = v_th
        self.tau = tau
        self.dt = dt
        self.W_in = np.random.uniform(-1, 1, (self.reservoir_size, self.input_dim))
        self.W_res = self._initialize_reservoir()
        self.v = np.zeros(self.reservoir_size)
        self.spikes = np.zeros(self.reservoir_size)

    def _initialize_reservoir(self):
        W = np.random.uniform(-1, 1, (self.reservoir_size, self.reservoir_size))
        W[np.random.rand(*W.shape) < self.sparsity] = 0
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        W *= self.spectral_radius / radius
        return W

    def step(self, input_vector):
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

if __name__ == '__main__':
    # parameter
    T = 1000  # total data length
    T_train, T_test = int(T*0.8), int(T*0.2)  # learning test data length
    dt = 0.02  # step width

    spectral_radius = 0.7
    for _ in range(1):
        # Lorentz data generation with random initial value
        x0 = np.random.uniform(0.5, 1.5, 3)  # generate initial value randomly
        
        dynamics = Lorenz(10.0, 28.0, 8.0/3.0)
        data = dynamics.Runge_Kutta(x0, T_train + T_test, dt)

        # training and test
        train_U = data[:int(T_train/dt)]
        train_D = data[1:int(T_train/dt)+1]
        
        test_U = data[int(T_train/dt):int((T_train+T_test)/dt)]
        test_D = data[1+int(T_train/dt):int((T_train+T_test)/dt)+1]
        
        # LSM model
        lsm = LiquidStateMachine(input_dim=3, reservoir_size=300, spectral_radius=spectral_radius)
        
        # get reservoir state
        reservoir_states_train = lsm.get_states(train_U)
        reservoir_states_test = lsm.get_states(test_U)
        
        # traning read out layer with ridge regression
        ridge_reg = Ridge(alpha=1e-4)
        ridge_reg.fit(reservoir_states_train, train_D)
        
        # predicting test data
        test_Y = ridge_reg.predict(reservoir_states_test)
        
    plt.figure(figsize=(14, 10))
    plt.plot(test_D[:, 0], label='True X')
    plt.plot(test_Y[:, 0], label=f'Predicted X (Spectral Radius={spectral_radius})')
    plt.legend()
    plt.xlabel('Time steps')
    plt.ylabel('X')
    plt.title(f'Time Series Prediction (Spectral Radius={spectral_radius})')
    
    plt.tight_layout()
    plt.savefig("time_series_predictions_240730.pdf")
    plt.show()
