import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import pandas as pd
import os
from scipy.integrate import solve_ivp
import pyLSM as lsm


np.random.seed(seed=42)

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

if __name__ == '__main__':
    # parameter
    T = 1000  # total data length
    T_train, T_test = int(T*0.8), int(T*0.2)  # learning test data length
    dt = 0.02  # step width
    num_steps = int(T/dt)

    spectral_radius = 0.90
    sparsity = 0.99

    # Lorentz data generation with random initial value        
    data = generate_lorenz_data(num_steps, dt)
    # training and test
    train_U = data[:int(T_train/dt)]
    train_D = data[1:int(T_train/dt)+1]
    
    test_U = data[int(T_train/dt):int((T_train+T_test)/dt)]
    test_D = data[1+int(T_train/dt):int((T_train+T_test)/dt)+1]
    
    # LSM model
    lsm = lsm.LiquidStateMachine(input_dim=3, reservoir_size=300, spectral_radius=spectral_radius, sparsity=sparsity)
    
    # get reservoir state
    reservoir_states_train = lsm.get_states(train_U)
    reservoir_states_test = lsm.get_states(test_U)
    
    # traning read out layer with ridge regression
    ridge_reg = Ridge(alpha=1e-4)
    ridge_reg.fit(reservoir_states_train, train_D)
    
    # predicting test data
    test_Y = ridge_reg.predict(reservoir_states_test)
        
    plt.figure(figsize=(14, 10))
    plt.plot(test_U[:1000, 0], label='True X')
    plt.plot(test_Y[:1000, 0], label=f'Predicted X (Spectral Radius={spectral_radius})')
    plt.legend()
    plt.xlabel('Time steps')
    plt.ylabel('X')
    plt.title(f'Time Series Prediction (Spectral Radius={spectral_radius})')
    
    plt.tight_layout()
    plt.savefig("time_series_predictions_240730.pdf")
    plt.show()

