import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import pandas as pd
import os
from scipy.integrate import solve_ivp
import pyLSM as lsm
from reservoirpy.datasets import lorenz
from datetime import datetime


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

def generate_sinusoid(num_steps, dt=0.1, freq=1.0, amplitude=1.0, phase=0.0):
    t = np.arange(0, num_steps * dt, dt)
    return amplitude * np.sin(2 * np.pi * freq * t + phase)

if __name__ == '__main__':
    # parameter
    T = 1000  # total data length
    T_train, T_test = int(T*0.8), int(T*0.2)  # learning test data length
    dt = 0.02  # step width
    num_steps = int(T/dt)

    spectral_radius = 1.0
    sparsity = 0.001

    # Lorentz data generation with random initial value        
    # data = generate_lorenz_data(num_steps, dt)
    data = lorenz(num_steps)
    data_sinusoid = generate_sinusoid(num_steps, dt, freq=0.1, amplitude=1.0, phase=0.0)
    # training and test
    train_U = data[:int(T_train/dt)]
    train_D = data[1:int(T_train/dt)+1]

    train_U_sinusoid = data_sinusoid[:int(T_train/dt)]
    train_D_sinusoid = data_sinusoid[1:int(T_train/dt)+1]
    
    test_U = data[int(T_train/dt):int((T_train+T_test)/dt)]
    test_D = data[1+int(T_train/dt):int((T_train+T_test)/dt)+1]

    test_U_sinusoid = data_sinusoid[int(T_train/dt):int((T_train+T_test)/dt)]
    test_D_sinusoid = data_sinusoid[1+int(T_train/dt):int((T_train+T_test)/dt)]
    
    # LSM model
    lsm_lorenz = lsm.LiquidStateMachine(input_dim=3, reservoir_size=300, spectral_radius=spectral_radius, sparsity=sparsity)
    lsm_sinusoid = lsm.LiquidStateMachine(input_dim=1, reservoir_size=300, spectral_radius=spectral_radius, sparsity=sparsity)

    # get reservoir state
    reservoir_states_train = lsm_lorenz.get_states(train_U)
    reservoir_states_test = lsm_lorenz.get_states(test_U)

    lsm_lorenz.test_esp(reservoir_states_train)

    reservoir_states_train_sinusoid = lsm_sinusoid.get_states(train_U_sinusoid)
    reservoir_states_test_sinusoid = lsm_sinusoid.get_states(train_D_sinusoid)

    
    # traning read out layer with ridge regression
    ridge_reg = Ridge(alpha=1e-4)
    ridge_reg.fit(reservoir_states_train, train_D)
    
    ridge_reg_sinusoid = Ridge(alpha=5)
    ridge_reg_sinusoid.fit(reservoir_states_train_sinusoid, train_D_sinusoid)

    # predicting test data
    test_Y = ridge_reg.predict(reservoir_states_test)
    test_Y_sinusoid = ridge_reg_sinusoid.predict(reservoir_states_test_sinusoid)



    accuracy = np.abs(test_Y[:, 0] - test_U[:, 0]) < 1  # Element-wise condition
    nb_points = len(test_Y[:, 0])
    nb_true = np.sum(accuracy)  # Sum of True values (True is 1, False is 0)
    accuracy_result = nb_true / nb_points * 100  # Accuracy as a percentage
    print(f"Accuracy over 100: {accuracy_result:.2f}%")

    mse = np.mean((test_Y - test_U) ** 2)
    print(f"Mean Squared Error: {mse:.2f}")
        
    plt.figure(figsize=(14, 10))
    abscissa = 2000
    plt.subplot(3, 1, 1)
    plt.plot(test_Y[:abscissa, 0], label=f'Predicted X (Spectral Radius={spectral_radius})')
    plt.plot(test_U[:abscissa, 0], label='True X')
    plt.legend()
    plt.title(f'Time Series Prediction (Spectral Radius={spectral_radius} & Sparse={sparsity})')
    plt.xlabel('Time steps')
    plt.ylabel('X')
    plt.subplot(3, 1, 2)
    plt.plot(test_Y[:abscissa, 1], label=f'Predicted Y (Spectral Radius={spectral_radius})')
    plt.plot(test_U[:abscissa, 1], label='True Y')
    plt.legend()
    plt.xlabel('Time steps')
    plt.ylabel('Y')
    plt.subplot(3, 1, 3)
    plt.plot(test_Y[:abscissa, 2], label=f'Predicted Z (Spectral Radius={spectral_radius})')
    plt.plot(test_U[:abscissa, 2], label='True Z')
    plt.legend()
    plt.xlabel('Time steps')
    plt.ylabel('Z')

    
    
    
    plt.tight_layout()
    date_of_today = datetime.now().strftime('%Y%m%d_%H%M')
    plt.savefig(f"curves/homemade_x_y_z/spectral_rad_{spectral_radius}_date_time_{date_of_today}.png")

    plt.figure(figsize=(14, 10))
    plt.plot(test_Y[:abscissa, 0], label=f'Predicted X (Spectral Radius={spectral_radius})')
    plt.plot(test_U[:abscissa, 0], label='True X')
    plt.legend()
    plt.xlabel('Time steps')
    plt.ylabel('X')
    plt.title(f'X Prediction (Spectral Radius={spectral_radius}, Sparse={sparsity})')
    plt.savefig(f"curves/homemade_x/spectral_rad_{spectral_radius}_sparsity_{sparsity}_date_time_{date_of_today}.png")
    plt.show()

    plt.figure(figsize=(14, 10))
    plt.plot(test_Y_sinusoid, label=f'Predicted Sinusoid (Spectral Radius={spectral_radius})')
    plt.plot(test_U_sinusoid, label='True Sinusoid')
    plt.legend()
    plt.xlabel('Time steps')
    plt.ylabel('Amplitude')
    plt.title(f'Sinusoid Prediction (Spectral Radius={spectral_radius})')
    plt.show()

