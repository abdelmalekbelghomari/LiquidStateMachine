import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import nest
from scipy.integrate import solve_ivp
import nest.raster_plot
from lsm.nest import LSM
import pandas as pd
import os
from reservoirpy.datasets import lorenz
from datetime import datetime
'''
lsm with wolfgang
'''
np.random.seed(seed=42)
nest.ResetKernel()


def inject_waveform(amplitudes:list, neuron_target, w_min=2.5 * 10 * 5.0,
                    w_max=7.5 * 10 * 5.0, N=100):
    '''
    param amplitudes: 入力データ(時系列)
    param neuron_target: ターゲットニューロン
    param w_min: シナプス重みの最小値
    param w_max: シナプス重みの最大値
    param N: リザバー層に接続するシナプス数
    '''
    def generate_delay_normal_clipped(mu=10., sigma=20., low=3., high=200.):
        delay = np.random.normal(mu, sigma)
        delay = max(min(delay, high), low)
        return delay
    times = [float(i+1) for i in range(len(amplitudes))]
    generators = nest.Create('step_current_generator',params={"amplitude_values":amplitudes, "amplitude_times":times})
    nest.Connect(generators, neuron_target,
                {'rule': 'fixed_outdegree', 'outdegree': N},
                {"synapse_model": "static_synapse",
                    "weight":  {'distribution': 'uniform',
                             'low': 2.5 * 10 * 5.0,
                             'high': 7.5 * 10 * 5.0},
                    "delay": generate_delay_normal_clipped()})
    
def visualize_networks(self, M):
    # extract position information, transpose to list of x, y and z positions
    xpos, ypos, zpos = zip(*nest.GetPosition(M))
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection="3d")
    ax1.scatter(xpos, ypos, zpos, s=15, facecolor="b")
    plt.show()

def get_response(sim_time, nodes_E):
    # スパイクデータの取得
    sr = nest.Create('spike_detector')
    nest.Connect(nodes_E, sr)
    # SIMULATE
    nest.Simulate(sim_time)
    # スパイクデータの取得
    spike_events = nest.GetStatus(sr)[0]['events']
    # スパイクイベントから送信者IDと時刻を取得
    senders = spike_events['senders']
    times = spike_events['times']
    # シミュレーションの最大時間とニューロンの最大IDを取得
    max_time = sim_time
    max_neuron_id = senders.max()
    # スパイク行列をゼロで初期化
    spike_binary = np.zeros((max_neuron_id, max_time))
    print('max_neuron_id', max_neuron_id)
    # スパイクイベントを行列に変換
    for sender, time in zip(senders, times):
        spike_binary[sender-1, int(np.floor(time-1))] = 1
    # ラスタープロットの生成
    nest.raster_plot.from_device(sr, hist=False)
    # # グラフ表示
    # plt.show()
    return spike_binary

def normalize(data):
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)

# Lorenz方程式の定義
def generate_lorenz_data(num_steps, t_span, t_eval, dt=0.01, sigma=10.0, rho=28.0, beta=8.0/3.0):
    def lorenz(t, state):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]

    initial_state = [1.0, 1.0, 1.0]
    sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval, method='RK45')

    print("Lorenz Attractor Data Shape:", sol.y.T.shape)  # Debugging print
    return sol.y

def grid_search_ridge(train_data, train_targets, test_data, test_targets):
    parameters = {'alpha': [1e-3]}
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(train_data, train_targets)
    best_ridge = grid_search.best_estimator_
    predictions = best_ridge.predict(test_data)
    best_alpha = grid_search.best_params_['alpha']
    best_mse = np.mean((predictions - test_targets) ** 2)
    print(f"Best alpha: {best_alpha}, Best MSE: {best_mse}")
    return predictions, best_ridge

if __name__ == '__main__':
    nest.SetKernelStatus({'print_time': True, 'local_num_threads': 8})
    # データ長

    n_exc = 1000  # Number of excitatory neurons
    n_inh = 250   # Number of inhibitory neurons
    n_rec = 500   # Number of recurrent neurons to record
    sim_time = 80000
    T_train = sim_time * 0.8
    T_test = sim_time * 0.2
 
    t_span = (0, 100)
    t_eval = np.linspace(*t_span, sim_time)

    x, y, z = generate_lorenz_data(sim_time,t_span, t_eval )
    # x, y, z = lorenz(sim_time).T
    print("x shape is : ", x.shape)

    # 正規化
    scaler = StandardScaler()
    x_normalized = scaler.fit_transform(x.reshape(-1, 1)).reshape(-1)
    y_normalized = scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)
    z_normalized = scaler.fit_transform(z.reshape(-1, 1)).reshape(-1)

    # シミュレート
    spectral_radius = 1.0
    model = LSM(n_exc=n_exc, n_inh=n_inh, n_rec=n_rec, spectral_radius=spectral_radius)

    num_neurons = len(model.exc_nodes)
    neurons_per_component = num_neurons // 3

    inject_waveform(x_normalized, model.exc_nodes[:neurons_per_component])
    inject_waveform(y_normalized, model.exc_nodes[neurons_per_component:2*neurons_per_component])
    inject_waveform(z_normalized, model.exc_nodes[2*neurons_per_component:])

# inject_waveform(z_normalized, model.exc_nodes)

    spikes_binary= get_response(sim_time, model.exc_nodes)
    
    print(spikes_binary.shape)
    # print('spikes_times', spikes_binary[1][:50])
    train_data = spikes_binary[:, :int(T_test)].T
    test_data = spikes_binary[:, int(T_train):].T
    
    train_targets_x = x_normalized[:int(T_test)]
    test_targets_x = x_normalized[int(T_train):]
    train_targets_y = y_normalized[:int(T_test)]
    test_targets_y = y_normalized[int(T_train):]
    train_targets_z = z_normalized[:int(T_test)]
    test_targets_z = z_normalized[int(T_train):]

    predictions_x, best_ridge_x = grid_search_ridge(train_data, train_targets_x, test_data, test_targets_x)
    predictions_y, best_ridge_y = grid_search_ridge(train_data, train_targets_y, test_data, test_targets_y)
    predictions_z, best_ridge_z = grid_search_ridge(train_data, train_targets_z, test_data, test_targets_z)


    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    window_size = 50
    smoothed_predictions_ridge = moving_average(predictions_x, window_size)
    smoothed_test_targets = moving_average(test_targets_x, window_size)

    smoothed_predictions_ridge_z = moving_average(predictions_z, window_size)
    smoothed_test_targets_z = moving_average(test_targets_z, window_size)

    smoothed_predictions_ridge_y = moving_average(predictions_y, window_size)
    smoothed_test_targets_y = moving_average(test_targets_y, window_size)

    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(t_eval[int(T_train):], predictions_x, linestyle=':', label='Prediction X')
    ax1.plot(t_eval[int(T_train):], x_normalized[int(T_train):], label= 'Target X')
    ax1.plot(t_eval[int(T_train):int(T_train)+len(smoothed_predictions_ridge)], smoothed_predictions_ridge,'r' ,linestyle='solid', label='Prediction X Smoothed')
    plt.legend()

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(t_eval[int(T_train):], predictions_y, linestyle=':', label='Prediction Y')
    ax2.plot(t_eval[int(T_train):], y_normalized[int(T_train):], label= 'Target Y')
    ax2.plot(t_eval[int(T_train):int(T_train)+len(smoothed_predictions_ridge_y)], smoothed_predictions_ridge_y, 'r', linestyle='solid', label='Prediction Y Smoothed')
    plt.legend()

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(t_eval[int(T_train):], predictions_z, linestyle=':', label='Prediction Z')
    ax3.plot(t_eval[int(T_train):], z_normalized[int(T_train):],label= 'Target Z')
    ax3.plot(t_eval[int(T_train):int(T_train)+len(smoothed_predictions_ridge_z)], smoothed_predictions_ridge_z, 'r', linestyle='solid', label='Prediction Z Smoothed')
    plt.legend()
    fig.tight_layout()
    date_of_today = datetime.now().strftime('%Y%m%d_%H%M')
    fig.savefig(f"curves/nest/lorenz_x_y_z_sr_{spectral_radius}_{date_of_today}.png")

    fig_3d = plt.figure(figsize=(14, 10))
    ax = fig_3d.add_subplot(111, projection='3d')
    ax.plot(x_normalized,y_normalized,z_normalized, label= 'target')
    ax.plot(predictions_x, predictions_y, predictions_z, label='prediction')
    ax.plot(smoothed_predictions_ridge, smoothed_predictions_ridge_y, smoothed_predictions_ridge_z, 'r', label='smoothed prediction')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.legend()
    plt.show()