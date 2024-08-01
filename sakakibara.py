import numpy as np
import matplotlib.pyplot as plt
import nest
from scipy.integrate import solve_ivp
import nest.raster_plot
from sklearn.linear_model import Ridge
'''
lsm with wolfgang
'''
np.random.seed(seed=42)
nest.ResetKernel()
class LSM:
    def __init__(self):
        self.nodes_E, self.nodes_I = self.create_iaf_psc_exp()
        self.connect_tsodyks(self.nodes_E, self.nodes_I)
    def create_iaf_psc_exp(self, n_E = 1000, n_I = 250):
        nodes = nest.Create('iaf_psc_exp', n_E + n_I,
                            {'C_m': 30.0,  # 1.0,
                            'I_e': 14.5,
                            'tau_m': 30.0,  # Membrane time constant in ms
                            'E_L': 0.0,
                            'V_th': 15.0,  # Spike threshold in mV
                            'tau_syn_ex': 3.0,
                            'tau_syn_in': 2.0,
                            'V_reset': 13.8})
        return nodes[:n_E], nodes[n_E:]
    def connect_tsodyks(self, nodes_E, nodes_I):
        f0 = 10.0
        n_syn_exc = 10  # number of excitatory synapses per neuron
        n_syn_inh = 5  # number of inhibitory synapses per neuron
        w_scale = 10.0
        J_EE = w_scale * 5.0  # strength of E->E synapses [pA]
        J_EI = w_scale * 25.0  # strength of E->I synapses [pA]
        J_IE = w_scale * -20.0  # strength of inhibitory synapses [pA]
        J_II = w_scale * -20.0  # strength of inhibitory synapses [pA]
        def get_u_0(U, D, F):
            return U / (1 - (1 - U) * np.exp(-1 / (f0 * F)))
        def get_x_0(U, D, F):
            return (1 - np.exp(-1 / (f0 * D))) / (1 - (1 - get_u_0(U, D, F)) * np.exp(-1 / (f0 * D)))
        def gen_syn_param(tau_psc, tau_fac, tau_rec, U):
            return {"tau_psc": tau_psc,
                    "tau_fac": tau_fac,  # facilitation time constant in ms
                    "tau_rec": tau_rec,  # recovery time constant in ms
                    "U": U,  # utilization
                    "u": get_u_0(U, tau_rec, tau_fac),
                    "x": get_x_0(U, tau_rec, tau_fac),
                    }
        def generate_delay_normal_clipped(mu=10., sigma=20., low=3., high=200.):
            delay = np.random.normal(mu, sigma)
            delay = max(min(delay, high), low)
            return delay
        def connect(src, trg, J, n_syn, syn_param):
            nest.Connect(src, trg,
                     {'rule': 'fixed_indegree', 'indegree': n_syn},
                     dict({'model': 'tsodyks_synapse', 'delay': generate_delay_normal_clipped(),
                           'weight': {"distribution": "normal_clipped", "mu": J, "sigma": 0.7 * abs(J),
                                      "low" if J >= 0 else "high": 0.
                           }},
                          **syn_param))
        connect(nodes_E, nodes_E, J_EE, n_syn_exc, gen_syn_param(tau_psc=2.0, tau_fac=1.0, tau_rec=813., U=0.59))
        connect(nodes_E, nodes_I, J_EI, n_syn_exc, gen_syn_param(tau_psc=2.0, tau_fac=1790.0, tau_rec=399., U=0.049))
        connect(nodes_I, nodes_E, J_IE, n_syn_inh, gen_syn_param(tau_psc=2.0, tau_fac=376.0, tau_rec=45., U=0.016))
        connect(nodes_I, nodes_I, J_II, n_syn_inh, gen_syn_param(tau_psc=2.0, tau_fac=21.0, tau_rec=706., U=0.25))
    def inject_waveform(self, amplitudes:list, neuron_target, w_min=2.5 * 10 * 5.0,
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
                     "weight": {'distribution': 'uniform',
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
    def get_response(self, sim_time):
        # スパイクデータの取得
        sr = nest.Create('spike_detector')
        nest.Connect(self.nodes_E, sr)
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
class Output:
    def __init__(self) -> None:
        pass
# Lorenz方程式の定義
def lorenz(t, X, sigma, rho, beta):
    x, y, z = X
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]
if __name__ == '__main__':
    nest.SetKernelStatus({'print_time': True, 'local_num_threads': 11})
    # データ長
    T = 18000  # 訓練用
    T_test = 2000  # 検証用
    sim_time = T + T_test
    pred_period = 20
    # 初期条件
    X0 = [1.0, 1.0, 1.0]
    # パラメータの設定
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    # 時間の設定
    t_span = (1, 100)
    t_eval = np.linspace(1, 100, T + T_test)
    # 数値解の計算
    solution = solve_ivp(lorenz, t_span, X0, args=(sigma, rho, beta), t_eval=t_eval, method='RK45')
    # データの抽出
    x, y, z = solution.y
    # 正規化関数の定義
    def normalize(data):
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)
    # 正規化
    x_normalized = normalize(x)
    y_normalized = normalize(y)
    z_normalized = normalize(z)
    # # lorenzのプロット
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(solution.y[0], solution.y[1], solution.y[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(3, 1, 1)
    plt.plot(t_eval, x_normalized, label='X')
    plt.legend()
    ax1 = fig.add_subplot(3, 1, 2)
    plt.plot(t_eval, y_normalized, label='Y')
    plt.legend()
    ax1 = fig.add_subplot(3, 1, 3)
    plt.plot(t_eval, z_normalized, label='Z')
    plt.legend()
    # シミュレート
    model = LSM()
    model.inject_waveform(x_normalized,  model.nodes_E)
    spikes_binary = model.get_response(T + T_test)
    print(spikes_binary.shape)
    print('spikes_times', spikes_binary[1][:50])
    train_data = spikes_binary[:, :T-pred_period].T
    train_targets = x_normalized[pred_period:T]
    test_data = spikes_binary[:, T:sim_time - pred_period].T
    test_targets = x_normalized[T+pred_period:]
   

    # モデルの評価
    readout = Ridge(alpha=5.0)
    readout.fit(train_data, train_targets)
    predictions_train = readout.predict(train_data)
    predictions_test = readout.predict(test_data)

    print(predictions_test.shape)
    # train_Y = Ro.train(spikes_binary[:,:T].T, labels=x_normalized[:T], epochs=500)
    # pred_Y = Ro.predict(spikes_binary[:,T:].T)
    # print(pred_Y)
    # 平均二乗誤差 (MSE) を計算します
    # mse = np.mean((pred_Y - u[T:]) ** 2)
    # # 平方平均二乗誤差 (RMSE) を計算します
    # rmse = np.sqrt(mse)
    # # 平均絶対誤差 (MAE) を計算します
    # mae = np.mean(np.abs(pred_Y- u[T:]))
    # # 相関係数を計算します
    # correlation_coefficient = np.corrcoef(pred_Y, u[T:])[0, 1]
    # # 結果を表示します
    # print("平均二乗誤差 (MSE):", mse)
    # print("平方平均二乗誤差 (RMSE):", rmse)
    # print("平均絶対誤差 (MAE):", mae)
    # print("相関係数:", correlation_coefficient)
    # グラフ表示
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(t_eval[T+pred_period:], x_normalized[T+pred_period:], linestyle=':')
    ax1.plot(t_eval[T+pred_period:], predictions_test, linestyle='--')
    plt.show()