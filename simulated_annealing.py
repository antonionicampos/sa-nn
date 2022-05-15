import os
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime


def SA(model, N, K, T0, eps):
    start = datetime.now()
    x_current = model.sample_weights()
    J_current = model.J(x_current)
    x_min, J_min = x_current, J_current
    history = np.zeros((2, N*K))

    for k in range(K):
        T = T0 / np.log2(2 + k)
        for n in range(N):
            x_hat = model.add_noise(x_current, eps)
            J_hat = model.J(x_hat)
            if np.random.uniform() < np.exp((J_current - J_hat) / T):
                x_current, J_current = x_hat, J_hat
            if J_hat < J_min:
                x_min, J_min = x_hat, J_hat
            history[0, N*k + n] = J_hat
            history[1, N*k + n] = T
    exec_time = datetime.now() - start
    return J_min, x_min, history, exec_time


def FSA(): pass


def plot_histories(histories, T0, K, N, savefig=False, dt=''):
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax2 = ax1.twinx()
    
    for col in histories.columns[1:]:
        plot1 = ax1.plot(np.arange(histories.shape[0]), histories[col], 'b-', alpha=1., label='J Mínimo')
    ax1.set_title(f'J em função das iterações, N = {N}, T0 = {T0}, K = {K}')
    ax1.set_ylabel('J(x)')
    ax1.set_xlabel('Iteração')
    ax1.grid()
    plot2 = ax2.plot(np.arange(histories.shape[0]), histories['T'], 'r-', label='Temperatura')
    ax2.set_ylabel('Temperatura')

    plots = plot1 + plot2
    labels = [p.get_label() for p in plots]
    ax1.legend(plots, labels)
    
    if savefig:
        try: os.mkdir(f".\\images\\{dt}")
        except: pass
        fig.savefig(f'.\\images\\{dt}\\history_plot_T0-{T0}_K-{K}_N-{N}.png', dpi=1000)
    else: plt.show()