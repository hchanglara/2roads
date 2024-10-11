from optimization import dp_minimization, step_minimization
import numpy as np
import matplotlib.pyplot as plt

N = 1000
T = 10
m1, m2 = 1, 2
epsilon, gamma = 0.1, 2

j1 = m1*np.ones(T+1)
j2 = m2*np.ones(T+1)
j1[T], j2[T] = 0, 0
#memo = {}

i = 0
h_prev = float('inf')
h_curr = 0

while abs(h_prev - h_curr) > 1e-5 and i < N:
    h_prev = h_curr

    for t in range(1, T):
        j1, j2 = step_minimization(j1, j2, t, epsilon, gamma)

    K = 0.5 * np.sum(np.diff(j1)**2) + 0.5 * np.sum(np.diff(j2)**2)
    I = gamma * np.sum(np.diff(j1) * np.diff(j2))
    Fb = epsilon * np.sum(j1) + epsilon * np.sum(j2)
    h_curr = K + I + Fb

    i += 1

gamma_start = 0
gamma_end = gamma
num_plots = 16
gamma_step = (gamma_end - gamma_start) / (num_plots - 1)

gamma_values = np.linspace(gamma_end, gamma_start, num_plots)

# Initialize a 4x4 grid for plotting
fig, axs = plt.subplots(4, 4, figsize=(16, 16))
axs = axs.ravel()

for idx, gamma in enumerate(gamma_values):
    i = 0
    h_prev = float('inf')
    h_curr = 0

    while abs(h_prev - h_curr) > 1e-5 and i < N:
        for t in range(1, T):
            j1, j2 = step_minimization(j1, j2, t, epsilon, gamma)
        K = 0.5 * np.sum(np.diff(j1)**2) + 0.5 * np.sum(np.diff(j2)**2)
        I = gamma * np.sum(np.diff(j1) * np.diff(j2))
        Fb = epsilon * np.sum(j1) + epsilon * np.sum(j2)
        h_curr = K + I + Fb
        i += 1

    axs[idx].plot(range(T+1), j1, label='j1', marker='o')
    axs[idx].plot(range(T+1), j2, label='j2', marker='x')
    axs[idx].set_title(f'ε = {epsilon}, γ = {gamma:.2f}, h = {h_curr:.4f}')
    axs[idx].legend()
    axs[idx].grid(True)
    axs[idx].set_ylim(-1, m2+1)

for i in range(num_plots, len(axs)):
    axs[i].axis('off')

plt.savefig('plot_2roads.pdf', format='pdf', bbox_inches='tight')

plt.tight_layout()
plt.show()