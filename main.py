from optimization import dp_minimization, step_minimization
import numpy as np
import matplotlib.pyplot as plt

N = 10
T = 10
m1, m2 = 1, 2
epsilon, gamma = 0.1, 2

j1 = np.zeros(T+1)
j2 = np.zeros(T+1)
j1[0], j2[0] = m1, m2
memo = {}

j1, j2, h = dp_minimization(N, 0, T, j1, j2, epsilon, gamma, memo)

# Flexibility to choose gamma range and number of plots
gamma_start = 0      # Start value of gamma
gamma_end = gamma    # End value of gamma
num_plots = 16       # Number of plots (4x4 grid)
gamma_step = (gamma_end - gamma_start) / (num_plots - 1)

# Generate gamma values in decreasing order
gamma_values = np.linspace(gamma_end, gamma_start, num_plots)

# Initialize a 4x4 grid for plotting
fig, axs = plt.subplots(4, 4, figsize=(20, 20))  # Increase the figure size
axs = axs.ravel()  # Flatten the 4x4 grid for easier indexing

# Loop over the selected values of gamma in decreasing order
for idx, gamma in enumerate(gamma_values):
    h = np.zeros(N)
    for i in range(N):
        for t in range(1, T):
            j1, j2 = step_minimization(j1, j2, t, epsilon, gamma)
        K = 0.5 * np.sum(np.diff(j1)**2) + 0.5 * np.sum(np.diff(j2)**2)
        I = gamma * np.sum(np.diff(j1) * np.diff(j2))
        Fb = epsilon * np.sum(j1) + epsilon * np.sum(j2)
        h[i] = K + I + Fb

    # Plot results for the current gamma in the respective subplot
    axs[idx].plot(range(T + 1), j1, label='j1', marker='o')
    axs[idx].plot(range(T + 1), j2, label='j2', marker='x')
    axs[idx].set_title(f'ε = {epsilon}, γ = {gamma:.2f}, h = {h[-1]:.4f}')
    axs[idx].legend()
    axs[idx].grid(True)
    axs[idx].set_ylim(-1, m2 + 1)

    # Adjust x-axis labels without rotation
    axs[idx].set_xticks(range(T + 1))  # Ensure ticks are set
    axs[idx].tick_params(axis='x', pad=10)  # Increase space between labels and x-axis

# Hide any unused subplots if num_plots < 16
for i in range(num_plots, len(axs)):
    axs[i].axis('off')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Adjust spacing between subplots
plt.show()