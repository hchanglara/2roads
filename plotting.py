def plot_level_sets_and_minimizer(a1, a2, b1, b2, epsilon, gamma):
    j1_vals = np.linspace(a1, b1, 100)
    j2_vals = np.linspace(a2, b2, 100)
    J1, J2 = np.meshgrid(j1_vals, j2_vals)

    H_vals = np.array([[H([j1, j2], a1, a2, b1, b2, epsilon, gamma) for j1, j2 in zip(row_j1, row_j2)]
                       for row_j1, row_j2 in zip(J1, J2)])

    minimizer = find_global_minimizer(a1, a2, b1, b2, epsilon, gamma)

    plt.contour(J1, J2, H_vals, levels=30, cmap='viridis')
    plt.colorbar(label='H(j1, j2)')

    plt.scatter(minimizer[0], minimizer[1], color='red', label='Minimizer', zorder=5)

    plt.xlabel('$j_1$')
    plt.ylabel('$j_2$')
    plt.title('Level Sets of $H(j_1, j_2)$ and Minimizer')

    margin = 0.1 * (b1 - a1)
    plt.xlim(a1 - margin, b1 + margin)
    plt.ylim(a2 - margin, b2 + margin)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.legend()
    plt.show()