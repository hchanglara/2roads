import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize

def H(params, a1, a2, b1, b2, epsilon, gamma):
    j1, j2 = params
    I = (j1 - a1) * (j2 - a2) + (j1 - b1) * (j2 - b2)
    K1 = 0.5 * ((j1 - a1)**2 + (j1 - b1)**2) + epsilon * j1
    K2 = 0.5 * ((j2 - a2)**2 + (j2 - b2)**2) + epsilon * j2
    return gamma * I + K1 + K2

# Minimize function over the bottom
def minimize_side1(a1, a2, b1, b2, epsilon, gamma):
    def func1(j1):
        return H([j1, a2], a1, a2, b1, b2, epsilon, gamma)
    if abs(a1 - b1) < 1e-5:
        return a1, a2, H([a1, a2], a1, a2, b1, b2, epsilon, gamma)
    else:
        result = minimize_scalar(func1, bounds=[a1, b1], method='bounded')
        return result.x, a2, result.fun

# Minimize function over the right
def minimize_side2(a1, a2, b1, b2, epsilon, gamma):
    def func2(j2):
        return H([b1, j2], a1, a2, b1, b2, epsilon, gamma)
    if abs(a2 - b2) < 1e-5:
        return b1, a2, H([a1, a2], a1, a2, b1, b2, epsilon, gamma)
    else:
        result = minimize_scalar(func2, bounds=[a2, b2], method='bounded')
        return b1, result.x, result.fun

def find_global_minimizer(a1, a2, b1, b2, epsilon, gamma):
    if gamma < 1:
      bounds = [(a1, b1), (a2, b2)]
      initial_guess = [(a1 + b1) / 2, (a2 + b2) / 2]
      m = minimize(H, initial_guess, args=(a1, a2, b1, b2, epsilon, gamma), bounds=bounds)
      return m.x
    else:
      if b1 - a1 < b2 - a2:
          side1_min = minimize_side1(a1, a2, b1, b2, epsilon, gamma)
          side2_min = minimize_side2(a1, a2, b1, b2, epsilon, gamma)
          return min(side1_min, side2_min, key=lambda x: x[2])
      else:
          side1_min = minimize_side1(a2, a1, b2, b1, epsilon, gamma)
          side2_min = minimize_side2(a2, a1, b2, b1, epsilon, gamma)

          # Swap the coordinates back to the original j1, j2 format
          side1_min_corrected = (side1_min[1], side1_min[0], side1_min[2])  # (j2, j1)
          side2_min_corrected = (side2_min[1], side2_min[0], side2_min[2])  # (j2, j1)

          return min(side1_min_corrected, side2_min_corrected, key=lambda x: x[2])

def step_minimization(j1, j2, t, epsilon, gamma):
    j1n, j2n = j1, j2
    params = np.array([j1[t],j2[t]])
    minimizer = find_global_minimizer(j1[t+1], j2[t+1], j1[t-1], j2[t-1], epsilon, gamma)
    j1n[t], j2n[t] = minimizer[0], minimizer[1]
    return j1n, j2n

def dp_minimization(N, t0, t1, j1, j2, epsilon, gamma, memo):
    if gamma < 1:
        h = np.zeros(N)
        for i in range(N):
            for t in range(t0+1, t1):
                minimizer = step_minimization(j1, j2, t, epsilon, gamma)

            # Energy
            K = 0.5 * np.sum(np.diff(j1[t0:t1+1])**2) + 0.5 * np.sum(np.diff(j2[t0:t1+1])**2)
            I = gamma * np.sum(np.diff(j1[t0:t1+1]) * np.diff(j2[t0:t1+1]))
            Fb = epsilon * np.sum(j1[t0:t1+1]) + epsilon * np.sum(j2[t0:t1+1])
            h[i] = K+I+Fb
        return j1, j2, h
    else:
        m1 = j1[t0] - j1[t1]
        m2 = j2[t0] - j2[t1]
        key = (m1, m2, t1 - t0)

        if key in memo:
            j1[t0:t1+1] = memo[key][0]+j1[t1]
            j2[t0:t1+1] = memo[key][1]+j2[t1]
            return j1, j2, memo[key][2]

        if t1 - t0 == 2:
            j1, j2 = step_minimization(j1, j2, t0 + 1, epsilon, gamma)

            # Energy
            K = 0.5 * np.sum(np.diff(j1[t0:t1+1])**2) + 0.5 * np.sum(np.diff(j2[t0:t1+1])**2)
            I = gamma * np.sum(np.diff(j1[t0:t1+1]) * np.diff(j2[t0:t1+1]))
            Fb = epsilon * np.sum(j1[t0:t1+1]) + epsilon * np.sum(j2[t0:t1+1])
            h = (K+I+Fb) * np.ones(N)

            memo[key] = (j1[t0:t1+1]-j1[t1], j2[t0:t1+1]-j2[t1], h)
            return j1, j2, h
        else:
            h = np.zeros(N)
            for i in range(N):
                j1, j2, hh = dp_minimization(N, t0, t1 - 1, j1, j2, epsilon, gamma, memo)
                j1, j2, hh = dp_minimization(N, t0 + 1, t1, j1, j2, epsilon, gamma, memo)

                # Energy
                K = 0.5 * np.sum(np.diff(j1[t0:t1+1])**2) + 0.5 * np.sum(np.diff(j2[t0:t1+1])**2)
                I = gamma * np.sum(np.diff(j1[t0:t1+1]) * np.diff(j2[t0:t1+1]))
                Fb = epsilon * np.sum(j1[t0:t1+1]) + epsilon * np.sum(j2[t0:t1+1])
                h[i] = K+I+Fb

            memo[key] = (j1[t0:t1+1]-j1[t1], j2[t0:t1+1]-j2[t1], h)
            return j1, j2, h