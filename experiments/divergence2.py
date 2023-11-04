import numpy as np
import matplotlib.pyplot as plt

plt.style.use('tableau-colorblind10')
plt.rcParams['font.size'] = 20 # Adjust as necessary

def kf_predict(X0, P0, A, Q, B, U1):
    X10 = np.dot(A, X0) + np.dot(B, U1)
    P10 = np.dot(np.dot(A, P0), A.T) + Q
    return (X10, P10)

def kf_update(X10, P10, Z, H, R):
    V = Z - np.dot(H, X10)
    K = np.dot(np.dot(P10, H.T), np.linalg.pinv(np.dot(np.dot(H, P10), H.T) + R))
    X1 = X10 + np.dot(K, V)
    P1 = np.dot(np.eye(K.shape[0]) - np.dot(K, H), P10)
    return (X1, P1, K)

n = 25 # Increase data quantity
nx = 3
t = np.linspace(0, 5, n)
dt = t[1] - t[0]

a_true = np.ones(n) * 9.8 + np.random.normal(0, 1, size=n)
v_true = np.cumsum(a_true * dt)
x_true = np.cumsum(v_true * dt)
X_true = np.concatenate([x_true, v_true, a_true]).reshape([nx, -1])

R = np.diag([1 ** 2])
e = np.random.normal(0, 2, n)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # Adjust as necessary
axs = axs.flatten()
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
outlier_ratios = range(0, 36, 10)  # Define outlier ratios
kf_estimates = []
for idx, outlier_ratio in enumerate(outlier_ratios):
    x_obs = x_true + e
    outliers = np.random.choice(n, size=int(n * outlier_ratio / 100), replace=False)
    x_obs[outliers] += np.random.normal(0, 36, len(outliers))

    A = np.array([1, dt, 0.5 * dt ** 2,
                  0, 1, dt,
                  0, 0, 1]).reshape([nx, nx])
    B = 0
    U1 = 0

    x0 = 0
    v0 = 0
    a0 = 10.0
    X0 = np.array([x0, v0, a0]).reshape(nx, 1)

    P0 = np.diag([0 ** 2, 0 ** 2, 0.2 ** 2])
    Q = np.diag([0.0 ** 2, 0 ** 2, 1 ** 2])

    X1_np = np.copy(X0)
    P1_list = [P0]
    X10_np = np.copy(X0)
    P10_list = [P0]

    for i in range(n):
        Zi = np.array(x_obs[i]).reshape([1, 1])
        Hi = np.array([1, 0, 0]).reshape([1, nx])

        if (i == 0):
            continue
        else:
            Xi = X1_np[:, i - 1].reshape([nx, 1])

        Pi = P1_list[i - 1]
        X10, P10 = kf_predict(Xi, Pi, A, Q, B, U1)

        X10_np = np.concatenate([X10_np, X10], axis=1)
        P10_list.append(P10)

        X1, P1, K = kf_update(X10, P10, Zi, Hi, R)
        X1_np = np.concatenate([X1_np, X1], axis=1)
        P1_list.append(P1)

    kf_estimates.append(X1_np[0, :])  # Add this line after your Kalman Filter calculations

    # Plotting
    axs[idx].plot(t, x_true, 'k-', label="Truth", lw=2, alpha=0.7)
    axs[idx].scatter(t, x_obs, label="Observations", color='blue', s = 70)
    axs[idx].scatter(t[outliers], x_obs[outliers], color='red', label="Outliers", marker='*', s = 200)
    axs[idx].plot(t, kf_estimates[idx], linestyle='dashed', color=colors[idx], linewidth=2,
                  label=f'MHE')
    axs[idx].legend()
    axs[idx].set_title(f'Outlier ratio: {outlier_ratio}%')
    axs[idx].set_ylim(-20, 150)

plt.tight_layout()
plt.show()