import numpy as np
import matplotlib.pyplot as plt
from ukf2 import UKF

# ---------------- models ---------------- #

def fx(x, dt, u):
    px, py, vx, vy, yaw = x
    px += vx * dt
    py += vy * dt
    return np.array([px, py, vx, vy, yaw])

def hx(x):
    return x[:2]

# ---------------- load data ---------------- #

def load_fake_data():
    t = np.linspace(0, 10, 1000)
    x = 0.5 * t
    y = 0.2 * t
    return t, x, y

gt_t, gt_x, gt_y = load_fake_data()

# ---------------- UKF setup ---------------- #

ukf = UKF(
    n=5,
    m=2,
    fx=fx,
    hx=hx,
    Q=np.diag([1e-4, 1e-4, 1e-3, 1e-3, 1e-5]),
    R=np.diag([1e-2, 1e-2])
)

ukf.x = np.array([gt_x[0], gt_y[0], 0.0, 0.0, 0.0])

# ---------------- run filter ---------------- #

estimates = []
errors = []
times = []

t_prev = gt_t[0]

for k in range(len(gt_t)):
    dt = gt_t[k] - t_prev
    t_prev = gt_t[k]

    ukf.predict(dt)

    z = np.array([gt_x[k], gt_y[k]])
    ukf.update(z)

    est = ukf.x.copy()
    err = est[:2] - np.array([gt_x[k], gt_y[k]])

    estimates.append(est)
    errors.append(err)
    times.append(gt_t[k])

estimates = np.array(estimates)
errors = np.array(errors)
times = np.array(times)

# ---------------- metrics ---------------- #

rmse = np.sqrt(np.mean(errors[:, 0]**2 + errors[:, 1]**2))
print(f"Position RMSE = {rmse:.4f} m")

# ---------------- plots ---------------- #

plt.figure(figsize=(12, 4))

# Trajectory
plt.subplot(1, 3, 1)
plt.plot(gt_x, gt_y, 'g-', label="Ground Truth")
plt.plot(estimates[:, 0], estimates[:, 1], 'r--', label="UKF")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("2D Trajectory")
plt.legend()
plt.grid()

# X error
plt.subplot(1, 3, 2)
plt.plot(times, errors[:, 0], 'b')
plt.xlabel("Time [s]")
plt.ylabel("X Error [m]")
plt.title("X Estimation Error")
plt.grid()

# Y error
plt.subplot(1, 3, 3)
plt.plot(times, errors[:, 1], 'm')
plt.xlabel("Time [s]")
plt.ylabel("Y Error [m]")
plt.title("Y Estimation Error")
plt.grid()

plt.tight_layout()
plt.show()

