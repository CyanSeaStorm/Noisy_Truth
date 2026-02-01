import numpy as np
import matplotlib.pyplot as plt
from ukf2 import UKF
from process_model2 import imu_process_model
import time

def hx(x):
    return x[:2]

# ---------------- File Loading ---------------- #
def load_and_merge_data(imu_path, gt_path):
    imu_raw = np.loadtxt(imu_path, comments='#')
    gt_raw = np.loadtxt(gt_path, comments='#')
    combined = []
    for r in imu_raw:
        combined.append({'t': r[0], 'type': 'IMU', 'val': [r[1], r[2], 0.0]})
    for r in gt_raw:
        combined.append({'t': r[0], 'type': 'GT', 'val': [r[1], r[2]]})
    combined.sort(key=lambda x: x['t'])
    return combined

# ---------------- Initialization ---------------- #
data_stream = load_and_merge_data('imu.txt', 'ground_truth.txt')

ukf = UKF(
    n=7, m=2, 
    fx=imu_process_model, hx=hx,
    Q=np.diag([0.01, 0.01, 0.05, 0.05, 0.01, 1e-4, 1e-4]),
    R=np.diag([0.05, 0.05])
)

# Find start time
t_start = None
for entry in data_stream:
    if entry['type'] == 'GT':
        ukf.x[0:2] = entry['val']
        t_start = entry['t']
        break

# Setup Real-time Plotting
plt.ion() # Turn on interactive mode
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("UKF Real-Time Localization", fontsize=16)

# Lists to store history for plotting
hist_t, hist_est, hist_gt = [], [], []

t_prev = t_start
last_imu = None

print(f"{'Time':<10} | {'Type':<5} | {'Est X':<8} | {'Est Y':<8} | {'GT X':<8} | {'GT Y':<8}")
print("-" * 70)

# ---------------- Real-Time Loop ---------------- #
for i, entry in enumerate(data_stream):
    if entry['t'] < t_start: continue

    t_curr = entry['t']
    dt = t_curr - t_prev
    
    if dt > 0:
        ukf.predict(dt, u=last_imu)
    
    gt_val = [np.nan, np.nan] # Placeholder if no GT this step
    if entry['type'] == 'GT':
        z = np.array(entry['val'])
        ukf.update(z)
        gt_val = entry['val']
        
        # Logging like ROS2
        print(f"{t_curr % 1000:10.3f} | [GT]  | {ukf.x[0]:8.3f} | {ukf.x[1]:8.3f} | {gt_val[0]:8.3f} | {gt_val[1]:8.3f}")
        
        # Save history for live plot
        hist_t.append(t_curr - t_start)
        hist_est.append(ukf.x[:2].copy())
        hist_gt.append(gt_val)
    else:
        last_imu = entry['val']
        # Optional: log IMU pulses
        if i % 20 == 0: # Log every 20th IMU update to avoid flooding
             print(f"{t_curr % 1000:10.3f} | [IMU] | {ukf.x[0]:8.3f} | {ukf.x[1]:8.3f} | {'-':^8} | {'-':^8}")

    # Update Plot every N frames to save CPU
    if i % 10 == 0 and len(hist_t) > 1:
        t_arr = np.array(hist_t)
        est_arr = np.array(hist_est)
        gt_arr = np.array(hist_gt)

        # Clear and Redraw
        for ax in axs.flat: ax.cla(); ax.grid(True)

        # 1. Trajectory
        axs[0, 0].plot(gt_arr[:, 0], gt_arr[:, 1], 'g-', label='GT')
        axs[0, 0].plot(est_arr[:, 0], est_arr[:, 1], 'r--', label='UKF')
        axs[0, 0].set_title("Trajectory"); axs[0, 0].legend()

        # 2. X vs Time
        axs[1, 0].plot(t_arr, gt_arr[:, 0], 'g')
        axs[1, 0].plot(t_arr, est_arr[:, 0], 'r--')
        axs[1, 0].set_title("X vs Time")

        # 3. Y vs Time
        axs[1, 1].plot(t_arr, gt_arr[:, 1], 'g')
        axs[1, 1].plot(t_arr, est_arr[:, 1], 'r--')
        axs[1, 1].set_title("Y vs Time")

        # 4. Error
        axs[0, 1].plot(t_arr, est_arr[:, 0] - gt_arr[:, 0], 'b', label='Err X')
        axs[0, 1].plot(t_arr, est_arr[:, 1] - gt_arr[:, 1], 'm', label='Err Y')
        axs[0, 1].set_title("Error"); axs[0, 1].legend()

        plt.pause(0.001)

    t_prev = t_curr

plt.ioff()
plt.show()
