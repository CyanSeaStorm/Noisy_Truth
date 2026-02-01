import numpy as np
import matplotlib.pyplot as plt
import sys
from ukf2 import UKF
from process_model2 import imu_process_model

def hx(x):
    return x[:2]

# ---------------- File Loading ---------------- #
def load_and_merge_data(imu_path, gt_path):
    try:
        imu_raw = np.loadtxt(imu_path, comments='#')
        gt_raw = np.loadtxt(gt_path, comments='#')
    except Exception as e:
        print(f"Error: {e}")
        return []

    combined = []
    for r in imu_raw:
        combined.append({'t': r[0], 'type': 'IMU', 'val': [r[1], r[2], 0.0]})
    for r in gt_raw:
        combined.append({'t': r[0], 'type': 'GT', 'val': [r[1], r[2]]})
    
    combined.sort(key=lambda x: x['t'])
    return combined

# ---------------- Initialization ---------------- #
data_stream = load_and_merge_data('imu.txt', 'ground_truth.txt')
ukf = UKF(n=7, m=2, fx=imu_process_model, hx=hx,
          Q=np.diag([0.01, 0.01, 0.05, 0.05, 0.01, 1e-4, 1e-4]),
          R=np.diag([0.05, 0.05]))

for entry in data_stream:
    if entry['type'] == 'GT':
        ukf.x[0:2] = entry['val']
        t_start = entry['t']
        break

# ---------------- High-Speed Plot Setup with Units ---------------- #
plt.ion()
fig, axs = plt.subplots(2, 2, figsize=(13, 9))
fig.canvas.manager.set_window_title('High-Speed UKF Fusion (SI Units)')

# Plot 1: Trajectory
line_gt_traj, = axs[0, 0].plot([], [], 'g-', label='GT', lw=1.5)
line_est_traj, = axs[0, 0].plot([], [], 'r--', label='UKF', lw=1.5)
axs[0, 0].set_title("2D Trajectory")
axs[0, 0].set_xlabel("X Position [m]")
axs[0, 0].set_ylabel("Y Position [m]")

# Plot 2: Error
line_err_x, = axs[0, 1].plot([], [], 'b', label='Err X', lw=1)
line_err_y, = axs[0, 1].plot([], [], 'm', label='Err Y', lw=1)
axs[0, 1].set_title("Estimation Error")
axs[0, 1].set_xlabel("Time [s]")
axs[0, 1].set_ylabel("Error [m]")

# Plot 3: X vs Time
line_x_gt, = axs[1, 0].plot([], [], 'g-', lw=1)
line_x_est, = axs[1, 0].plot([], [], 'r--', lw=1)
axs[1, 0].set_title("X tracking")
axs[1, 0].set_xlabel("Time [s]")
axs[1, 0].set_ylabel("X Position [m]")

# Plot 4: Y vs Time
line_y_gt, = axs[1, 1].plot([], [], 'g-', lw=1)
line_y_est, = axs[1, 1].plot([], [], 'r--', lw=1)
axs[1, 1].set_title("Y tracking")
axs[1, 1].set_xlabel("Time [s]")
axs[1, 1].set_ylabel("Y Position [m]")

for ax in axs.flat:
    ax.grid(True)
    ax.legend(loc='upper right', fontsize='small')

hist_t, hist_est, hist_gt = [], [], []
t_prev = t_start
last_imu = None

print(f"\n{'TIMESTAMP [s]':<15} | {'STATUS':<7} | {'EST X [m]':<10} | {'EST Y [m]':<10}")
print("="*65)

# ---------------- Optimized Execution Loop ---------------- #
for i, entry in enumerate(data_stream):
    if entry['t'] < t_start: continue

    t_curr = entry['t']
    dt = t_curr - t_prev
    
    if dt > 0:
        ukf.predict(dt, u=last_imu)
    
    if entry['type'] == 'GT':
        z = np.array(entry['val'])
        ukf.update(z)
        
        # Fast sys.stdout logging
        sys.stdout.write(f"{t_curr:<15.4f} | [OK]    | {ukf.x[0]:<10.4f} | {ukf.x[1]:<10.4f}\n")
        
        hist_t.append(t_curr - t_start)
        hist_est.append(ukf.x[:2].copy())
        hist_gt.append(z.copy())
    else:
        last_imu = entry['val']

    # Update UI only every 50 frames for 5x speed boost
    if i % 50 == 0 and len(hist_t) > 1:
        t_arr = np.array(hist_t)
        est_arr = np.array(hist_est)
        gt_arr = np.array(hist_gt)

        line_gt_traj.set_data(gt_arr[:, 0], gt_arr[:, 1])
        line_est_traj.set_data(est_arr[:, 0], est_arr[:, 1])
        line_x_gt.set_data(t_arr, gt_arr[:, 0])
        line_x_est.set_data(t_arr, est_arr[:, 0])
        line_y_gt.set_data(t_arr, gt_arr[:, 1])
        line_y_est.set_data(t_arr, est_arr[:, 1])
        line_err_x.set_data(t_arr, est_arr[:, 0] - gt_arr[:, 0])
        line_err_y.set_data(t_arr, est_arr[:, 1] - gt_arr[:, 1])

        for ax in axs.flat:
            ax.relim()
            ax.autoscale_view()

        plt.pause(0.000001)

    t_prev = t_curr

print("\nProcessing complete.")
plt.ioff()
plt.show()
