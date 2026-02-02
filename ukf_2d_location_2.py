import numpy as np
import matplotlib.pyplot as plt
import sys
from ukf2 import UKF
from process_model2 import imu_process_model

def hx(x):
    return x[:2]

def load_data(imu_path, gt_path):
    try:
        imu_raw = np.loadtxt(imu_path, comments='#')
        gt_raw = np.loadtxt(gt_path, comments='#')
        combined = []
        for r in imu_raw:
            combined.append({'t': r[0], 'type': 'IMU', 'val': [r[1], r[2], 0.0]})
        for r in gt_raw:
            combined.append({'t': r[0], 'type': 'GT', 'val': [r[1], r[2]]})
        combined.sort(key=lambda x: x['t'])
        return combined
    except Exception as e:
        print(f"Error: {e}")
        return []

# ---------------- Initialization ---------------- #
data_stream = load_data('imu.txt', 'ground_truth.txt')
ukf = UKF(n=7, m=2, fx=imu_process_model, hx=hx,
          Q=np.diag([0.01, 0.01, 0.05, 0.05, 0.01, 1e-4, 1e-4]),
          R=np.diag([0.05, 0.05]))

t_start = 0
for entry in data_stream:
    if entry['type'] == 'GT':
        ukf.x[0:2] = entry['val']; t_start = entry['t']
        break

# ---------------- Hierarchy-Focused Layout ---------------- #
plt.ion()
fig = plt.figure(figsize=(15, 20)) 
# Gridspec: 6 rows total. 
# Row 0: Trajectory (Tallest)
# Row 1: Position Error
# Row 2: Position Tracking
# Row 3: Accel X
# Row 4: Accel Y
gs = fig.add_gridspec(5, 2, height_ratios=[4, 1.2, 1.2, 1.2, 1.2], hspace=0.6)

# 1. MAIN FOCUS: Trajectory
ax_traj = fig.add_subplot(gs[0, :])
line_gt_traj, = ax_traj.plot([], [], 'g-', label='Ground Truth Path', lw=3)
line_est_traj, = ax_traj.plot([], [], 'r--', label='UKF Estimated Path', lw=2)
rmse_text = ax_traj.text(0.02, 0.95, '', transform=ax_traj.transAxes, weight='bold', color='darkred', bbox=dict(facecolor='white', alpha=0.8))
ax_traj.set_title("MAIN TELEMETRY: Robot Trajectory [m]", fontsize=16, fontweight='bold')
ax_traj.set_xlabel("X-Position [m]"); ax_traj.set_ylabel("Y-Position [m]")

# 2. Position Errors
ax_ex = fig.add_subplot(gs[1, 0]); line_err_x, = ax_ex.plot([], [], 'b', label='X Error')
ax_ey = fig.add_subplot(gs[1, 1]); line_err_y, = ax_ey.plot([], [], 'm', label='Y Error')
ax_ex.set_title("X Position Error [m]"); ax_ey.set_title("Y Position Error [m]")

# 3. Position vs Time
ax_px = fig.add_subplot(gs[2, 0]); line_x_gt, = ax_px.plot([], [], 'g-', alpha=0.4); line_x_est, = ax_px.plot([], [], 'r--')
ax_py = fig.add_subplot(gs[2, 1]); line_y_gt, = ax_py.plot([], [], 'g-', alpha=0.4); line_y_est, = ax_py.plot([], [], 'r--')
ax_px.set_title("X tracking [m]"); ax_py.set_title("Y tracking [m]")

# 4. Acceleration X (Separated with labels)
ax_ax = fig.add_subplot(gs[3, :])
line_x_imu_raw, = ax_ax.plot([], [], color='orange', alpha=0.3, label='Raw Noisy IMU Acceleration (X)')
line_x_accel_clean, = ax_ax.plot([], [], color='black', lw=1.2, label='UKF Predicted Accel (Bias Removed)')
ax_ax.set_title("X-Axis Acceleration Analysis [m/s²]")

# 5. Acceleration Y (Separated with labels)
ax_ay = fig.add_subplot(gs[4, :])
line_y_imu_raw, = ax_ay.plot([], [], color='purple', alpha=0.3, label='Raw Noisy IMU Acceleration (Y)')
line_y_accel_clean, = ax_ay.plot([], [], color='black', lw=1.2, label='UKF Predicted Accel (Bias Removed)')
ax_ay.set_title("Y-Axis Acceleration Analysis [m/s²]")

for a in [ax_traj, ax_ex, ax_ey, ax_ax, ax_ay]:
    a.grid(True); a.legend(loc='upper right', fontsize='x-small')

# Buffers
h_t, h_est, h_gt, h_imu_t, h_ax_raw, h_ay_raw, h_ax_clean, h_ay_clean = [], [], [], [], [], [], [], []
t_prev = t_start; last_imu = [0, 0, 0]

print(f"\n{'TIME':<8} | {'X':<7} | {'Y':<7} | {'BiasX':<8} | {'BiasY':<8}")
print("-" * 55)

# ---------------- 10X ULTRA-SPEED LOOP ---------------- #
STRIDE = 200 

for i, entry in enumerate(data_stream):
    if entry['t'] < t_start: continue
    dt = entry['t'] - t_prev
    
    if dt > 0: 
        ukf.predict(dt, u=last_imu)
    
    if entry['type'] == 'GT':
        z = np.array(entry['val'])
        ukf.update(z)
        h_t.append(entry['t'] - t_start); h_est.append(ukf.x[:2].copy()); h_gt.append(z.copy())
        
        # Throttled terminal logging
        if len(h_t) % 20 == 0:
            sys.stdout.write(f"{entry['t']:<8.1f} | {ukf.x[0]:<7.2f} | {ukf.x[1]:<7.2f} | {ukf.x[5]:<8.4f} | {ukf.x[6]:<8.4f}\n")
    else:
        last_imu = entry['val']
        h_imu_t.append(entry['t'] - t_start); h_ax_raw.append(last_imu[0]); h_ay_raw.append(last_imu[1])
        h_ax_clean.append(last_imu[0] - ukf.x[5]); h_ay_clean.append(last_imu[1] - ukf.x[6])

    if i % STRIDE == 0 and len(h_t) > 1:
        t_arr, est_arr, gt_arr = np.array(h_t), np.array(h_est), np.array(h_gt)
        it_arr, axr_arr, ayr_arr = np.array(h_imu_t), np.array(h_ax_raw), np.array(h_ay_raw)
        axc_arr, ayc_arr = np.array(h_ax_clean), np.array(h_ay_clean)

        current_rmse = np.sqrt(np.mean((est_arr - gt_arr)**2, axis=0))
        rmse_text.set_text(f"Live RMSE [m]\nX: {current_rmse[0]:.4f}\nY: {current_rmse[1]:.4f}")

        line_gt_traj.set_data(gt_arr[:, 0], gt_arr[:, 1])
        line_est_traj.set_data(est_arr[:, 0], est_arr[:, 1])
        line_err_x.set_data(t_arr, est_arr[:, 0] - gt_arr[:, 0])
        line_err_y.set_data(t_arr, est_arr[:, 1] - gt_arr[:, 1])
        line_x_gt.set_data(t_arr, gt_arr[:, 0]); line_x_est.set_data(t_arr, est_arr[:, 0])
        line_y_gt.set_data(t_arr, gt_arr[:, 1]); line_y_est.set_data(t_arr, est_arr[:, 1])
        line_x_imu_raw.set_data(it_arr, axr_arr); line_x_accel_clean.set_data(it_arr, axc_arr)
        line_y_imu_raw.set_data(it_arr, ayr_arr); line_y_accel_clean.set_data(it_arr, ayc_arr)

        for a in [ax_traj, ax_ex, ax_ey, ax_px, ax_py, ax_ax, ax_ay]: 
            a.relim(); a.autoscale_view()
        plt.pause(0.0000001)
    
    t_prev = entry['t']

# ---------------- FINAL LOG ---------------- #
print("-" * 55)
print(">>> FINISHED PROCESS: DATA STREAM EXHAUSTED")
print("-" * 55)
final_rmse = np.sqrt(np.mean((np.array(h_est) - np.array(h_gt))**2, axis=0))
print(f"Final Report: X-RMSE={final_rmse[0]:.5f}, Y-RMSE={final_rmse[1]:.5f}")
print(f"Final Bias:   X={ukf.x[5]:.6f}, Y={ukf.x[6]:.6f}")

plt.ioff(); plt.show()
