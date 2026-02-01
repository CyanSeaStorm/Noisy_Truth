import numpy as np

def imu_process_model(state, dt, imu):
    x, y, vx, vy, yaw, bax, bay = state
    
    if imu is None:
        ax_b, ay_b, wz = 0.0, 0.0, 0.0
    else:
        ax_b, ay_b, wz = imu

    # Bias subtraction
    ax = ax_b - bax
    ay = ay_b - bay

    # Orientation
    yaw_new = yaw + wz * dt
    c, s = np.cos(yaw_new), np.sin(yaw_new)

    # World frame acceleration
    ax_w = c * ax - s * ay
    ay_w = s * ax + c * ay

    # Integration
    vx_new = vx + ax_w * dt
    vy_new = vy + ay_w * dt
    x_new = x + vx * dt + 0.5 * ax_w * dt**2
    y_new = y + vy * dt + 0.5 * ay_w * dt**2

    return np.array([x_new, y_new, vx_new, vy_new, yaw_new, bax, bay])
