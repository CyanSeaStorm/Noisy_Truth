import numpy as np

def imu_process_model(state, dt, imu):
    x, y, vx, vy, yaw, bax, bay = state
    ax_b, ay_b, wz = imu

    # Subtract estimated biases
    ax = ax_b - bax
    ay = ay_b - bay

    # Update orientation
    yaw_new = yaw + wz * dt
    c, s = np.cos(yaw_new), np.sin(yaw_new)

    # Transform body acceleration to world frame
    ax_w = c * ax - s * ay
    ay_w = s * ax + c * ay

    # Integrate for velocity and position
    vx_new = vx + ax_w * dt
    vy_new = vy + ay_w * dt
    x_new = x + vx * dt + 0.5 * ax_w * dt**2
    y_new = y + vy * dt + 0.5 * ay_w * dt**2

    return np.array([x_new, y_new, vx_new, vy_new, yaw_new, bax, bay])