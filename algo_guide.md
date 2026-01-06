# Unscented Kalman Filter (UKF) Algorithm Guide

This guide breaks down the mathematical steps of the UKF so you can implement them directly into your Python template. The steps are mapped to the specific functions in the code.

---

## 1. Initialization Phase
**Code Location:** `__init__` and `__get_sigmas`

Before the filter loop begins, we must define the "geometry" of the sigma points (how spread out they are) and calculate the weights used to average them.

### Step 1: Calculate Scaling Parameter ($\lambda$)
This parameter determines how far the sigma points are spread from the mean state.

* **Formula:**
    $$\lambda = \alpha^2 (n + k) - n$$
    * Where $n$ is the number of states (`num_states`).

### Step 2: Calculate Weights ($W$)
These weights are used later to recover the mean and covariance from the sigma points.

* **Mean Weight 0 ($W_m^0$):**
    $$\frac{\lambda}{n + \lambda}$$
* **Covariance Weight 0 ($W_c^0$):**
    $$\frac{\lambda}{n + \lambda} + (1 - \alpha^2 + \beta)$$
* **All other weights ($i = 1 \dots 2n$):**
    $$\frac{1}{2(n + \lambda)}$$

### Step 3: Generate Sigma Points ($\mathcal{X}$)
We generate $2n + 1$ points centered around the current state estimate ($x$).

* **Point 0:** The current state mean $x$.
* **Points $1$ to $n$:**
    $$x + \left(\sqrt{(n + \lambda)P}\right)_i$$
    *(Add the $i$-th column of the covariance square root)*
* **Points $n+1$ to $2n$:**
    $$x - \left(\sqrt{(n + \lambda)P}\right)_i$$
    *(Subtract the $i$-th column of the covariance square root)*

---

## 2. Prediction Phase (Time Update)
**Code Location:** `predict(timestep)`

In this step, we move the sigma points forward in time using the physics of the system.

### Step 1: Propagate Sigma Points
Pass every sigma point ($\mathcal{X}$) through the transition function (`iterate_function`).

* **Result:** A new set of transformed sigma points ($\mathcal{Y}$).

### Step 2: Calculate Predicted Mean ($\hat{x}^-$)
Compute the weighted sum of the transformed points using the `mean_weights`.

* **Formula:**
    $$\hat{x}^- = \sum_{i=0}^{2n} W_m^i \mathcal{Y}_i$$

### Step 3: Calculate Predicted Covariance ($P^-$)
Compute the weighted variance of the points around the new mean, then add process noise ($Q$).

* **Formula:**
    $$P^- = \sum_{i=0}^{2n} W_c^i (\mathcal{Y}_i - \hat{x}^-)(\mathcal{Y}_i - \hat{x}^-)^T + Q$$
    * *Note: In the code, add `timestep * self.q`.*

---

## 3. Update Phase (Measurement Correction)
**Code Location:** `update(states, data, r_matrix)`

In this step, we correct our prediction using real-world sensor data.

### Step 1: Transform to Measurement Space
We need to see what our sigma points "look like" to the sensors.

1.  **Select Sigmas:** Take the rows from the current sigma points that correspond to the measured states (e.g., if only measuring position, ignore velocity rows).
2.  **Calculate Expected Measurement Mean ($\hat{z}$):**
    $$\hat{z} = \sum_{i=0}^{2n} W_m^i \mathcal{Z}_i$$
    *(Where $\mathcal{Z}$ are the sigma points projected into measurement space)*

### Step 2: Calculate Measurement Covariance ($P_{zz}$)
Calculate the variance of the measurement sigma points, then add measurement noise ($R$).

* **Formula:**
    $$P_{zz} = \sum_{i=0}^{2n} W_c^i (\mathcal{Z}_i - \hat{z})(\mathcal{Z}_i - \hat{z})^T + R$$

### Step 3: Calculate Cross-Covariance ($P_{xz}$)
Calculate the correlation between the *state* sigma points ($\mathcal{X}$) and the *measurement* sigma points ($\mathcal{Z}$).

* **Formula:**
    $$P_{xz} = \sum_{i=0}^{2n} W_c^i (\mathcal{X}_i - \hat{x}^-)(\mathcal{Z}_i - \hat{z})^T$$

### Step 4: Calculate Kalman Gain ($K$)
Determine how much we should trust the new measurement versus our prediction.

* **Formula:**
    $$K = P_{xz} P_{zz}^{-1}$$

### Step 5: Final State Update
Update the state estimate using the difference between the real data ($z_{actual}$) and our expected measurement ($\hat{z}$).

* **New State ($x$):**
    $$x = \hat{x}^- + K (z_{actual} - \hat{z})$$
* **New Covariance ($P$):**
    $$P = P^- - K P_{zz} K^T$$

---
*Reference: Wan, E. A., & Van Der Merwe, R. (2000). The unscented Kalman filter for nonlinear estimation.*