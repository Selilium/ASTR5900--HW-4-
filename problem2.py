
# ============================================================
# ASTR 5900 - HW04
# PROBLEM 2: Reduced Velocity Orbit (v = 0.8 v_orb)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONSTANTS
# ============================================================
G = 4 * np.pi**2

# ============================================================
# ACCELERATION FUNCTION
# ============================================================
def acceleration(r):
    dist = np.linalg.norm(r)
    return -G * r / dist**3

# ============================================================
# ENERGY FUNCTION
# ============================================================
def energy(r, v):
    r_mag = np.linalg.norm(r, axis=1)
    v_mag = np.linalg.norm(v, axis=1)
    return 0.5 * v_mag**2 - G / r_mag

# ============================================================
# SIMULATION PARAMETERS
# ============================================================
dt = 0.001
t_max = 5.0
N = int(t_max / dt)

time = np.linspace(0, t_max, N)

# ============================================================
# INITIAL CONDITIONS (v = 0.8 v_orb)
# ============================================================
r0 = np.array([1.0, 0.0])
v0 = np.array([0.0, 0.8 * 2*np.pi])

# ============================================================
# EULER METHOD
# ============================================================
r_euler = np.zeros((N, 2))
v_euler = np.zeros((N, 2))

r_euler[0] = r0
v_euler[0] = v0

for i in range(N-1):
    a = acceleration(r_euler[i])
    v_euler[i+1] = v_euler[i] + a * dt
    r_euler[i+1] = r_euler[i] + v_euler[i] * dt

# ============================================================
# LEAPFROG METHOD
# ============================================================
r_lf = np.zeros((N, 2))
v_lf = np.zeros((N, 2))

r_lf[0] = r0

v_half = v0 + 0.5 * acceleration(r0) * dt

for i in range(N-1):
    r_lf[i+1] = r_lf[i] + v_half * dt
    a_new = acceleration(r_lf[i+1])
    v_half = v_half + a_new * dt
    v_lf[i+1] = v_half - 0.5 * a_new * dt

# ============================================================
# ENERGY CALCULATIONS
# ============================================================
energy_euler = energy(r_euler, v_euler)
energy_lf = energy(r_lf, v_lf)

# ============================================================
# (2a) TRAJECTORY PLOT
# ============================================================
plt.figure(figsize=(6,6))
plt.plot(r_euler[:,0], r_euler[:,1], label="Euler")
plt.plot(r_lf[:,0], r_lf[:,1], label="Leapfrog")
plt.scatter(0, 0, color='yellow', label='Sun')

plt.xlabel("x [AU]")
plt.ylabel("y [AU]")
plt.title("Orbit with v = 0.8 v_orb")
plt.legend(loc='upper right')
plt.axis('equal')
plt.grid()

plt.savefig("../figures/orbit_v08.png")
plt.show()

# ============================================================
# (2b) ENERGY - SEPARATE PLOTS
# ============================================================

# Euler energy
plt.figure()
plt.plot(time, energy_euler, color='blue', label='Euler')
plt.xlabel("Time [yr]")
plt.ylabel("Specific Energy")
plt.title("Euler Energy (v = 0.8 v_orb)")
plt.legend()
plt.grid()

plt.savefig("../figures/energy_euler_v08.png")
plt.show()

# Leapfrog energy
plt.figure()
plt.plot(time, energy_lf, color='orange', label='Leapfrog')
plt.xlabel("Time [yr]")
plt.ylabel("Specific Energy")
plt.title("Leapfrog Energy (v = 0.8 v_orb)")
plt.legend()
plt.grid()

plt.savefig("../figures/energy_leapfrog_v08.png")
plt.show()

# ============================================================
# PRINT ANALYSIS
# ============================================================
print("\n===== PROBLEM 2 ANALYSIS =====")
print("Orbit becomes elliptical due to reduced velocity.")
print("Euler method shows energy drift and instability.")
print("Leapfrog method conserves energy and remains stable.\n")