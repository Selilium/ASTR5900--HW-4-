
# ============================================================
# ASTR 5900 - HW04
# Problem 1: Earth Orbit (Euler vs Leapfrog)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# CONSTANTS (dimensionless units)
# ============================================================
G = 4 * np.pi**2   # gravitational constant

# ============================================================
# ACCELERATION FUNCTION
# ============================================================
def acceleration(r):
    dist = np.linalg.norm(r)
    return -G * r / dist**3

# ============================================================
# SIMULATION PARAMETERS
# ============================================================
dt = 0.001        # time step (yr)
t_max = 3.0       # simulate 3 years (3 orbits)
N = int(t_max / dt)

time = np.linspace(0, t_max, N)

# ============================================================
# INITIAL CONDITIONS
# ============================================================
r0 = np.array([1.0, 0.0])          # 1 AU
v0 = np.array([0.0, 2*np.pi])      # circular velocity

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
# SPEED CALCULATIONS (PART B)
# ============================================================
speed_euler = np.linalg.norm(v_euler, axis=1)
speed_lf = np.linalg.norm(v_lf, axis=1)

# ============================================================
# ENERGY CALCULATIONS (PART C)
# ============================================================
def energy(r, v):
    r_mag = np.linalg.norm(r, axis=1)
    v_mag = np.linalg.norm(v, axis=1)
    return 0.5 * v_mag**2 - G / r_mag

energy_euler = energy(r_euler, v_euler)
energy_lf = energy(r_lf, v_lf)

# ============================================================
# (a) TRAJECTORY PLOT
# ============================================================
plt.figure(figsize=(6,6))
plt.plot(r_euler[:,0], r_euler[:,1], label="Euler")
plt.plot(r_lf[:,0], r_lf[:,1], label="Leapfrog")
plt.scatter(0, 0, color='yellow', label='Sun')
plt.xlabel("x [AU]")
plt.ylabel("y [AU]")
plt.title("Earth Orbit: Euler vs Leapfrog")
plt.legend(loc='upper right')
plt.axis('equal')
plt.grid()
plt.savefig("../figures/orbit_comparison.png")
plt.show()

# ============================================================
# (b) SPEED VS TIME
# ============================================================
plt.figure()
plt.plot(time, speed_euler, label="Euler")
plt.plot(time, speed_lf, label="Leapfrog")
plt.xlabel("Time [yr]")
plt.ylabel("Speed [AU/yr]")
plt.title("Speed vs Time")
plt.legend()
plt.grid()
plt.savefig("../figures/speed_vs_time.png")
plt.show()

# ============================================================
# (c) ENERGY VS TIME
# ============================================================
plt.figure()
plt.plot(time, energy_euler, label="Euler")
plt.plot(time, energy_lf, label="Leapfrog")
plt.xlabel("Time [yr]")
plt.ylabel("Specific Energy")
plt.title("Energy vs Time")
plt.legend()
plt.grid()
plt.savefig("../figures/energy_vs_time.png")
plt.show()

# ============================================================
# (a) ANIMATION
# ============================================================
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

line_euler, = ax.plot([], [], label="Euler")
line_lf, = ax.plot([], [], label="Leapfrog")

def update(frame):
    line_euler.set_data(r_euler[:frame,0], r_euler[:frame,1])
    line_lf.set_data(r_lf[:frame,0], r_lf[:frame,1])
    return line_euler, line_lf

ani = FuncAnimation(fig, update, frames=500, interval=10)
ani.save("../animations/orbit.gif", writer='pillow')

plt.legend()
plt.show()

# ============================================================
# (d) PRINT RESULTS
# ============================================================
print("\n===== ANALYSIS =====")
print("Euler: Orbit becomes unstable (spirals outward).")
print("Leapfrog: Orbit remains stable (nearly circular).")
print("Leapfrog conserves energy much better than Euler.\n")
