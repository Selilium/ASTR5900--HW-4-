
# ============================================================
# ASTR 5900 - HW04
# Problem 2 Animation: Reduced Velocity Orbit (0.8 v_circular)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
# SIMULATION PARAMETERS
# ============================================================
dt = 0.001
t_max = 5.0
N = int(t_max / dt)

# ============================================================
# INITIAL CONDITIONS (KEY CHANGE: 0.8 v_orb)
# ============================================================
r0 = np.array([1.0, 0.0])
v0 = np.array([0.0, 0.8 * 2*np.pi])

# ============================================================
# LEAPFROG INTEGRATION (BEST FOR ORBITS)
# ============================================================
r = np.zeros((N, 2))
r[0] = r0

v_half = v0 + 0.5 * acceleration(r0) * dt

for i in range(N-1):
    r[i+1] = r[i] + v_half * dt
    a_new = acceleration(r[i+1])
    v_half = v_half + a_new * dt

# ============================================================
# SETUP FIGURE
# ============================================================
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_title("Orbit (v = 0.8 v_orb)")
ax.set_xlabel("x [AU]")
ax.set_ylabel("y [AU]")

# Sun
ax.scatter(0, 0, color='yellow', label='Sun')

# Orbit line
line, = ax.plot([], [], color='blue', label='Voyager')

# ============================================================
# ANIMATION FUNCTION
# ============================================================
def update(frame):
    line.set_data(r[:frame,0], r[:frame,1])
    return line,

# ============================================================
# CREATE ANIMATION
# ============================================================
ani = FuncAnimation(fig, update, frames=500, interval=20)

# ============================================================
# SAVE AS MP4 
# ============================================================
ani.save("../animations/orbit_v08.mp4", writer='ffmpeg', fps=30)

plt.legend()
plt.show()