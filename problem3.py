
# ============================================================
# PROBLEM 3: VOYAGER 2 GRAVITY ASSIST (WORKING VERSION)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# CONSTANTS (dimensionless units)
# ============================================================
G = 4 * np.pi**2

dt = 0.002
t_max = 12.0
N = int(t_max / dt)
time = np.linspace(0, t_max, N)

# ============================================================
# JUPITER ORBIT
# ============================================================
r_J = 5.2

def jupiter_position(t):
    omega = np.sqrt(G / r_J**3)
    theta = omega * t
    return np.array([r_J*np.cos(theta), r_J*np.sin(theta)])

# ============================================================
# VOYAGER INITIAL CONDITIONS (TUNED TO HIT JUPITER)
# ============================================================
r_v = np.zeros((N, 2))
v_v = np.zeros((N, 2))

r_v[0] = np.array([1.0, 0.0])

# tuned launch direction + speed (escape-like + angled)
v_v[0] = np.array([2.0, 6.5])

# ============================================================
# ACCELERATION (SUN + JUPITER)
# ============================================================
def acceleration(r, t):
    rJ = jupiter_position(t)

    a_sun = -G * r / np.linalg.norm(r)**3
    a_jup = -0.001 * G * (r - rJ) / np.linalg.norm(r - rJ)**3

    return a_sun + a_jup

# ============================================================
# LEAPFROG INTEGRATION
# ============================================================
v_half = v_v[0] + 0.5 * acceleration(r_v[0], 0) * dt

for i in range(N-1):
    t = time[i]

    r_v[i+1] = r_v[i] + v_half * dt
    a_new = acceleration(r_v[i+1], t)
    v_half = v_half + a_new * dt
    v_v[i+1] = v_half - 0.5 * a_new * dt

# ============================================================
# JUPITER TRAJECTORY
# ============================================================
r_j = np.array([jupiter_position(t) for t in time])

# ============================================================
# SPEED (convert to km/s)
# ============================================================
speed = np.linalg.norm(v_v, axis=1)
speed_kms = speed * 4.743  # 1 AU/yr ≈ 4.743 km/s

# ============================================================
# CLOSEST APPROACH
# ============================================================
dist = np.linalg.norm(r_v - r_j, axis=1)
min_idx = np.argmin(dist)

print("\n===== VOYAGER ANALYSIS =====")
print(f"Closest distance: {dist[min_idx]:.3f} AU")
print(f"Time of closest approach: {time[min_idx]:.2f} yr")

# ============================================================
# TRAJECTORY PLOT
# ============================================================
plt.figure(figsize=(6,6))
plt.plot(r_v[:,0], r_v[:,1], label="Voyager")
plt.plot(r_j[:,0], r_j[:,1], label="Jupiter")
plt.scatter(0,0,color='yellow',label="Sun")

plt.legend(loc='upper right')
plt.title("Voyager Gravity Assist")
plt.xlabel("x [AU]")
plt.ylabel("y [AU]")
plt.axis('equal')
plt.grid()
plt.savefig("../figures/voyager.png")
plt.show()

# ============================================================
# ANIMATION (WITH TIME + SPEED)
# ============================================================
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-6,6)
ax.set_ylim(-6,6)

line_v, = ax.plot([], [], label="Voyager")
line_j, = ax.plot([], [], label="Jupiter")
text = ax.text(-5.5, 5.0, "", fontsize=10)

def update(frame):
    line_v.set_data(r_v[:frame,0], r_v[:frame,1])
    line_j.set_data(r_j[:frame,0], r_j[:frame,1])

    t_months = time[frame] * 12
    v_now = speed_kms[frame]

    text.set_text(f"t = {t_months:.1f} months\nv = {v_now:.2f} km/s")

    return line_v, line_j, text

ani = FuncAnimation(fig, update, frames=1200, interval=10)
ani.save("../animations/voyager.gif", writer='pillow')

plt.legend()
plt.show()


