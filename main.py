import numpy as np
from scipy.integrate import solve_ivp, RK45
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from numba import njit

N = 41       # Discretization
L = 100      # Line length [m]
rho = 0.100  # Density [kg/m] (main + backup) - Joker + mamba?
# rho = 0.066  # Density [kg/m] (main + backup) - Y2K
# rho = 0.001
kl = 130*1E3 # Spring constant times length - Joker
# kl = 500*1E3 # Spring constant times length - Y2K
l = L/(N-1)  # length of discretized line segment
m = l*rho    # mass of point
zeta = 0.5   # Dampening

# Adjust tension by adding/decreasing webbing
# Add by 2m      : w = 2
# Decrease by 2m : w = -2
def add_tension(w):
    global l
    alpha = w/L
    l = l*(1+alpha)

def animate_rope(result, skip=50):
    fig, ax = plt.subplots()

    ax.set_xlim(-0.1 * L, 1.1 * L)
    ax.set_ylim(-0.2*L, 0.2*L)
    ax.set_aspect('equal')
    ax.grid(True)

    line, = ax.plot([], [], 'o-', lw=2, ms=4)

    plt.show(block=False)

    start_wall = time.perf_counter()
    start_sim = result.t[0]

    for i in range(0, len(result.t), skip):
        t = result.t[i]

        # Wait until wall clock matches simulation time
        while time.perf_counter() - start_wall < (t - start_sim):
            plt.pause(0.00001)

        Z = result.y[:, i]

        x = Z[0:2*N:2]
        y = Z[1:2*N:2]

        line.set_data(x, y)
        ax.set_title(f"t = {t:.3f} s")
        fig.canvas.draw_idle()
        plt.pause(0.00001)

    plt.show()


# Tension of section
@njit
def tension(zi, zip1):
    dist = np.linalg.norm(zip1 - zi)
    e = dist - l

    if (e > 0):
        beta = e / l
        return kl * beta

    if (e <= 0):
        return 0

    return 0

@njit
def net_force(zim1, zi, zip1, mm):
    g = np.array([0, -9.82])

    vim1 = (zim1 - zi)/np.linalg.norm(zim1 - zi)
    vip1 = (zip1 - zi)/np.linalg.norm(zip1 - zi)

    F = mm*g + tension(zim1, zi) * vim1 + tension(zi, zip1)*vip1
    return F

@njit
def rhs(t, Z):
    out = np.zeros(4*N)

    out[0:2*N] = Z[2*N:]

    for i in range(0,2*N,2):
        if (i < 2 or i >= 2*N-2): continue

        if (i == N-1): 
            mm = 80
        else:
            mm = m
        

        zi = np.array([Z[i], Z[i+1]])
        zim1 = np.array([Z[i-2], Z[i-1]])
        zip1 = np.array([Z[i+2], Z[i+3]])

        F = net_force(zim1, zi, zip1, mm)
        out[i + 2*N] = F[0]/mm   -zeta*Z[i + 2*N]  # x
        out[i+1 + 2*N] = F[1]/mm -zeta*Z[i+1 + 2*N]  # y

    return out

def main():
    x = np.linspace(0, L, N)
    y = np.zeros_like(x) 

    add_tension(-2)
    
    positions = np.column_stack((x, y)).ravel()
    velocities = np.zeros_like(positions)
    
    Z = np.concatenate((positions, velocities))

    tmp = rhs(0,Z)

    t0 = 0
    t1 = 30

    result = solve_ivp(rhs, (t0,t1), Z)

    print(f"lowest point: {np.min(result.y[1:2*N:2,-1])}")

    animate_rope(result)

if __name__ == "__main__":
    main()
