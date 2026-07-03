import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from numba import njit
import sys

from dataclasses import dataclass
from functools import cached_property

m_slackliner = 89  # Mass if slackliner [kg]
N = 41             # Discretization
i_slackliner = 20  # id of pt with slackliner hanging/standing
L = 100            # Line length [m]
l_leash = 1.3      # Length of leash [m]
l_leg = 1.1        # Length of legs [m] (until harness connection point)
rho = 0.1          # Density [kg/m] (main + backup) - Joker + mamba?
# rho = 0.066      # Density [kg/m] (main + backup) - Y2K
# rho = 0.0001
kl = 130*1E3       # Spring constant times length - Joker
kl_leash = 400*1E4 # Spring constant times length - Joker
# kl = 500*1E3     # Spring constant times length - Y2K
l = L/(N-1)        # length of discretized line segment
m = l*rho          # mass of point [kg]
zeta = 0.5        # Dampening parameter for linear dampening

# Degrees of Freedom handler for ODE
@dataclass
class DoFHandler:
    N_main       : int = N
    N_slackliner : int = 1

    @cached_property
    def start_main():
        return 0

    @cached_property
    def start_slackliner():
        return 2*self.N_slackliner

    @cached_property 
    def offset_velocities():
        return 2*self.N_main + 2*self.N_slackliner

dofhandler = DoFHandler()

# Adjust tension by adding/decreasing webbing
# Add by 2m      : w = 2
# Decrease by 2m : w = -2
def add_tension(w):
    global l
    alpha = w/L
    l = l*(1+alpha)

def compute_tension(pos):
    t = np.zeros(2*N-2)

    for i in range(0,2*N-2,2):
        zi = np.array([pos[i], pos[i+1]])
        zip1 = np.array([pos[i+2], pos[i+3]])
        t[i] = tension(zi, zip1)

    return np.max(t)


def plot_rope(Z,ax = None, label = None):
    if (ax is None):
       fig, ax = plt.subplots(figsize=(16, 9))

    ax.set_xlim(-0.1 * L, 1.1 * L)
    ax.set_ylim(-0.2 * L, 0.2 * L)
    ax.set_aspect('equal')
    ax.grid(True)

    if (label is None):
        label = 'Line'


    x = Z[0:2*N:2]
    y = Z[1:2*N:2]

    ax.plot(x, y, 'o-', lw=2, ms=4, label = label)
    ax.legend()

    return ax


def animate_rope(result, skip=20):
    fig, ax = plt.subplots(figsize=(16, 9))

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
def ODE_rhs(t, Z):
    out = np.zeros(4*N)

    out[0:2*N] = Z[2*N:]

    for i in range(0,2*N,2):
        if (i < 2 or i >= 2*N-2): continue

        if (i == N-1): 
            mm = m + m_slackliner
        else:
            mm = m
        
        zi = np.array([Z[i], Z[i+1]])
        zim1 = np.array([Z[i-2], Z[i-1]])
        zip1 = np.array([Z[i+2], Z[i+3]])

        F = net_force(zim1, zi, zip1, mm)
        out[i + 2*N] = F[0]/mm   -zeta*Z[i + 2*N]  # x
        out[i+1 + 2*N] = F[1]/mm -zeta*Z[i+1 + 2*N]  # y

    return out

@njit
def static_rhs(Z):
    out = np.zeros(2*N)

    out[0] = Z[0]             # x0 = 0
    out[1] = Z[1]             # y0 = 0
    out[2*N-2] = Z[2*N-2] - L # x1 = L
    out[2*N-1] = Z[2*N-1]     # y1 = 0

    for i in range(0,2*N,2):
        if (i < 2 or i >= 2*N-2): continue

        if (i == 2*i_slackliner): 
            mm = m + m_slackliner
        else:
            mm = m
        
        zi = np.array([Z[i], Z[i+1]])
        zim1 = np.array([Z[i-2], Z[i-1]])
        zip1 = np.array([Z[i+2], Z[i+3]])

        F = net_force(zim1, zi, zip1, mm)
        out[i] = F[0]
        out[i+1] = F[1]

    return out

# Initial guess assuming that with weight of the slackliner and line in the middle
# the line will sag to yield a tension of T_kN
def get_initial_pos_from_tension(T_kN = 2):
    T_kg = 1000.0/9.82 * T_kN
    mass = m_slackliner + rho*L
    s = mass * L / (4*T_kg)
    a = -2*s/L

    x = np.linspace(0, L, N)
    y = np.maximum(x*a, (L-x)*a) 

    positions = np.column_stack((x, y)).ravel()

    return positions

def get_static_position(pos):
    sol, info, ier, mesg = fsolve(static_rhs, pos, full_output=True)
    # print("Solution:", sol)
    print("ier:", ier)
    print("Message:", mesg)
    return sol


def main():

    add_tension(0)
    print(dofhandler)
    
    pos = get_initial_pos_from_tension(T_kN = 3)

    ax = plot_rope(pos)
    plt.title(f"Initial Guess")

    pos = get_static_position(pos)
    
    plot_rope(pos, ax = ax, label = 'static pos')
    plt.title(f"Tension = {compute_tension(pos)/1000} kN")
    print(f"lowest point: {np.min(pos[1:2*N:2])}")
    print(f"weight of line: {rho*L}")
    plt.show()

    # vel = np.zeros_like(pos)
    # Z = np.concatenate((pos, vel))

    # t0 = 0
    # t1 = 50

    # result = solve_ivp(ODE_rhs, (t0,t1), Z)
    # animate_rope(result)
    # Z = result.y[:,-1]
    # plot_rope(Z)

    # t0 = 0
    # t1 = 20

    # result = solve_ivp(ODE_rhs, (t0,t1), Z)

    # print(f"lowest point: {np.min(result.y[1:2*N:2,-1])}")

    # animate_rope(result)

if __name__ == "__main__":
    main()
