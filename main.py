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

g = np.array([0, -9.82])
m_slackliner = 89  # Mass if slackliner [kg]
N = 51             # Discretization
i_leashring = int(N/2)  # id of pt with slackliner hanging/standing
L = 50             # Line length [m]
l_leash = 1.3      # Length of leash [m]
l_leg = 1.1        # Length of legs [m] (until harness connection point)
rho = 0.055        # Density [kg/m] (main + backup) - Joker + mamba?
rho_backup = 0.050 # Density [kg/m] (main + backup) - Joker + mamba?
# rho = 0.066      # Density [kg/m] (main + backup) - Y2K
# rho = 0.0001
kl = 130*1E3       # Spring constant times length - Joker
kl_leash = 400*1E3 # Spring constant times length - Joker
# kl = 500*1E3     # Spring constant times length - Y2K
l = L/(N-1)        # length of discretized line segment
m = l*rho + (L+3)/(N-1)*rho_backup          # mass of point [kg]
zeta = 1        # Dampening parameter for linear dampening

# Degrees of Freedom handler for ODE
@dataclass
class DoFHandler:
    N_main       : int = N
    N_slackliner : int = 1

    @cached_property
    def with_slackliner(self):
        return self.N_slackliner > 0

    @cached_property
    def start_main(self):
        return 0

    @cached_property
    def start_slackliner(self):
        return 2*self.N_main

    @cached_property 
    def offset(self):
        return 2*self.N_main + 2*self.N_slackliner

    def get_position_line_and_slackliner(self, pos, walking = False):
        if walking:
            v = np.array([0, l_leg])
        else:
            v = np.array([0, -l_leash])

        print(f"pos: {pos}")

        pos_leashring = pos[2*i_leashring:2*i_leashring + 2]

        if (pos.size <= self.offset):
            pos = np.concatenate((pos, pos_leashring + v))
        elif (pos.size >= self.offset):
            pos[self.start_slackliner: self.start_slackliner+2]  = pos_leashring + v

        print(f"pos: {pos}")

        return pos

dofhandler = DoFHandler()

# Adjust tension by adding/decreasing webbing
# Add by 2m      : w = 2
# Decrease by 2m : w = -2
def add_tension(w):
    global l
    alpha = w/L
    l = l*(1+alpha)

def compute_tension_mainline(pos):
    t = np.zeros(2*N-2)

    for i in range(0,2*N-2,2):
        zi = np.array([pos[i], pos[i+1]])
        zip1 = np.array([pos[i+2], pos[i+3]])
        t[i] = tension(zi, zip1, kl, l)

    return np.max(t)


def plot_rope(Z, ax = None, label = None):
    if (ax is None):
       fig, ax = plt.subplots(figsize=(16, 9))

    ax.set_xlim(-0.05 * L, 1.05 * L)
    ax.set_ylim(-0.2 * L, 0.01 * L)
    ax.set_aspect('equal')
    ax.grid(True)

    if (label is None):
        label = 'Line'


    x = Z[0:2*N:2]
    y = Z[1:2*N:2]

    ax.plot(x, y, 'o-', lw=2, ms=4, label = label)

    if (dofhandler.with_slackliner and Z.size >= dofhandler.offset):
        ax.plot([Z[2*i_leashring],Z[dofhandler.start_slackliner]],
                [Z[2*i_leashring+1], Z[dofhandler.start_slackliner+1]],
                'o-', lw = 2, ms = 4, label = 'Leash')


    ax.legend()

    return ax


def animate_rope(result, skip=20):
    fig, ax = plt.subplots(figsize=(16, 9))

    ax.set_xlim(-0.1 * L, 1.1 * L)
    ax.set_ylim(-0.2*L, 0.2*L)
    ax.set_aspect('equal')
    ax.grid(True)

    line, = ax.plot([], [], 'o-', lw=2, ms=4)
    line_slackliner, = ax.plot([], [], 'o-', lw=2, ms=4)

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

        xs = [Z[2*i_leashring], Z[dofhandler.start_slackliner]]
        ys = [Z[2*i_leashring+1], Z[dofhandler.start_slackliner+1]]
        line_slackliner.set_data(xs, ys)
        ax.set_title(f"t = {t:.3f} s")
        fig.canvas.draw_idle()
        plt.pause(0.00001)

    plt.show()


# Tension of section
# @njit
def tension(zi, zip1, kl, l):
    dist = np.linalg.norm(zip1 - zi)
    e = dist - l

    if (e > 0):
        beta = e / l
        return kl * beta

    if (e <= 0):
        return 0

    return 0

# @njit
def tension_force(z1, z2, kl, l):
    v = (z1 - z2)/np.linalg.norm(z1 - z2)
    return tension(z1, z2, kl, l)*v

# @njit
def net_force_mainline(zim1, zi, zip1, mm):

    Fim1 = tension_force(zim1, zi, kl, l)
    Fip1 = tension_force(zip1, zi, kl, l)

    F = mm*g + Fim1 + Fip1
    return F

# @njit
def ODE_rhs(t, Z):
    out = np.zeros(2*dofhandler.offset)

    out[0:dofhandler.offset] = Z[dofhandler.offset:]

    # Main line equations
    for i in range(0,2*N,2):
        if (i < 2 or i >= 2*N-2): continue

        zi = np.array([Z[i], Z[i+1]])
        zim1 = np.array([Z[i-2], Z[i-1]])
        zip1 = np.array([Z[i+2], Z[i+3]])

        F = net_force_mainline(zim1, zi, zip1, m)

        if (dofhandler.with_slackliner and i == 2*i_leashring):
            zslackliner = Z[dofhandler.start_slackliner: dofhandler.start_slackliner+2]
            # print(f"F: {F}")
            F = F + tension_force(zslackliner, zi, kl_leash, l_leash)
            # print(f"i: {i}")
            # print(f"Fadded: {tension_force(zslackliner, zi, kl_leash, l_leash)}")

        out[i + dofhandler.offset] = F[0]/m   -zeta/m*Z[i + dofhandler.offset]  # x
        out[i+1 + dofhandler.offset] = F[1]/m -zeta/m*Z[i+1 + dofhandler.offset]  # y

    # Equation for slackliner
    if (dofhandler.with_slackliner):
        i = dofhandler.start_slackliner
        zslackliner = Z[i: i+2]
        zleashring = Z[2*i_leashring:2*i_leashring+2]

        F = m_slackliner*g + tension_force(zleashring, zslackliner, kl_leash, l_leash)

        out[i + dofhandler.offset] = F[0]/m_slackliner   -zeta/m_slackliner*Z[i + dofhandler.offset]
        out[i+1 + dofhandler.offset] = F[1]/m_slackliner -zeta/m_slackliner*Z[i+1 + dofhandler.offset]

    return out

# @njit
def static_rhs(Z):
    out = np.zeros(2*N)

    out[0] = Z[0]             # x0 = 0
    out[1] = Z[1]             # y0 = 0
    out[2*N-2] = Z[2*N-2] - L # x1 = L
    out[2*N-1] = Z[2*N-1]     # y1 = 0

    for i in range(0,2*N,2):
        if (i < 2 or i >= 2*N-2): continue

        if (i == 2*i_leashring): # Add slackliner where he is hanging from leashring
            mm = m + m_slackliner
        else:
            mm = m
        
        zi = np.array([Z[i], Z[i+1]])
        zim1 = np.array([Z[i-2], Z[i-1]])
        zip1 = np.array([Z[i+2], Z[i+3]])

        F = net_force_mainline(zim1, zi, zip1, mm)
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

    add_tension(-1)
    print(dofhandler)
    
    pos = get_initial_pos_from_tension(T_kN = 3)
    pos = get_static_position(pos)

    print(f"possize: {pos.size}")
    pos = dofhandler.get_position_line_and_slackliner(pos, walking = True)
    print(f"possize: {pos.size}")
    
    plot_rope(pos, label = 'static pos')
    plt.title(f"Tension = {compute_tension_mainline(pos)/1000} kN")
    print(f"lowest point: {np.min(pos[1:2*N:2])}")
    print(f"weight of line: {m*(N-1)}")
    # plt.show()

    vel = np.zeros_like(pos)
    Z = np.concatenate((pos, vel))

    # print("Z:")
    # print(Z)
    out = ODE_rhs(0,Z)
    # print("out:")
    # print(out)

    t0 = 0
    t1 = 10

    result = solve_ivp(ODE_rhs, (t0,t1), Z, rtol = 1E-4, atol = 1E-5)
    print(f"lowest point: {np.min(result.y[1:2*N:2])}")
    animate_rope(result)

if __name__ == "__main__":
    main()
