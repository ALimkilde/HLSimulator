import numpy as np
import math
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from numba import njit
import sys
from dataclasses import dataclass
from functools import cached_property
from tqdm import tqdm

detect_collision = True
break_mainline = True

g = np.array([0, -9.82])
m_slackliner = 89  # Mass if slackliner [kg]
N = 101             # Discretization
i_leashring = int(N/2)  # id of pt with slackliner hanging/standing
L = 60             # Line length [m]
L_backup = L + 3.5 - 1.25
l_leash = 1.3      # Length of leash [m]
l_leg = 1.1        # Length of legs [m] (until harness connection point)
rho = 0.055        # Density [kg/m] (main) - Joker 
rho_backup = 0.050 # Density [kg/m] (main) - Mamba
# rho = 0.033      # Density [kg/m] (main) - Y2K
# rho_backup = 0.033      # Density [kg/m] (backup) - Y2K
# rho = 0.0001
kl = 139*1E3       # Spring constant times length - Joker
kl_backup = 139*1E3       # Spring constant times length - Solid
kl_leash = 400*1E3 # Spring constant times length - Leash
# kl = 500*1E3     # Spring constant times length - Y2K
l = L/(N-1)        # length of discretized line segment
l_backup = L_backup/(N-1)        # length of discretized line segment
m =  ( L*rho + L_backup*rho_backup)/(N-2) # mass of point [kg]
zeta = 0.0055        # Dampening parameter for linear dampening
c = zeta *2*math.sqrt(m*kl/l)
cslack = 0.0 #zeta *2*math.sqrt(m*kl/(l*m_slackliner))

# ODE setting
t0 = 0
t1 = 2

# progress bar
pbar = tqdm(total=t1 - t0, unit = "sim s", unit_scale=False)
last_t = t0
last_update = t0
update_every = 0.01  # simulated seconds

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

        pos_leashring = pos[2*i_leashring:2*i_leashring + 2]

        if (pos.size <= self.offset):
            pos = np.concatenate((pos, pos_leashring + v))
        elif (pos.size >= self.offset):
            pos[self.start_slackliner: self.start_slackliner+2]  = pos_leashring + v

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


def animate_rope(result, pp, skip=500, keep_initial_in_background = True):
    fig, ax = plt.subplots(figsize=(16, 9))

    ax.set_xlim(-0.05 * L, 1.05 * L)
    ax.set_ylim(-0.2 * L, 0.01 * L)
    ax.set_aspect('equal')
    ax.grid(True)

    line, = ax.plot([], [], 'o-', lw=2, ms=4)
    line_slackliner, = ax.plot([], [], 'o-', lw=2, ms=4)

    plt.show(block=False)

    start_wall = time.perf_counter()
    start_sim = result["t"][0]

    f_w = pp["f_webbing"]
    f_a1 = pp["f_anchor1"]
    f_a2 = pp["f_anchor2"]
    f_leash = pp["f_leash"]

    for i in range(0, len(result["t"]), skip):
        t = result["t"][i]

        # Wait until wall clock matches simulation time
        while time.perf_counter() - start_wall < (t - start_sim):
            plt.pause(0.00001)

        Z = result["y"][:, i]

        x = Z[0:2*N:2]
        y = Z[1:2*N:2]


        line.set_data(x, y)
        if (pp["backup_activated"][i]):
            line.set_color('red')
        else:
            line.set_color('blue')

        xs = [Z[2*i_leashring], Z[dofhandler.start_slackliner]]
        ys = [Z[2*i_leashring+1], Z[dofhandler.start_slackliner+1]]
        line_slackliner.set_data(xs, ys)

        if (i == 0 and keep_initial_in_background):
            plt.plot(x,y,'--')
            plt.plot(xs,ys,'--')

        ax.set_title(f"t = {t:.1f}s, F_w = {f_w[i]/1000:.1f}kN, F_l = {f_leash[i]/1000:.1f}kN")
        fig.canvas.draw_idle()
        plt.pause(0.00001)

    plt.show()

def post_process(result, skip = 1):

    #max force in
    f_webbing = np.empty_like(result["t"])
    f_anchor1 = np.empty_like(result["t"])
    f_anchor2 = np.empty_like(result["t"])
    f_leash = np.empty_like(result["t"])
    backup_activated = np.empty_like(result["t"], dtype = bool)

    # lowest point
    lp_start_webbing = np.min(result["y"][0:2*N,0])
    lp_webbing = 0


    for i in tqdm(range(0, len(result["t"]), skip), desc = "Post processing:"):

        lp_webbing = min(np.min(result["y"][0:2*N,i]), lp_webbing)

        Z = result["y"][:,i]

        pos = Z[:2*N].reshape(N, 2)
        vel = Z[dofhandler.offset:dofhandler.offset+2*N].reshape(N, 2)

        # vectors to previous and next nodes
        d_prev = pos[:-2,:] - pos[1:-1,:]
        dist_prev = np.linalg.norm(d_prev, axis=1)
        if (break_mainline):
            kl_beta_prev = kl_backup * np.maximum(dist_prev - l_backup, 0.0) / l_backup
        else:
            kl_beta_prev = kl * np.maximum(dist_prev - l, 0.0) / l + kl_backup * np.maximum(dist_prev - l_backup, 0.0) / l_backup

        F_mag_prev = kl_beta_prev[:] 

        f_webbing[i] = np.max(F_mag_prev)
        f_anchor1[i] = F_mag_prev[0]
        f_anchor2[i] = F_mag_prev[-1]
        backup_activated[i] = np.any(np.maximum(0.0, dist_prev - l_backup))

        if (dofhandler.with_slackliner):
            jj = dofhandler.start_slackliner
            zslackliner = Z[jj: jj+2]
            zleashring = Z[2*i_leashring:2*i_leashring+2]

            f_leash[i] = tension(zleashring, zslackliner, kl_leash, l_leash)


    print(
            f"Max Forces:\n"
            f"  Webbing:  {np.max(f_webbing):.2f}\n"
            f"  Anchor 1: {np.max(f_anchor1):.2f}\n"
            f"  Anchor 2: {np.max(f_anchor2):.2f}\n"
            f"  Leash:    {np.max(f_leash):.2f}\n\n"
            f"Lowest Points:\n"
            f"  Start Webbing: {lp_start_webbing:.2f}\n"
            f"  Webbing:       {lp_webbing:.2f}\n"
        f"Backup activated : {np.any(backup_activated)}"
    )

    return {
            "f_webbing": f_webbing,
            "f_anchor1": f_anchor1,
            "f_anchor2": f_anchor2,
            "f_leash": f_leash,
            "backup_activated": backup_activated,
            }

def leash_event(t, Z):
    if not dofhandler.with_slackliner:
        return 1.0  # never trigger

    z_ring = Z[2*i_leashring:2*i_leashring + 2]

    i = dofhandler.start_slackliner
    z_slack = Z[i:i+2]

    dist = np.linalg.norm(z_slack - z_ring) 

    # v_ring = Z[2*i_leashring + dofhandler.offset:2*i_leashring + dofhandler.offset + 2]
    # v_slack = Z[i + dofhandler.offset:i + dofhandler.offset + 2]
    # velocity_innerprod = np.dot(v_ring, v_slack)
    # if (velocity_innerprod > 0):
    #     return 1.0

    return dist - l_leash

def apply_collision(Z):
    """
    Applies an inelastic collision between the slackliner
    and the leash ring.
    """

    Z = Z.copy()

    ring = i_leashring

    # ring velocity
    i_ring = dofhandler.offset + 2*ring
    v_ring = Z[i_ring:i_ring+2]

    # slackliner velocity
    i_slack = dofhandler.start_slackliner + dofhandler.offset
    v_slack = Z[i_slack:i_slack+2]

    # momentum-conserving velocity
    v = (
        m*v_ring +
        m_slackliner*v_slack
    )/(m + m_slackliner)

    # Update velocities
    Z[i_ring:i_ring+2] = v
    Z[i_slack:i_slack+2] = v

    # Slightly modify leash position to avoid retriggering event
    Z[2*ring+1] = Z[2*ring + 1] + 1E-12

    return Z

leash_event.terminal = True
leash_event.direction = 1   # only detect slack -> taut

def ODE_rhs_vectorized(t, Z):
    global last_t, last_update

    if t > last_t:
        last_t = t

    if last_t - last_update >= update_every:
        pbar.update(last_t - last_update)
        last_update = last_t

    out = np.zeros_like(Z)

    n_nodes = N

    pos = Z[:2*n_nodes].reshape(n_nodes, 2)
    vel = Z[dofhandler.offset:dofhandler.offset+2*n_nodes].reshape(n_nodes, 2)

    out[:dofhandler.offset] = Z[dofhandler.offset:]

    ############################################################
    # Spring forces
    ############################################################

    # vectors to previous and next nodes
    d_prev = pos[:-2,:] - pos[1:-1,:]
    d_next = pos[2:,:]  - pos[1:-1,:]

    dist_prev = np.linalg.norm(d_prev, axis=1)
    dist_next = np.linalg.norm(d_next, axis=1)

    if (break_mainline):
        kl_beta_prev = kl_backup * np.maximum(dist_prev - l_backup, 0.0) / l_backup
        kl_beta_next = kl_backup * np.maximum(dist_next - l_backup, 0.0) / l_backup
    else:
        kl_beta_prev = kl * np.maximum(dist_prev - l, 0.0) / l + kl_backup * np.maximum(dist_prev - l_backup, 0.0) / l_backup
        kl_beta_next = kl * np.maximum(dist_next - l, 0.0) / l + kl_backup * np.maximum(dist_next - l_backup, 0.0) / l_backup

    F_prev = kl_beta_prev[:, None] * d_prev / dist_prev[:, None]
    F_next = kl_beta_next[:, None] * d_next / dist_next[:, None]

    F = m * g + F_prev + F_next

    ############################################################
    # Slackliner
    ############################################################

    if dofhandler.with_slackliner:

        ring = i_leashring

        z_ring = pos[ring]
        z_slack = Z[dofhandler.start_slackliner:
                    dofhandler.start_slackliner+2]

        d = z_slack - z_ring
        dist = np.linalg.norm(d)

        if dist > l_leash:

            beta = (dist - l_leash)/l_leash
            F_leash = kl_leash * beta * d/dist

            F[ring-1] += F_leash


    ############################################################
    # Drag + acceleration
    ############################################################

    vel_norm = np.linalg.norm(vel[1:-1], axis=1)

    acc = (
        F/m
        - c*vel[1:-1]*vel_norm[:, None]
    )

    out[dofhandler.offset+2:
        dofhandler.offset+2*(n_nodes-1)] = acc.reshape(-1)

    ############################################################
    # Slackliner equation
    ############################################################

    if dofhandler.with_slackliner:

        i = dofhandler.start_slackliner

        z_slack = Z[i:i+2]
        z_ring = pos[i_leashring, :]
        vel_slack = Z[i + dofhandler.offset: i +dofhandler.offset+2]
        vel_norm = np.linalg.norm(vel_slack)

        d = z_ring - z_slack
        dist = np.linalg.norm(d)

        if dist > l_leash:
            beta = (dist-l_leash)/l_leash
            F_slack = m_slackliner*g + kl_leash*beta*d/dist
        else:
            F_slack = m_slackliner*g

        out[i+dofhandler.offset:i+dofhandler.offset+2] = (
            F_slack/m_slackliner - cslack*vel_slack*vel_norm
        )

    return out

# Tension of section
# @njit
def tension(zi, zip1, kl, l, kl_backup = 0):
    out = 0.0

    dist = np.linalg.norm(zip1 - zi)

    # main
    e = dist - l
    if (e > 0):
        beta = e / l
        out += kl * beta

    # backup
    e = dist - l_backup
    if (e > 0):
        beta = e/l_backup
        out += kl_backup * beta

    return out

# @njit
def tension_force(z1, z2, kl, l, kl_backup = 0):
    v = (z1 - z2)/np.linalg.norm(z1 - z2)
    return tension(z1, z2, kl, l, kl_backup)*v

# @njit
def net_force_mainline(zim1, zi, zip1, mm):

    Fim1 = tension_force(zim1, zi, kl, l, kl_backup)
    Fip1 = tension_force(zip1, zi, kl, l, kl_backup)

    F = mm*g + Fim1 + Fip1
    return F


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

def integrate_with_collisions(y0, t0, tf, **solve_kwargs):
    """
    Integrate until tf, applying collisions whenever `leash_event` occurs.
    """

    t_current = t0
    y_current = y0

    t_all = []
    y_all = []
    collision_times = []

    while t_current < tf:

        sol = solve_ivp(
            ODE_rhs_vectorized,
            (t_current, tf),
            y_current,
            events=leash_event,
            **solve_kwargs
        )

        # Append solution (avoid duplicating first point)
        if len(t_all) == 0:
            t_all.extend(sol.t)
            y_all.append(sol.y)
        else:
            t_all.extend(sol.t[1:])
            y_all.append(sol.y[:, 1:])

        # Finished without another collision
        if sol.status != 1:
            break

        # Collision
        t_current = sol.t_events[0][0]
        y_current = apply_collision(sol.y_events[0][0])
        collision_times.append(t_current)

        print(f"detected collision at t = {t_current:.2f}")

    return {
        "t": np.array(t_all),
        "y": np.hstack(y_all),
        "collision_times": collision_times,
    }

def main():

    add_tension(-1.25)
    print(f"c: {c}")
    
    pos = get_initial_pos_from_tension(T_kN = 0.03)
    pos = get_static_position(pos)

    pos = dofhandler.get_position_line_and_slackliner(pos, walking = True)
    
    plot_rope(pos, label = 'static pos')
    plt.title(f"Tension = {compute_tension_mainline(pos)/1000} kN")
    print(f"Tension = {compute_tension_mainline(pos)/1000} kN")
    print(f"weight of line: {m*(N-2)}")

    vel = np.zeros_like(pos)
    Z = np.concatenate((pos, vel))

    result = integrate_with_collisions(
        Z,
        t0,
        t1,
        rtol=1e-8,
        atol=1e-10,
    )

    pp = post_process(result, skip = 1)

    animate_rope(result, pp)

if __name__ == "__main__":
    main()
