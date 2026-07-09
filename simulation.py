import numpy as np
import math
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from numba import njit
from dataclasses import dataclass
from functools import cached_property
from tqdm import tqdm
import sys

from config import * # Change when we switch to segmented setups
from polyline import project_along_y, interpolate

from itertools import accumulate

def discretize_segments(segments, x):
    """
    Parameters
    ----------
    segments : list[segment]
        Consecutive line segments.
    x : array_like
        Node positions (length N).

    Returns
    -------
    kl : (N-1,) ndarray
        Main spring constant × length for each interval.
    kl_backup : (N-1,) ndarray
        Backup spring constant × length for each interval.
    l : (N-1,) ndarray
        Interval lengths.
    l_backup : (N-1,) ndarray
        Backup interval lengths (same as l).
    m : (N-2,) ndarray
        Mass associated with interior nodes.
    """
    x = np.asarray(x)

    l = np.diff(x)
    mid = x[:-1] + 0.5 * l

    # Segment boundaries
    bounds = np.array([0.0, *accumulate(s.L_main for s in segments)])

    # Segment index for each interval
    idx = np.searchsorted(bounds, mid, side="right") - 1
    idx = np.clip(idx, 0, len(segments) - 1)

    kl = np.array([segments[i].kl_main for i in idx])
    kl_backup = np.array([segments[i].kl_backup for i in idx])

    rho = np.array([segments[i].rho_main for i in idx])
    rho_backup = np.array([segments[i].rho_backup for i in idx])

    l_backup = np.array([l[j] * segments[i].L_backup/segments[i].L_main for (j,i) in enumerate(idx)])

    # Mass of each interval
    interval_mass = rho * l + rho_backup * l_backup

    # Interior node masses (half from each neighbouring interval)
    m = 0.5 * (interval_mass[:-1] + interval_mass[1:])

    break_mainline = np.array([segments[i].break_mainline for i in idx], dtype=bool)

    return kl, m, l, kl_backup, l_backup, rho, rho_backup, break_mainline, idx

g = np.array([0, -9.82])

kl, m, l, kl_backup, l_backup, rho, rho_backup, break_mainline, seg_ids = discretize_segments(segs, np.linspace(0,L,N))

c = zeta *2*np.sqrt(m[0]*kl[0]/l[0])             # dampening TODO
cslack = 0.0 #zeta *2*math.sqrt(m*kl/(l*m_slackliner))


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

        y_min = np.min(pos[1::2])
        p_slacker = np.array([x_slacker, y_min]) # TODO maybe a hack?

        if (pos.size <= self.offset):
            pos = np.concatenate((pos, p_slacker + v))
        elif (pos.size >= self.offset):
            pos[self.start_slackliner: self.start_slackliner+2]  = p_slacker + v

        return pos

dofhandler = DoFHandler()

# Adjust tension by adding/decreasing webbing
# Add by 2m      : w = 2
# Decrease by 2m : w = -2
def add_tension(w, seg_id):
    if (seg_id >= len(segs)):
        print("wrong section for tensioning")
        sys.exit()
    global l
    L_main = np.sum(l[seg_ids == seg_id])
    alpha = w/L_main
    l[seg_ids == seg_id] = l[seg_ids == seg_id]*(1+alpha)

    interval_mass = rho * l + rho_backup * l_backup

    sum_main = np.sum(interval_mass[seg_ids == seg_id])
    mass_removed = w*(segs[seg_id].rho_main + segs[seg_id].rho_backup)
    alpha = mass_removed/sum_main
    interval_mass[seg_ids == seg_id] = interval_mass[seg_ids == seg_id]*(1+alpha)

    m = 0.5 * (interval_mass[:-1] + interval_mass[1:])

def compute_tension_mainline(pos):
    t = np.zeros(2*N-2)

    for i in range(0,2*N-2,2):
        zi = np.array([pos[i], pos[i+1]])
        zip1 = np.array([pos[i+2], pos[i+3]])
        t[i] = tension(zi, zip1, kl[i], l[i])

    return np.max(t)

def post_process(result, skip = 1):

    #max force in
    f_webbing = np.empty_like(result["t"])
    f_anchor1 = np.empty_like(result["t"])
    f_anchor2 = np.empty_like(result["t"])
    f_leash = np.empty_like(result["t"])
    backup_activated = np.empty_like(result["t"], dtype = bool)
    backup_activated_segments = np.empty((N-1,len(result["t"])), dtype = bool)
    
    # Gforces in leash
    G_leash = np.zeros_like(result["t"])

    # lowest point
    lp_start_webbing = np.min(result["y"][0:2*N,0])
    lp_webbing = 0
    lp_slackliner = 0


    for i in tqdm(range(0, len(result["t"]), skip), desc = "Post processing:"):

        lp_webbing = min(np.min(result["y"][1:2*N:2,i]), lp_webbing)
        lp_slackliner = min(result["y"][dofhandler.start_slackliner + 1,i], lp_slackliner)

        Z = result["y"][:,i]

        pos = Z[:2*N].reshape(N, 2)
        vel = Z[dofhandler.offset:dofhandler.offset+2*N].reshape(N, 2)

        # vectors to previous and next nodes
        d_prev = pos[:-1,:] - pos[1:,:] 
        dist_prev = np.linalg.norm(d_prev, axis=1)

        main = kl * np.maximum(dist_prev - l, 0.0) / l
        backup = kl_backup * np.maximum(dist_prev - l_backup, 0.0) / l_backup
        
        kl_beta_prev = np.where(
            break_mainline,
            backup,
            main + backup,
        )

        F_mag_prev = kl_beta_prev[:] 

        f_webbing[i] = np.max(F_mag_prev)
        f_anchor1[i] = F_mag_prev[0]
        f_anchor2[i] = F_mag_prev[-1]
        backup_activated_segments[:,i] = np.maximum(0.0, dist_prev - l_backup)
        backup_activated[i] = np.any(backup_activated_segments[:,i])

        if (dofhandler.with_slackliner):
            jj = dofhandler.start_slackliner
            zslackliner = Z[jj: jj+2]
    
            proj, dist, _, _, _ = project_along_y(zslackliner, pos)

            f_leash[i] = tension(proj, zslackliner, kl_leash, l_leash)
            if (i > 0):
                i_prev = i - 20
                dt = result["t"][i] - result["t"][i_prev]
                vel_slack = Z[dofhandler.offset + dofhandler.start_slackliner:dofhandler.offset + dofhandler.start_slackliner + 2]
                vel_slack_prev = result["y"][dofhandler.offset + dofhandler.start_slackliner:dofhandler.offset + dofhandler.start_slackliner + 2, i_prev]
                acc = np.linalg.norm(vel_slack - vel_slack_prev) / dt          # m/s²
                G_leash[i] = np.linalg.norm(acc) / 9.82
                # if (G_leash[i] > 10):
                    # print("High G forces")


    print(
            f"\n\nWalking Forces:\n"
            f"  Webbing:  {f_webbing[0]:.2f}\n"
            f"  Anchor 1: {f_anchor1[0]:.2f}\n"
            f"  Anchor 2: {f_anchor2[0]:.2f}\n"
            f"  Leash:    {f_leash[0]:.2f}\n\n"
            f"Max Forces:\n"
            f"  Webbing:  {np.max(f_webbing):.2f}\n"
            f"  Anchor 1: {np.max(f_anchor1):.2f}\n"
            f"  Anchor 2: {np.max(f_anchor2):.2f}\n"
            f"  Leash:    {np.max(f_leash):.2f}\n\n"
            f"Gforce:    {np.max(G_leash):.2f}\n\n"
            f"Lowest Points:\n"
            f"  Start Webbing: {lp_start_webbing:.2f}\n"
            f"  Webbing:       {lp_webbing:.2f}\n"
            f"  Slackliner:    {lp_slackliner:.2f}\n\n"
        f"Backup activated : {np.any(backup_activated)}"
        f"\n\n"
    )

    result.update({
        "f_webbing": f_webbing,
        "f_anchor1": f_anchor1,
        "f_anchor2": f_anchor2,
        "f_leash": f_leash,
        "G_leash": G_leash,
        "backup_activated": backup_activated,
        "backup_activated_segments": backup_activated_segments,
        "lp_start_webbing": lp_start_webbing,
        "lp_webbing": lp_webbing,
        "lp_slackliner": lp_slackliner,
    })
    
    return result

def leash_event(t, Z):
    if (not dofhandler.with_slackliner or not detect_collision):
        return 1.0  # never trigger

    pos = Z[:2*N].reshape(N, 2)

    p_slacker = Z[dofhandler.start_slackliner:dofhandler.start_slackliner+2]

    _, dist, _, _, _ = project_along_y(p_slacker, pos)

    return dist - l_leash

def apply_collision(Z):
    """
    Applies an inelastic collision between the slackliner
    and the leash ring.
    """

    Z = Z.copy()

    pos = Z[:2*N].reshape(N, 2)
    p_slacker = Z[dofhandler.start_slackliner:dofhandler.start_slackliner+2]
    _, dist, i_prev, i_next, alpha = project_along_y(p_slacker, pos)

    # ring velocity
    i_vel_prev = dofhandler.offset + 2*i_prev
    v_prev = Z[i_vel_prev:i_vel_prev+2]

    i_vel_next = dofhandler.offset + 2*i_next
    v_next = Z[i_vel_next:i_vel_next+2]

    v_ring = interpolate(v_prev, v_next, alpha)
    m_ring = interpolate(m[i_prev-1], m[i_next-1], alpha)# TODO make function to map i_pos to i_mass

    # slackliner velocity
    i_slack = dofhandler.start_slackliner + dofhandler.offset
    v_slack = Z[i_slack:i_slack+2]

    # momentum-conserving velocity
    v = (
        m_ring*v_ring + 
        m_slackliner*v_slack
    )/(m_ring + m_slackliner)

    # Update velocities
    Z[i_vel_prev:i_vel_prev+2] = v 
    Z[i_vel_next:i_vel_next+2] = v
    Z[i_slack:i_slack+2] = v

    # Slightly modify leash position to avoid retriggering event
    Z[2*i_prev+1] += 1E-12
    Z[2*i_next+1] += 1E-12

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

    main_prev = kl[:-1] * np.maximum(dist_prev - l[:-1], 0.0) / l[:-1]
    main_next = kl[1:]  * np.maximum(dist_next - l[1:], 0.0) / l[1:]
    
    backup_prev = kl_backup[:-1] * np.maximum(dist_prev - l_backup[:-1], 0.0) / l_backup[:-1]
    backup_next = kl_backup[1:]  * np.maximum(dist_next - l_backup[1:], 0.0) / l_backup[1:]
    
    kl_beta_prev = np.where(
        break_mainline[:-1],
        backup_prev,
        main_prev + backup_prev,
    )
    
    kl_beta_next = np.where(
        break_mainline[1:],
        backup_next,
        main_next + backup_next,
    )

    F_prev = kl_beta_prev[:, None] * d_prev / dist_prev[:, None]
    F_next = kl_beta_next[:, None] * d_next / dist_next[:, None]

    F = m[:, None] * g + F_prev + F_next

    ############################################################
    # Slackliner
    ############################################################

    if dofhandler.with_slackliner:

        z_slack = Z[dofhandler.start_slackliner:
                    dofhandler.start_slackliner+2]

        proj, dist, i_prev, i_next, alpha = project_along_y(z_slack, pos)

        d = z_slack - proj

        if dist > l_leash:
            beta = (dist - l_leash)/l_leash
            F_leash = kl_leash * beta * d/dist

            F[i_prev-1] += (1-alpha)*F_leash # TODO Make a mapping from i_pos to i_mass/force
            F[i_next-1] += alpha    *F_leash
            # ====================================================


    ############################################################
    # Drag + acceleration
    ############################################################

    vel_norm = np.linalg.norm(vel[1:-1], axis=1)

    acc = (
        F/m[:, None]
        - c*vel[1:-1]*vel_norm[:, None]
    )

    out[dofhandler.offset+2:
        dofhandler.offset+2*(n_nodes-1)] = acc.reshape(-1)

    ############################################################
    # Slackliner equation
    ############################################################

    if dofhandler.with_slackliner:

        i = dofhandler.start_slackliner

        vel_slack = Z[i + dofhandler.offset: i +dofhandler.offset+2]
        vel_norm = np.linalg.norm(vel_slack)

        d = proj - z_slack
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
def tension(zi, zip1, kl, l, kl_backup = None, l_backup = None):
    out = 0.0

    dist = np.linalg.norm(zip1 - zi)

    # main
    e = dist - l
    if (e > 0):
        beta = e / l
        out += kl * beta

    # backup
    if (kl_backup is not None and l_backup is not None):
        e = dist - l_backup
        if (e > 0):
            beta = e/l_backup
            out += kl_backup * beta

    return out

# @njit
def tension_force(z1, z2, kl, l, kl_backup = None, l_backup = None):
    v = (z1 - z2)/np.linalg.norm(z1 - z2)
    return tension(z1, z2, kl, l, kl_backup, l_backup)*v

# @njit
def net_force_mainline(zim1, zi, zip1, kl, l, mm, kl_backup = None, l_backup = None):

    Fim1 = tension_force(zim1, zi, kl, l, kl_backup, l_backup)
    Fip1 = tension_force(zip1, zi, kl, l, kl_backup, l_backup)

    F = mm*g + Fim1 + Fip1
    return F


def static_rhs(Z):

    out = np.zeros_like(Z)

    pos = Z.reshape(N,2)

    # boundary conditions
    out[0] = pos[0,0]
    out[1] = pos[0,1]

    out[-2] = pos[-1,0] - L
    out[-1] = pos[-1,1]

    d_prev = pos[:-2] - pos[1:-1]
    d_next = pos[2:]  - pos[1:-1]

    dist_prev = np.linalg.norm(d_prev, axis=1)
    dist_next = np.linalg.norm(d_next, axis=1)

    beta_prev = (
        kl[:-1]*np.maximum(dist_prev-l[:-1],0)/l[:-1]
        + kl_backup[:-1]*np.maximum(dist_prev-l_backup[:-1],0)/l_backup[:-1]
    )

    beta_next = (
        kl[1:]*np.maximum(dist_next-l[1:],0)/l[1:]
        + kl_backup[1:]*np.maximum(dist_next-l_backup[1:],0)/l_backup[1:]
    )

    F_prev = beta_prev[:,None]*d_prev/dist_prev[:,None]
    F_next = beta_next[:,None]*d_next/dist_next[:,None]

    masses = m.copy()

    if dofhandler.with_slackliner:
        p_slacker = np.array([x_slacker, np.min(pos[:,1])]) # TODO maybe a hack?
        _, dist, i_prev, i_next, alpha = project_along_y(p_slacker, pos)

        masses[i_prev-1] += (1-alpha)*m_slackliner
        masses[i_next-1] += alpha    *m_slackliner

    F = masses[:,None]*g + F_prev + F_next

    out[2:-2] = F.reshape(-1)

    return out

# Initial guess assuming that with weight of the slackliner and line in the middle
# the line will sag to yield a tension of T_kN
def get_initial_pos_from_tension(T_kN = 2):
    T_kg = 1000.0/9.82 * T_kN
    mass = m_slackliner + np.sum(m)
    s = mass * L / (4*T_kg)
    a = -2*s/L

    x = np.linspace(0, L, N)
    y = np.maximum(x*a, (L-x)*a) 

    positions = np.column_stack((x, y)).ravel()

    return positions

def get_static_position(pos = None):
    if (pos is None):
        w_line = np.sum(m)
        pos = get_initial_pos_from_tension(T_kN = w_line*9.82/1000)

    sol, info, ier, mesg = fsolve(static_rhs, pos, full_output=True)

    if (ier != 1):
        print("Static solver could not converge!")
        print("ier:", ier)
        print("Message:", mesg)
        sys.exit()
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

        pbar.write(f"Leash started to see tension at t = {t_current:.2f}")

    return {
        "t": np.array(t_all),
        "y": np.hstack(y_all),
        "collision_times": collision_times,
    }

def simulate(pos = None):
    add_tension(pull_webbing, len(segs)-1)

    if (pos is None):
        pos = get_static_position()

    pos = dofhandler.get_position_line_and_slackliner(pos, walking = True)
    vel = np.zeros_like(pos)
    Z = np.concatenate((pos, vel))

    result = integrate_with_collisions(
        Z,
        t0,
        t1,
        rtol=1e-5,
        atol=1e-6,
    )

    result = post_process(result, skip = 1) # Add postprocessing to results
    return result

