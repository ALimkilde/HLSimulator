import numpy as np
import math
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from dataclasses import dataclass
from functools import cached_property
from tqdm import tqdm
import sys

import matplotlib.pyplot as plt

from itertools import accumulate

@dataclass
class Slackliner:
    m       : float # mass in [kg]
    l_leg   : float # length from harness to feet
    l_leash : float # length of leash
    x_coor  : float # x-coordinate of slackliner

@dataclass
class Webbing:
    stretch_pct : float # Stretch percentage at the tension
    tension_kN  : float # The tension for which the stretch is given in [kN]
    weight_g_m  : float # Weight in [g/m]

    @cached_property
    def rho(self):
        return self.weight_g_m/1000 # switch to SI unit [kg]

    @cached_property
    def kl(self):
        return 100*self.tension_kN/self.stretch_pct * 1E3


@dataclass
class Segment:
    kl_main    : float # Spring constant times length
    kl_backup  : float # Spring constant times length
    rho_main   : float # Density of main
    rho_backup : float # Density of backup
    L_main     : float # Length of main piece
    L_backup   : float # Length of backup piece
    break_mainline : bool

    def __init__(
        self,
        webbing_main: webbing,
        webbing_backup: webbing,
        L_main: float,
        L_backup: float,
        break_mainline: bool,
    ):
        self.kl_main = webbing_main.kl
        self.kl_backup = webbing_backup.kl
        self.rho_main = webbing_main.rho
        self.rho_backup = webbing_backup.rho
        self.L_main = L_main
        self.L_backup = L_backup
        self.break_mainline = break_mainline


class SlacklineSpringModel:
    @cached_property
    def N(self):
        return self.N_main

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

    def init_progress_bar(self):
        # progress bar
        self.pbar = tqdm(total=self.t1 - self.t0, unit = "sim s", unit_scale=False)
        self.last_t = self.t0
        self.last_update = self.t0
        self.update_every = 0.01  # simulated seconds

        # Progress bookkeeping
        self.last_t = 0.0
        self.last_update = 0.0

    def precompute_constants(self):

        # Precompute constants
        self.k = self.kl / self.l
        self.k_backup = self.kl_backup / self.l_backup

        self.gravity_force = self.m[:, None] * self.g
        self.drag_constant = (
            0.5 * self.rho_air * self.C_D * (self.webbing_width / 2)
        )

    def preallocate_workspace(self):
        self.d_edge = np.empty((self.n_edges, 2))
        self.dist_edge = np.empty(self.n_edges)

        self.stretch = np.empty(self.n_edges)
        self.backup = np.empty(self.n_edges)
        self.beta = np.empty(self.n_edges)
        self.scale = np.empty(self.n_edges)

        self.F = np.empty((self.n_masses, 2))
        self.drag_coef = np.empty(self.n_masses)
        self.vel_norm = np.empty(self.n_masses)

    @cached_property
    def n_edges(self):
        return self.N - 1

    @cached_property
    def n_masses(self):
        return self.N - 2

    @cached_property
    def n_segs(self):
        return len(self.segs)

    def __init__(
        self,
        L,              # Length of highline spot
        N,              # Number of discretization verticies
        slackliner,     # Stats of a slackliner (weight, leg length, leash_length)
        segs,           # Segmented webbing
        T,              # Length of simulation in [s]
        pull_webbing,   # Amount of webbing to pull to add tension in [m]
    ):
        # Inputs
        self.slackliner = slackliner
        self.segs = segs 
        self.L = L
        self.t1 = T

        # Start time 
        self.t0 = 0

        # Set degrees of freedoms
        self.N_main = N
        self.N_slackliner = 1

        # Setup parameters of numerical model
        self.detect_collision = True
        #TODO : move rtol and atol here!

        # Physical parameters
        self.g = np.array([0, -9.82])   # gravitation [m/s^2]
        self.rho_air = 1.225            # [kg/m^3]
        self.C_D = 1.15                 # Drag coeff of rectangle
        self.webbing_width = 0.0254     # [m]
        self.kl_leash = 200*1E3         # Spring constant times length - Leash

        # Progress bar
        self.init_progress_bar()

        # Create discretization (mesh)
        self.spacings = self.init_spacings()
        self.discretize_segments()

        # Add tension
        self.add_tension(pull_webbing, self.n_segs-1)

        # Setup
        self.precompute_constants()
        self.preallocate_workspace()

    # Meshing routine essentially
    def init_spacings(self):
       return np.linspace(0,self.L,self.N)

    # Adjust tension by adding/decreasing webbing
    # Add by 2m      : w = -2
    # Decrease by 2m : w = 2
    def add_tension(self, w, seg_id):
        if (seg_id >= len(self.segs)):
            print("wrong section for tensioning")
            sys.exit()
        L_main = np.sum(self.l[self.seg_ids == seg_id])
        alpha = -w/L_main
        self.l[self.seg_ids == seg_id] = self.l[self.seg_ids == seg_id]*(1+alpha)
    
        # Update mass of line, as some has been removed
        self.m = self.get_mass_from_l()

    def get_position_line_and_slackliner(self, pos, walking = False):
        if walking:
            v = np.array([0, self.slackliner.l_leg])
        else:
            v = np.array([0, -self.slackliner.l_leash])

        y_min = np.min(pos[1::2])
        zslackliner = np.array([self.slackliner.x_coor, y_min]) # Initial guess
        proj, dist, _, _, _ = project_along_y(zslackliner, pos.reshape(self.N, 2) # Project to line
)
        p_slacker = proj # set position on line to projection

        if (pos.size <= self.offset):
            pos = np.concatenate((pos, p_slacker + v))
        elif (pos.size >= self.offset):
            pos[self.start_slackliner: self.start_slackliner+2]  = p_slacker + v

        return pos

    def ode_rhs(self, t, Z):

        ########################################################
        # Progress bar
        ########################################################

        if t > self.last_t:
            self.last_t = t

        if self.last_t - self.last_update >= self.update_every:
            self.pbar.update(self.last_t - self.last_update)
            self.last_update = self.last_t

        ########################################################

        out = np.zeros_like(Z)

        pos = Z[:2*self.N].reshape(self.N, 2)

        vel = Z[
            self.offset:
            self.offset + 2*self.N
        ].reshape(self.N, 2)

        out[:self.offset] = Z[self.offset:]

        ########################################################
        # Spring forces
        ########################################################

        np.subtract(pos[1:], pos[:-1], out=self.d_edge)

        np.sqrt(
            self.d_edge[:,0]**2 +
            self.d_edge[:,1]**2,
            out=self.dist_edge,
        )

        self.stretch[:] = self.dist_edge
        self.stretch -= self.l
        self.stretch.clip(min=0, out=self.stretch)

        self.beta[:] = self.k
        self.beta *= self.stretch

        self.backup[:] = self.dist_edge
        self.backup -= self.l_backup
        self.backup.clip(min=0, out=self.backup)
        self.backup *= self.k_backup

        self.beta += self.backup
        self.beta[self.break_mainline] = self.backup[self.break_mainline]

        np.divide(
            self.beta,
            self.dist_edge,
            out=self.scale,
        )

        self.F[:] = self.gravity_force

        self.F -= self.d_edge[:-1] * self.scale[:-1, None]
        self.F += self.d_edge[1:]  * self.scale[1:, None]

        ########################################################
        # Slackliner
        ########################################################

        if self.with_slackliner:

            z_slack = Z[
                self.start_slackliner:
                self.start_slackliner+2
            ]

            proj, dist, i_prev, i_next, alpha = (
                project_along_y(z_slack, pos)
            )

            d = z_slack - proj

            if dist > self.slackliner.l_leash:

                beta = (dist-self.slackliner.l_leash)/self.slackliner.l_leash

                F_leash = (
                    self.kl_leash
                    * beta
                    * d
                    / dist
                )

                self.F[i_prev-1] += (1-alpha)*F_leash
                self.F[i_next-1] += alpha*F_leash

        ########################################################
        # Drag
        ########################################################

        np.sqrt(
            vel[1:-1,0]**2 +
            vel[1:-1,1]**2,
            out=self.vel_norm,
        )

        self.drag_coef[:] = (
            self.drag_constant
            * (self.dist_edge[:-1] + self.dist_edge[1:])
        )

        acc = (
            self.F
            - self.drag_coef[:,None]
            * vel[1:-1]
            * self.vel_norm[:,None]
        ) / self.m[:,None]

        out[
            self.offset+2:
            self.offset+2*(self.N-1)
        ] = acc.ravel()

        ########################################################
        # Slackliner equation
        ########################################################

        if self.with_slackliner:

            i = self.start_slackliner

            vel_slack = Z[
                i+self.offset:
                i+self.offset+2
            ]

            vel_norm = np.sqrt(
                vel_slack[0]**2 +
                vel_slack[1]**2
            )

            d = proj - z_slack

            if dist > self.slackliner.l_leash:

                beta = (dist-self.slackliner.l_leash)/self.slackliner.l_leash

                F_slack = (
                    self.slackliner.m*self.g
                    + self.kl_leash*beta*d/dist
                )

            else:

                F_slack = self.slackliner.m*self.g

            out[
                i+self.offset:
                i+self.offset+2
            ] = F_slack/self.slackliner.m

        return out

    # TODO rewrite to vectorize and assume reshaped pos
    def compute_tension_mainline(self, pos):
        t = np.empty(2*N-2)
    
        for i in range(0,2*N-2,2):
            zi = np.array([pos[i], pos[i+1]])
            zip1 = np.array([pos[i+2], pos[i+3]])
            t[i] = tension(zi, zip1, self.kl[i], self.l[i])
    
        return np.max(t)
    
    def get_force_from_pos(self, pos):
        # vectors to previous and next nodes
        d_prev = pos[:-1,:] - pos[1:,:] 
        dist_prev = np.linalg.norm(d_prev, axis=1)
    
        main = self.k * np.maximum(dist_prev - self.l, 0.0) 
        backup = self.k * np.maximum(dist_prev - self.l_backup, 0.0)
        
        kl_beta_prev = np.where(
            self.break_mainline,
            backup,
            main + backup,
        )
    
        F_mag_prev = kl_beta_prev[:] 
    
        return F_mag_prev, dist_prev
    
    def post_process(self, result, skip = 1):
    
        #max force in
        f_webbing = np.empty_like(result["t"])
        f_anchor1 = np.empty_like(result["t"])
        f_anchor2 = np.empty_like(result["t"])
        f_leash = np.empty_like(result["t"])
        backup_activated = np.empty_like(result["t"], dtype = bool)
        backup_activated_segments = np.empty((self.n_edges,len(result["t"])), dtype = bool)
        
        # Gforces in leash
        G_leash = np.zeros_like(result["t"])
    
        # lowest point
        lp_start_webbing = np.min(result["y"][0:2*self.N,0])
        lp_webbing = 0
        lp_slackliner = 0
    
        N = self.N

        for i in tqdm(range(0, len(result["t"]), skip), desc = "Post processing:"):
    
            lp_webbing = min(np.min(result["y"][1:2*N:2,i]), lp_webbing)
            lp_slackliner = min(result["y"][self.start_slackliner + 1,i], lp_slackliner)
    
            Z = result["y"][:,i]
    
            pos = Z[:2*N].reshape(N, 2)
            vel = Z[self.offset:self.offset+2*N].reshape(N, 2)
    
            F_mag_prev, dist_prev = self.get_force_from_pos(pos)
    
            f_webbing[i] = np.max(F_mag_prev)
            f_anchor1[i] = F_mag_prev[0]
            f_anchor2[i] = F_mag_prev[-1]
            backup_activated_segments[:,i] = np.maximum(0.0, dist_prev - self.l_backup)
            backup_activated[i] = np.any(backup_activated_segments[:,i])
    
            if (self.with_slackliner):
                jj = self.start_slackliner
                zslackliner = Z[jj: jj+2]
        
                proj, dist, _, _, _ = project_along_y(zslackliner, pos)
    
                f_leash[i] = tension(proj, zslackliner, self.kl_leash, self.slackliner.l_leash)
                if (i > 0):
                    i_prev = i - 20
                    dt = result["t"][i] - result["t"][i_prev]
                    vel_slack = Z[self.offset + self.start_slackliner:self.offset + self.start_slackliner + 2]
                    vel_slack_prev = result["y"][self.offset + self.start_slackliner:self.offset + self.start_slackliner + 2, i_prev]
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
    
    def leash_event(self, t, Z):
        if (not self.with_slackliner or not self.detect_collision):
            return 1.0  # never trigger
    
        pos = Z[:2*self.N].reshape(self.N, 2)
    
        p_slacker = Z[self.start_slackliner:self.start_slackliner+2]
    
        _, dist, _, _, _ = project_along_y(p_slacker, pos)
    
        return dist - self.slackliner.l_leash
    
    def apply_collision(self, Z):
        """
        Applies an inelastic collision between the slackliner
        and the leash ring.
        """
    
        Z = Z.copy()
    
        pos = Z[:2*self.N].reshape(self.N, 2)
        p_slacker = Z[self.start_slackliner:self.start_slackliner+2]
        _, dist, i_prev, i_next, alpha = project_along_y(p_slacker, pos)
    
        # ring velocity
        i_vel_prev = self.offset + 2*i_prev
        v_prev = Z[i_vel_prev:i_vel_prev+2]
    
        i_vel_next = self.offset + 2*i_next
        v_next = Z[i_vel_next:i_vel_next+2]
    
        v_ring = interpolate(v_prev, v_next, alpha)
        m_ring = interpolate(self.m[i_prev-1], self.m[i_next-1], alpha)# TODO make function to map i_pos to i_mass
    
        # slackliner velocity
        i_slack = self.start_slackliner + self.offset
        v_slack = Z[i_slack:i_slack+2]
    
        # momentum-conserving velocity
        v = (
            m_ring*v_ring + 
            self.slackliner.m*v_slack
        )/(m_ring + self.slackliner.m)
    
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
    
    
    def static_rhs(self, Z, with_slackliner, after_break):
    
        out = np.zeros_like(Z)
    
        pos = Z.reshape(self.N,2)
    
        # boundary conditions
        out[0] = pos[0,0]
        out[1] = pos[0,1]
    
        out[-2] = pos[-1,0] - self.L
        out[-1] = pos[-1,1]
    
        d_prev = pos[:-2] - pos[1:-1]
        d_next = pos[2:]  - pos[1:-1]
    
        dist_prev = np.linalg.norm(d_prev, axis=1)
        dist_next = np.linalg.norm(d_next, axis=1)
    
        backup_prev = self.k_backup[:-1]*np.maximum(dist_prev-self.l_backup[:-1],0)
    
        beta_prev = (
            self.k[:-1]*np.maximum(dist_prev-self.l[:-1],0) + backup_prev
        )
    
        backup_next = self.k_backup[1:]*np.maximum(dist_next-self.l_backup[1:],0)
        beta_next = (
            self.k[1:]*np.maximum(dist_next-self.l[1:],0) + backup_prev
        )
    
        if (after_break):
            beta_prev[self.break_mainline[:-1]] = backup_prev[self.break_mainline[:-1]]
            beta_next[self.break_mainline[1:]] = backup_next[self.break_mainline[1:]]
    
        F_prev = beta_prev[:,None]*d_prev/dist_prev[:,None]
        F_next = beta_next[:,None]*d_next/dist_next[:,None]
    
        masses = self.m.copy()
    
        if with_slackliner:
            p_slacker = np.array([self.slackliner.x_coor, np.min(pos[:,1])]) # TODO maybe a hack?
            _, dist, i_prev, i_next, alpha = project_along_y(p_slacker, pos)
    
            masses[i_prev-1] += (1-alpha)*self.slackliner.m
            masses[i_next-1] += alpha    *self.slackliner.m
    
        F = masses[:,None]*self.g + F_prev + F_next
    
        out[2:-2] = F.reshape(-1)
    
        return out
    
    # Initial guess assuming that with weight of the slackliner and line in the middle
    # the line will sag to yield a tension of T_kN
    def get_initial_pos_from_tension(self, T_kN = 2):
        T_kg = 1000.0/9.82 * T_kN
        mass = self.slackliner.m + np.sum(self.m)
        s = mass * self.L / (4*T_kg)
        a = -2*s/self.L
    
        y = np.maximum(self.spacings*a, (self.L-self.spacings)*a) 
    
        positions = np.column_stack((self.spacings, y)).ravel()
    
        return positions
    
    def get_static_position(self, 
                            pos = None, 
                            with_slackliner = True, 
                            after_break = False):
        if (pos is None):
            w_line = np.sum(self.m) + self.slackliner.m
            pos = self.get_initial_pos_from_tension(T_kN = w_line*9.82/1000)
            # pos = get_initial_pos_from_tension(T_kN = 0.05)
    
        sol, info, ier, mesg = fsolve(self.static_rhs, pos, full_output=True, 
                                      args = (with_slackliner, after_break))
    
        if (ier != 1):
            print("Static solver could not converge!")
            print("ier:", ier)
            print("Message:", mesg)
            sys.exit()
        return sol
    
    def integrate_with_collisions(self, y0, t0, tf, **solve_kwargs):
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
                self.ode_rhs,
                (t_current, tf),
                y_current,
                events=self.leash_event,
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
            y_current = self.apply_collision(sol.y_events[0][0])
            collision_times.append(t_current)
    
            # pbar.write(f"Leash started to see tension at t = {t_current:.2f}")
    
        return {
            "t": np.array(t_all),
            "y": np.hstack(y_all),
            "collision_times": collision_times,
        }
    
    def simulate(self):
        pos = self.get_static_position(with_slackliner = True)
    
        pos = self.get_position_line_and_slackliner(pos, walking = True)
        vel = np.zeros_like(pos)
        Z = np.concatenate((pos, vel))
    
        # Simulate backup fall
        result_backupfall = None
        # if (np.any(break_mainline)):
        #     print("Simulating backup fall:")
        #     result_backupfall = integrate_with_collisions(
        #         Z,
        #         t0,
        #         t1,
        #         rtol=1e-10,
        #         atol=1e-10,
        #     )
        #     result_backupfall = post_process(result_backupfall, skip = 1) # Add postprocessing to result_backupfalls
    
        # Simulate leash fall
        print("Simulating leash fall:")
        pos = self.get_static_position(with_slackliner = True)
        self.break_mainline[:] = False
    
        result_leashfall = self.integrate_with_collisions(
            Z,
            self.t0,
            self.t1,
            rtol=1e-5, #TODO switch to module var
            atol=1e-5,
        )
        result_leashfall = self.post_process(result_leashfall, skip = 1) # Add postprocessing to result_backupfalls
    
        pos = self.get_static_position(with_slackliner = False)
        pos_static = pos[:2*self.N].reshape(self.N, 2)
        F_mag_prev, _ = self.get_force_from_pos(pos_static)
        
        f_standing = np.max(F_mag_prev)
        result_leashfall["f_standing"] = f_standing
        # result_backupfall["f_standing"] = f_standing
    
        result_leashfall["w_line"] = np.sum(self.m)
        # result_backupfall["w_line"] = np.sum(m)
    
        return result_leashfall, result_backupfall
    
    def discretize_segments(self):
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
        x = self.spacings

        self.l = np.diff(x)
        mid = x[:-1] + 0.5 * self.l
    
        # Segment boundaries
        bounds = np.array([0.0, *accumulate(s.L_main for s in self.segs)])
    
        if (bounds[-1] < x[-1]):
            print(f" Webbing not long enough. {bounds[-1]}m webbing doesn't bridge the {x[-1]}m gap")
            sys.exit()
    
        # Segment index for each interval
        self.seg_ids = np.searchsorted(bounds, mid, side="right") - 1
        self.seg_ids = np.clip(self.seg_ids, 0, self.n_segs - 1)
    
        self.kl = np.array([self.segs[i].kl_main for i in self.seg_ids])
        self.kl_backup = np.array([self.segs[i].kl_backup for i in self.seg_ids])
    
        self.rho = np.array([self.segs[i].rho_main for i in self.seg_ids])
        self.rho_backup = np.array([self.segs[i].rho_backup for i in self.seg_ids])
    
        self.l_backup = np.array([self.l[j] * self.segs[i].L_backup/self.segs[i].L_main for (j,i) in enumerate(self.seg_ids)])
    
        # Interior node masses (half from each neighbouring interval)
        self.m = self.get_mass_from_l()
    
        self.break_mainline = np.array([self.segs[i].break_mainline for i in self.seg_ids], dtype=bool)
    
    
    def get_mass_from_l(self):
        interval_mass = self.rho * self.l + self.rho_backup * self.l_backup
        m = 0.5 * (interval_mass[:-1] + interval_mass[1:])
        m += 0.5 * (interval_mass[0] + interval_mass[-1]) / len(m)
        return m

# TODO move this to a helpers file?
def interpolate(a,b,alpha):
    return (1-alpha)*a + alpha*b

def project_along_y(point, vertices):
    """
    Project a point vertically onto a polyline.

    Returns
    -------
    projection : (2,) ndarray
    segment0 : int
    segment1 : int
    alpha : float
    """

    x = point[0]

    n,m = vertices.shape
    seg = np.searchsorted(vertices[:, 0], x, side="right") - 1

    if seg < 0 or seg >= len(vertices)-1:
        raise ValueError("x-coordinate does not intersect the polyline.")

    dx = vertices[seg+1,0] - vertices[seg,0]

    if dx == 0:
        alpha = 0.0
        y = vertices[seg, 1]
    else:
        alpha = (x - vertices[seg,0]) / dx
        y = vertices[seg, 1] + alpha * (vertices[seg+1, 1] - vertices[seg, 1])

    proj = np.array([x, y])
    dist = np.sqrt((proj[0] - point[0])**2 + (proj[1] - point[1])**2)

    i_prev = seg
    i_next = seg+1

    if (i_next >= n-1):
        i_next = i_prev

    if (i_prev <= 0):
        i_prev = i_next

    return proj, dist, i_prev, i_next, alpha

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
