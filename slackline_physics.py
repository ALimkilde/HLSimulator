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
    def n_edges_line(self):
        return self.N_line - 1

    @cached_property
    def n_edges_leash(self):
        return self.N_leash - 1

    @cached_property
    def n_masses_line(self):
        return self.N_line - 2

    @cached_property
    def n_masses_leash(self):
        return self.N_leash

    @cached_property
    def n_edges(self):
        return self.n_edges_line + self.n_edges_leash

    @cached_property
    def n_masses(self):
        return self.n_masses_line + self.n_masses_leash

    @cached_property
    def n_segs(self):
        return len(self.segs)

    @cached_property
    def start_main(self):
        return 0

    @cached_property
    def start_masses_leash(self):
        return self.n_masses_line

    @cached_property
    def end_edges_leash(self):
        return self.start_masses_leash + self.n_masses_leash

    @cached_property
    def start_edges_leash(self):
        return self.n_edges_line

    @cached_property
    def end_edges_leash(self):
        return self.start_edges_leash + self.n_edges_leash

    @cached_property
    def start_leash(self):
        return self.N_line

    @cached_property
    def end_leash(self):
        return self.start_leash + self.N_leash

    @cached_property 
    def offset(self):
        return self.N_line + self.N_leash - 1

    @cached_property
    def I_line_masses(self):
        return range(0,self.n_masses_line)

    @cached_property
    def I_line_edges(self):
        return range(0,self.n_edges_line)

    @cached_property
    def I_leash_edges(self):
        return range(self.start_edges_leash,self.end_edges_leash)

    @cached_property
    def I_leash_masses(self):
        return range(self.n_masses_line,self.n_masses_line + self.n_masses_leash)

    @cached_property
    def I_line_prev(self):
        return range(0,self.n_edges_line - 1)

    @cached_property
    def I_line_next(self):
        return range(1,self.n_edges_line)

    @cached_property
    def I_leash_prev(self):
        return range(self.start_edges_leash, self.end_edges_leash)

    @cached_property
    def I_leash_next(self):
        return range(self.start_edges_leash + 1, self.end_edges_leash)

    @cached_property
    def i_mass_slackliner(self):
        return self.n_masses - 1

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

    def allocate_properties(self):
        self.break_mainline = np.zeros(self.n_edges, dtype=bool)
        self.m = np.empty(self.n_masses)
        self.l = np.empty(self.n_edges)
        self.l_backup = np.empty(self.n_edges)
        self.kl = np.empty(self.n_edges)
        self.kl_backup = np.zeros(self.n_edges)
        self.rho = np.empty(self.n_edges)
        self.rho_backup = np.empty(self.n_edges)

    def preallocate_workspace(self):
        self.d_edge = np.empty((self.n_edges, 2))
        self.dist_edge_squared = np.empty((self.n_edges, 2))
        self.dist_edge = np.empty(self.n_edges)

        self.d_vel = np.empty((self.n_edges, 2))
        self.proj_vel = np.empty(self.n_edges)

        self.stretch = np.empty(self.n_edges)
        self.backup = np.empty(self.n_edges)
        self.beta = np.empty(self.n_edges)
        self.scale = np.empty(self.n_edges)

        self.F = np.empty((self.n_masses, 2))
        self.drag_coef = np.empty(self.n_masses)
        self.vel_norm = np.empty(self.n_masses)

    def __init__(
        self,
        L,              # Length of highline spot
        N,              # Number of discretization verticies
        N_leash,        # Number of discretization verticies for leash
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
        self.N_line = N
        self.N_leash = N_leash

        # Setup parameters of numerical model
        self.detect_collision = False
        #TODO : move rtol and atol here!

        # Physical parameters
        self.g = np.array([0, -9.82])   # gravitation [m/s^2]
        self.rho_air = 1.225            # [kg/m^3]
        self.C_D = 1.15                 # Drag coeff of rectangle
        self.webbing_width = 0.0254     # [m]
        self.kl_leash = 200*1E3         # Spring constant times length - Leash
        self.rho_leash = 0.3            # Density of leash!
        self.damp_kelvin_voigt = 2E3      # Kelving Voigt Dampening Coefficient

        # Progress bar
        self.init_progress_bar()

        # Create discretization (mesh)
        self.spacings = self.init_spacings()
        self.allocate_properties()
        self.discretize_segments()

        # Add tension
        self.add_tension(pull_webbing, self.n_segs-1)

        # Setup
        self.precompute_constants()
        self.preallocate_workspace()

    # Meshing routine essentially
    def init_spacings(self):
       return np.linspace(0,self.L,self.N_line)

    # Adjust tension by adding/decreasing webbing
    # Add by 2m      : w = -2
    # Decrease by 2m : w = 2
    def add_tension(self, w, seg_id):
        if (seg_id >= len(self.segs)):
            print("wrong section for tensioning")
            sys.exit()
        l_line = self.l[self.I_line_edges]

        L_main = np.sum(l_line[self.seg_ids == seg_id])
        alpha = -w/L_main

        l_line[self.seg_ids == seg_id] = l_line[self.seg_ids == seg_id]*(1+alpha)

        self.l[self.I_line_edges] = l_line
    
        # Update mass of line, as some has been removed
        self.m[self.I_line_masses] = self.get_mass_from_l_line(self.I_line_edges)

    def get_position_line_and_slackliner(self, pos, walking = False):
        # TODO Retire N_leash and only store n_edges_leash
        if walking:
            v = np.linspace(0, self.slackliner.l_leg, self.N_leash)
        else:
            v = np.linspace(0, -self.slackliner.l_leash, self.N_leash)

        y_min = np.min(pos[1::2])
        zslackliner = np.array([self.slackliner.x_coor, y_min]) # Initial guess
        proj, dist, _, _, _ = project_along_y(zslackliner, pos.reshape(self.N_line, 2)) # Project to line
        p_slacker = proj # set position on line to projection

        p_leash = np.zeros((self.n_edges_leash, 2))
        p_leash[:,1] = v[1:] 
        p_leash += p_slacker[None,:]

        if (pos.size <= self.N_line*2):
            pos = np.concatenate((pos, p_leash.ravel()))
        elif (pos.size >= self.offset*2):
            pos[2*self.start_leash: 2*self.end_leash]  = p_leash.ravel()

        return pos

    def set_d_edge(self, pos, proj):
        # Calculate length of all line segments
        np.subtract(pos[1:], pos[:-1], out=self.d_edge)

        # Correct first segment of leash
        self.d_edge[self.start_edges_leash] = pos[self.start_leash]- proj


    def set_d_vel(self, vel, i_prev, i_next, alpha):
        np.subtract(vel[1:], vel[:-1], out=self.d_vel)

        vel_ring = (1 - alpha) * vel[i_prev] + alpha * vel[i_next]

        self.d_vel[self.start_edges_leash] = vel[self.start_edges_leash] - vel_ring

    def add_drag_forces(self, vel):
        np.sqrt(
            vel[1:-1,0]**2 +
            vel[1:-1,1]**2,
            out=self.vel_norm,
        )

        # Note that the drag is scaled with length of section to acount for 
        # the area of this part of the webbing
        self.drag_coef[:] = self.drag_constant * (self.dist_edge[:-1] + self.dist_edge[1:])
        

        self.F -= self.drag_coef[:,None] * vel[1:-1] * self.vel_norm[:,None]

    def add_kelvin_voigt_dampening(self):
        # < delta vel, delta p > / ||delta p||^2
        np.sum(self.d_vel * (self.d_edge / self.dist_edge_squared[:,None]), axis=1, out=self.proj_vel)

        # TODO Figure out why the force is not opp sign of springs!
        self.proj_vel = np.where(np.maximum(self.dist_edge > self.l, np.logical_not(self.break_mainline)), self.proj_vel, 0.0) + np.where(self.dist_edge > self.l_backup, self.proj_vel, 0.0)

        # Update forces for line
        self.F[self.I_line_prev]-= self.damp_kelvin_voigt * self.proj_vel[self.I_line_prev, None] * self.d_edge[self.I_line_prev]
        self.F[self.I_line_next] += self.damp_kelvin_voigt * self.proj_vel[self.I_line_next, None] * self.d_edge[self.I_line_next]

        self.F[self.I_leash_prev]-= self.damp_kelvin_voigt * self.proj_vel[self.I_leash_prev, None] * self.d_edge[self.I_leash_prev]
        self.F[self.I_line_next] += self.damp_kelvin_voigt * self.proj_vel[self.I_line_next, None] * self.d_edge[self.I_line_next]

    def set_gravity_and_spring_forces(self):

        self.dist_edge_squared = self.d_edge[:,0]**2 + self.d_edge[:,1]**2
        np.sqrt(self.dist_edge_squared, out=self.dist_edge)

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

        self.F[:] = 0.0

        # Forces on line and leash
        self.F[self.I_line_prev, :] -= self.d_edge[self.I_line_prev, :] * self.scale[self.I_line_prev, None]
        self.F[self.I_line_next, :] += self.d_edge[self.I_line_next, :] * self.scale[self.I_line_next, None]

        self.F[self.I_leash_prev, :] -= self.d_edge[self.I_leash_prev, :] * self.scale[self.I_leash_prev, None]
        self.F[self.I_leash_next, :] += self.d_edge[self.I_leash_next, :] * self.scale[self.I_leash_next, None]

        F_leash = self.F[self.start_edges_leash].copy()

        self.F[:] += self.gravity_force

        return F_leash


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

        reshaped = Z.reshape(2*self.offset, 2)
        out = np.zeros_like(reshaped)

        pos = reshaped[:self.offset, :]
        vel = reshaped[self.offset:, :]

        # The change of position is simply the velocities.
        out[:self.offset, :] = reshaped[self.offset:, :]

        ########################################################
        # Spring forces
        ########################################################

        # Get leash ring position
        proj, _, i_prev, i_next, alpha = (
                project_along_y(pos[self.start_leash, :], pos[self.I_line_edges])
        )

        # Set lenght of line segments
        self.set_d_edge(pos, proj)

        # TODO: work on name?
        # It also computes and stores :
        #    self.dist_edge_squared
        #    self.dist_edge
        #    Maybe more?
        F_leash = self.set_gravity_and_spring_forces() # Calculates and put spring forces into self.F

        self.F[i_prev] -= (1-alpha)*F_leash
        self.F[i_next] -= alpha*F_leash

        #######################################################
        # Kelving Voigt Dampening
        ########################################################

        # Set delta velocities
        # self.set_d_vel(vel, i_prev, i_next, alpha)

        # Add dampening forces
        # self.add_kelvin_voigt_dampening()

        ########################################################
        # Drag
        ########################################################

        # self.add_drag_forces(vel)

        ########################################################
        # Combine all
        ########################################################

        acc = (
            self.F
        ) / self.m[:,None]

        # TODO: How do I set the out easily?
        #     - Set first and last part of line to zero
        #     - Set leash x coordinates to zero
        # TODO: Mis this really correct?
        out[ self.offset+1: self.offset+self.n_masses_line, :] = acc[1:self.n_masses_line, :]
        out[ self.offset+self.N_line:, :] = acc[self.n_masses_line+1:, :]

        return out.ravel()

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
    
        main = self.k[self.I_line_edges] * np.maximum(dist_prev - self.l[self.I_line_edges], 0.0) 
        backup = self.k_backup[self.I_line_edges] * np.maximum(dist_prev - self.l_backup[self.I_line_edges], 0.0)
        
        kl_beta_prev = np.where(
            self.break_mainline[self.I_line_edges] ,
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
        backup_activated_segments = np.empty((self.n_edges_line,len(result["t"])), dtype = bool)
        
        # Gforces in leash
        G_leash = np.zeros_like(result["t"])

        N = self.N_line
    
        # lowest point
        lp_start_webbing = np.min(result["y"][0:2*N,0])
        lp_webbing = 0
        lp_slackliner = 0
    

        for i in tqdm(range(0, len(result["t"]), skip), desc = "Post processing:"):
    
            lp_webbing = min(np.min(result["y"][1:2*N:2,i]), lp_webbing)
            lp_slackliner = min(result["y"][self.start_leash + 1,i], lp_slackliner)
    
            Z = result["y"][:,i]
    
            pos = Z[:2*N].reshape(N, 2)
            vel = Z[self.offset:self.offset+2*N].reshape(N, 2)
    
            F_mag_prev, dist_prev = self.get_force_from_pos(pos)
    
            f_webbing[i] = np.max(F_mag_prev)
            f_anchor1[i] = F_mag_prev[0]
            f_anchor2[i] = F_mag_prev[-1]
            backup_activated_segments[:,i] = np.maximum(0.0, dist_prev - self.l_backup[self.I_line_edges])
            backup_activated[i] = np.any(backup_activated_segments[:,i])
    
            jj = 2*(self.end_leash-2)
            zslackliner = Z[jj:jj+2]
        
            proj, dist, _, _, _ = project_along_y(zslackliner, pos)
    
            # TODO update and modernize this routine
            f_leash[i] = tension(proj, zslackliner, self.kl_leash, self.slackliner.l_leash)
    
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
                # f"Gforce:    {np.max(G_leash):.2f}\n\n"
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
        if (not self.detect_collision):
            return 1.0  # never trigger
    
        # TODO Retire this code
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
    
    
    # TODO: Rewrite and modernize using 'get_gravity_and_spring_forces
    def static_rhs(self, Z, with_slackliner, after_break):
        
        N = self.N_line
    
        out = np.zeros_like(Z)
    
        pos = Z.reshape(N,2)
    
        # boundary conditions
        out[0] = pos[0,0]
        out[1] = pos[0,1]
    
        out[-2] = pos[-1,0] - self.L
        out[-1] = pos[-1,1]
    
        d_edges = pos[1:]  - pos[:-1]
    
        dist_edges = np.linalg.norm(d_edges, axis=1)
    
        backup = self.k_backup[self.I_line_edges]*np.maximum(dist_edges-self.l_backup[self.I_line_edges],0)
    
        beta = self.k[self.I_line_edges]*np.maximum(dist_edges-self.l[self.I_line_edges],0) + backup
    
        if (after_break):
            beta[self.break_mainline[self.I_line_edges]] = backup[self.break_mainline[self.I_line_edges]]
    
        F = beta[:,None]*d_edges/dist_edges[:,None]
    
        masses = np.empty(self.n_masses_line)
        masses[:] = self.m[:self.n_masses_line]
    
        if with_slackliner:
            p_slacker = np.array([self.slackliner.x_coor, np.min(pos[:,1])]) # Random y position
            _, dist, i_prev, i_next, alpha = project_along_y(p_slacker, pos)
    
            masses[i_prev-1] += (1-alpha)*self.slackliner.m
            masses[i_next-1] += alpha    *self.slackliner.m
    
        net_force = masses[:,None]*self.g - F[self.I_line_prev] + F[self.I_line_next]
    
        out[2:-2] = net_force.ravel()
    
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
    
        # TODO : fix such that pos is always on non ravel format.
        pos = self.get_position_line_and_slackliner(pos, walking = False)
        vel = np.zeros_like(pos)
        Z = np.concatenate((pos, vel))
    
        # Simulate backup fall
        result_backupfall = None
        # if (np.any(self.break_mainline)):
        #     print("Simulating backup fall:")
        #     result_backupfall = self.integrate_with_collisions(
        #         Z,
        #         self.t0,
        #         self.t1,
        #         rtol=1e-10,
        #         atol=1e-10,
        #     )
        #     result_backupfall = self.post_process(result_backupfall, skip = 1) # Add postprocessing to result_backupfalls
    
        # Simulate leash fall
        print("Simulating leash fall:")
        pos = self.get_static_position(with_slackliner = True)
        self.break_mainline[:] = False
    
        result_leashfall = self.integrate_with_collisions(
            Z,
            self.t0,
            self.t1,
            rtol=1e-2, #TODO switch to module var
            atol=1e-2,
        )
        result_leashfall = self.post_process(result_leashfall, skip = 1) # Add postprocessing to result_backupfalls
    
        pos = self.get_static_position(with_slackliner = False)
        pos_static = pos[:2*self.N_line].reshape(self.N_line, 2)
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
        x_line = self.spacings

        # Fill data for Line
        bounds = np.array([0.0, *accumulate(s.L_main for s in self.segs)])
    
        if (bounds[-1] < x_line[-1]):
            print(f" Webbing not long enough. {bounds[-1]}m webbing doesn't bridge the {x_line[-1]}m gap")
            sys.exit()

        self.l[self.I_line_edges] = np.diff(x_line)
        mid = x_line[:-1] + 0.5 * self.l[self.I_line_edges]
    
        self.seg_ids = np.searchsorted(bounds, mid, side="right") - 1
        self.seg_ids = np.clip(self.seg_ids, 0, self.n_segs - 1)
    
        self.kl[self.I_line_edges] = np.array([self.segs[i].kl_main for i in self.seg_ids])
        self.kl_backup[self.I_line_edges] = np.array([self.segs[i].kl_backup for i in self.seg_ids])
    
        self.rho[self.I_line_edges] = np.array([self.segs[i].rho_main for i in self.seg_ids])
        self.rho_backup[self.I_line_edges] = np.array([self.segs[i].rho_backup for i in self.seg_ids])
    
        self.l_backup[self.I_line_edges] = np.array([self.l[j] * self.segs[i].L_backup/self.segs[i].L_main for (j,i) in enumerate(self.seg_ids)])

        self.break_mainline[self.I_line_edges] = np.array([self.segs[i].break_mainline for i in self.seg_ids], dtype=bool)

        self.m[self.I_line_masses] = self.get_mass_from_l_line(self.I_line_edges)

        # Fill data for leash
        x_leash = np.linspace(0,self.slackliner.l_leash,self.N_leash)

        self.l[self.I_leash_edges] = np.diff(x_leash)
        self.kl[self.I_leash_edges] = self.kl_leash
        self.rho[self.I_leash_edges] = self.rho_leash
        self.m[self.I_leash_masses] = self.get_mass_from_l_leash(self.I_leash_edges)
    
    # Interior node masses (half from each neighbouring interval)
    def get_mass_from_l_line(self, I):
        interval_mass = self.rho[I] * self.l[I] + self.rho_backup[I] * self.l_backup[I]
        m = 0.5 * (interval_mass[:-1] + interval_mass[1:])
        m += 0.5 * (interval_mass[0] + interval_mass[-1]) / len(m)
        return m

    # Interior node masses (half from each neighbouring interval)
    def get_mass_from_l_leash(self, I):
        interval_mass = self.rho[I] * self.l[I] + self.rho_backup[I] * self.l_backup[I]
        m = np.empty(self.n_masses_leash)

        m[1:-1] = 0.5 * (interval_mass[:-1] + interval_mass[1:])
        m[0] = 0.5 * interval_mass[0] 
        m[-1] = 0.5 * interval_mass[-1] + self.slackliner.m
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
