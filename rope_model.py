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

# This class represents a rope modelled by connecting nodes with springs
# 
# It consists of n points
# n_free represents the free nodes.
# You can ask it to fix one or two of the endpoints
# You can have it model two springs between nodes
# It can essentially just compute all the net internal forces given positions of noed
# It can also compute dampening forces given the velocities
# Finally it can compute the static position using fsolve
#
# Another class is needed to do time-integration
@dataclass
class RopeModel:
    fix_start : bool
    fix_end   : bool

    def precompute_constants(self):
        # Precompute constants
        self.k = self.kl / self.l
        if (self.has_backup):
            self.k_backup = self.kl_backup / self.l_backup

        self.drag_constant = (
            0.5 * self.rho_air * self.C_D * (self.webbing_width / 2)
        )

    def preallocate_workspace(self):
        self.d_edge = np.empty((self.n_edges, 2))
        self.dist_edge_squared = np.empty((self.n_edges, 2))
        self.dist_edge = np.empty(self.n_edges)

        self.d_vel = np.empty((self.n_edges, 2))
        self.proj_vel = np.empty(self.n_edges)

        self.stretch = np.empty(self.n_edges)
        self.stretch_backup = np.zeros(self.n_edges)
        self.backup = np.empty(self.n_edges)
        self.beta = np.empty(self.n_edges)
        self.scale = np.empty(self.n_edges)

        # Largest stretch seen so far, per edge - the load history the unload
        # path is anchored to
        self.s_max = np.zeros(self.n_edges)
        self.s_max_backup = np.zeros(self.n_edges)

        # Rate of change of the edge lengths, and the low pass of it that says
        # which way the edge is going
        self.rate = np.zeros(self.n_edges)
        self.rate_filtered = np.zeros(self.n_edges)

        self.F = np.zeros((self.n, 2))
        self.F_free = np.zeros((self.n_free, 2))
        self.drag_coef = np.empty(self.n_free)
        self.vel_norm = np.empty(self.n_free)

    @cached_property
    def n_edges(self):
        return self.n - 1

    @cached_property
    def n_free(self):
        return self.n - self.fix_start - self.fix_end

    def __init__(
        self,
        L,              # Length of rope
        n,              # Number of discretization verticies
        kl, 
        l,
        break_mainline,
        fix_start,
        fix_end,
        kl_backup=None,
        l_backup=None,
    ):
        self.L = L
        self.n = n
        self.kl = kl
        self.kl_backup = kl_backup
        self.l = l
        self.l_backup = l_backup
        self.break_mainline = break_mainline
        self.fix_start = fix_start
        self.fix_end = fix_end

        self.has_backup = self.l_backup is not None

        # ========================================
        # Setup parameters of numerical model
        # ========================================

        # Physical parameters
        self.g = np.array([0, -9.82])   # gravitation [m/s^2]
        self.rho_air = 1.225            # [kg/m^3]
        self.C_D = 0.0                  # Drag coeff, off - the load path damps
        self.webbing_width = 0.0254     # [m]
        self.damp_kelvin_voigt = 0.0      # Kelving Voigt Dampening Coefficient

        # Load and unload along different paths. Every edge remembers s_max, the
        # largest stretch it has ever seen, and anything below that comes back
        # down a softer curve than it went up, so a load cycle returns less
        # energy than it took. unload_ratio is how much of the virgin stiffness
        # is left at the bottom of that curve (1 turns the hysteresis off);
        # unload_rate [1/s] is the strain rate above which an edge counts as
        # unloading rather than as standing still.
        self.unload_ratio = 0.4
        self.unload_rate = 0.01

        # How sharply the unload path peels away from the virgin one just below
        # the peak. 1 leaves them tangent there and most of the loop unenclosed;
        # higher values drop the tension as soon as the edge turns around, which
        # is what a real unloading curve does and what carries most of the loss.
        self.unload_exp = 3.0

        # Whether an edge is loading or unloading is read off a low pass of its
        # stretch rate, with this time constant [s]. Without damping the mesh
        # rings on undisturbed at a few hundred Hz, and the raw rate changes sign
        # with the ringing rather than with the fall: the load and the unload
        # path then get picked at random and the load cycle stops enclosing any
        # area at all. A hundredth of a second is far below anything the fall
        # does and far above the ringing.
        self.unload_tau = 0.01

        # Setup
        self.precompute_constants()
        self.preallocate_workspace()

    def compute_max_tension(self, pos):
        N = self.n
        t = np.empty(2*N-2)
    
        for i in range(0,2*N-2,2):
            zi = np.array([pos[i], pos[i+1]])
            zip1 = np.array([pos[i+2], pos[i+3]])
            t[i] = tension(zi, zip1, self.kl[i], self.l[i])
    
        return np.max(t)

    def get_net_forces(self, pos, after_break = True):
        ########################################################
        # Spring forces
        ########################################################

        np.subtract(pos[1:], pos[:-1], out=self.d_edge)

        self.dist_edge_squared = self.d_edge[:,0]**2 + self.d_edge[:,1]**2
        np.sqrt(self.dist_edge_squared, out=self.dist_edge)

        self.stretch[:] = self.dist_edge
        self.stretch -= self.l
        self.stretch.clip(min=0, out=self.stretch)

        self.beta[:] = self.k
        self.beta *= self.stretch

        if (self.has_backup):
            self.backup[:] = self.dist_edge
            self.backup -= self.l_backup
            self.backup.clip(min=0, out=self.backup)
            self.stretch_backup[:] = self.backup
            self.backup *= self.k_backup

            self.beta += self.backup
            if after_break: 
                self.beta[self.break_mainline] = self.backup[self.break_mainline]

        # An edge of zero length carries no tension, but beta/dist_edge is 0/0
        # there and NaNs the whole state. It happens on the leash whenever the
        # slackliner starts on the line (l_leg = 0).
        self.scale[:] = 0.0
        np.divide(
            self.beta,
            self.dist_edge,
            out=self.scale,
            where=self.dist_edge > 0,
        )

        F = np.zeros_like(self.F)
        F[1:] = -self.d_edge * self.scale[:, None]
        F[:-1] += self.d_edge * self.scale[:, None]

        return F

    def get_net_forces_free_nodes(self, pos, after_break = True):
        F = self.get_net_forces(pos, after_break)

        if (self.fix_start and self.fix_end):
            return F[1:-1]
        elif(self.fix_start):
            return F[1:]
        elif(self.fix_end):
            return F[:-1]
        else:
            return F
        

    # The load history: the largest stretch every edge has seen, and the low pass
    # of its stretch rate. Both have to be advanced once per accepted time step,
    # which is why the solver takes fixed steps: an adaptive one retries steps and
    # interpolates between them, and the history would then follow the solver's
    # trials rather than the motion of the line.
    # get_hysteresis_forces has to be called first
    def advance_load_history(self, dt):
        np.maximum(self.s_max, self.stretch, out=self.s_max)
        if (self.has_backup):
            np.maximum(self.s_max_backup, self.stretch_backup, out=self.s_max_backup)

        self.rate_filtered += (dt / self.unload_tau) * (self.rate - self.rate_filtered)

    def reset_load_history(self):
        self.s_max[:] = 0.0
        self.s_max_backup[:] = 0.0
        self.rate[:] = 0.0
        self.rate_filtered[:] = 0.0

    # How far the unload curve sits below the virgin one. Loading follows the
    # linear spring; unloading follows
    #     F = k*e * (unload_ratio + (1 - unload_ratio) * (e/s_max)**unload_exp)
    # which meets the virgin curve at s_max, where the edge turned around, and
    # runs below it all the way down. It reaches zero at zero stretch, so no edge
    # is left with a permanent set, and it never goes negative, so the webbing
    # still cannot push.
    def unload_drop(self, beta, stretch, s_max):
        u = np.zeros_like(beta)
        np.divide(stretch, s_max, out=u, where=s_max > 0)

        # s_max is only advanced at the end of a step, so within a step an edge
        # climbing to a new maximum sits above it. That is virgin loading, no drop
        np.clip(u, 0.0, 1.0, out=u)

        return (1 - self.unload_ratio) * (1 - u**self.unload_exp) * beta

    # get_net_forces has to be called first
    def get_hysteresis_forces(self, vel, after_break = True):
        np.subtract(vel[1:], vel[:-1], out=self.d_vel)

        # d/dt of the edge length. A zero length edge has no direction to measure
        # it along, guarded like the spring force
        self.rate[:] = 0.0
        np.divide(
            np.sum(self.d_vel * self.d_edge, axis=1),
            self.dist_edge,
            out=self.rate,
            where=self.dist_edge > 0,
        )

        if (self.unload_ratio >= 1.0):
            return np.zeros_like(self.F)

        # 1 while the edge is shortening, 0 while it is lengthening, and a ramp
        # between the two so that an edge which has stopped moving is back on its
        # virgin curve - the state the line settles on is then the one it would
        # settle on with none of this.
        #
        # Both the filtered rate and the raw one have to say the edge is
        # shortening. The filter is what the fall is doing, and on its own it
        # lags: it would leave the unload path switched on for a few
        # milliseconds after an edge starts stretching again, and taking tension
        # away from an edge that is lengthening puts energy in rather than
        # taking it out. The raw rate is the guard against that.
        phi = (np.clip(-self.rate_filtered / (self.unload_rate * self.l), 0.0, 1.0)
               * np.clip(-self.rate / (self.unload_rate * self.l), 0.0, 1.0))

        drop = self.unload_drop(self.k * self.stretch, self.stretch, self.s_max)

        if (self.has_backup):
            drop_backup = self.unload_drop(self.backup, self.stretch_backup,
                                           self.s_max_backup)
            if after_break:
                drop = np.where(self.break_mainline, drop_backup, drop + drop_backup)
            else:
                drop += drop_backup

        drop *= phi

        scale = np.zeros(self.n_edges)
        np.divide(drop, self.dist_edge, out=scale, where=self.dist_edge > 0)

        # The opposite sign of the spring force: this takes tension away
        F = np.zeros_like(self.F)
        F[1:] = self.d_edge * scale[:, None]
        F[:-1] -= self.d_edge * scale[:, None]

        return F

    def get_hysteresis_forces_free(self, vel, after_break = True):
        F = self.get_hysteresis_forces(vel, after_break)

        if (self.fix_start and self.fix_end):
            return F[1:-1]
        elif(self.fix_start):
            return F[1:]
        elif(self.fix_end):
            return F[:-1]
        else:
            return F


    # get_net_forces has to be called first TODO maybe refactor that
    def get_kelvin_voigt_dampening(self, vel):

        if (not self.damp_kelvin_voigt):
            return np.zeros_like(self.F)


        np.subtract(vel[1:], vel[:-1], out=self.d_vel)

        # < delta vel, delta p > / ||delta p||^2
        # Guarded like the spring force: a zero length edge has no direction to
        # damp along, and this is 0/0 there
        self.proj_vel[:] = 0.0
        np.divide(
            np.sum(self.d_vel * self.d_edge, axis=1),
            self.dist_edge_squared,
            out=self.proj_vel,
            where=self.dist_edge > 0,
        )

        taut = np.logical_and(self.dist_edge > self.l, np.logical_not(self.break_mainline))
        if self.has_backup:
            taut = np.logical_or(taut, self.dist_edge > self.l_backup)
        self.proj_vel = np.where(taut, self.proj_vel, 0.0)

        # Damping per unit length, so that refining the mesh keeps modelling the
        # same material. self.k = self.kl / self.l does the same for the springs
        c = self.damp_kelvin_voigt / self.l

        # Webbing cannot push. The dashpot is in parallel with the springs, so cap
        # it where it would drag the total edge tension, beta + c*proj_vel*dist,
        # below zero - which happens on any edge unloading faster than damp/(k*l)
        limit = np.zeros_like(self.proj_vel)
        np.divide(-self.beta, c * self.dist_edge, out=limit, where=self.dist_edge > 0)
        np.maximum(self.proj_vel, limit, out=self.proj_vel)

        # Subtract force in both directions prev and next.
        F = np.zeros_like(self.F)
        F[1:] = -(c * self.proj_vel)[:, None] * self.d_edge
        F[:-1] += (c * self.proj_vel)[:, None] * self.d_edge

        return F

    
    def get_kelvin_voigt_dampening_free(self, vel):
        F = self.get_kelvin_voigt_dampening(vel)

        if (self.fix_start and self.fix_end):
            return F[1:-1]
        elif(self.fix_start):
            return F[1:]
        elif(self.fix_end):
            return F[:-1]
        else:
            return F


    def get_drag_force(self, vel):
        np.sqrt(
            vel[1:-1,0]**2 +
            vel[1:-1,1]**2,
            out=self.vel_norm,
        )

        # Note that the drag is scaled with length of section to acount for 
        # the area of this part of the webbing
        self.drag_coef[:] = self.drag_constant * (self.dist_edge[:-1] + self.dist_edge[1:])

        F_free = -self.drag_coef[:,None] * vel[1:-1] * self.vel_norm[:,None]
        
        return F_free


if __name__ == "__main__":
    print("Hello world")
    
