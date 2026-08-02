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
        self.backup = np.empty(self.n_edges)
        self.beta = np.empty(self.n_edges)
        self.scale = np.empty(self.n_edges)

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
        self.C_D = 1.15                 # Drag coeff of rectangle
        self.webbing_width = 0.0254     # [m]
        self.damp_kelvin_voigt = 2E3      # Kelving Voigt Dampening Coefficient

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
        

    # get_net_forces has to be called first TODO maybe refactor that
    def get_kelvin_voigt_dampening(self, vel):


        np.subtract(vel[1:], vel[:-1], out=self.d_vel)

        # < delta vel, delta p > / ||delta p||^2
        np.sum(self.d_vel * (self.d_edge / self.dist_edge_squared[:,None]), axis=1, out=self.proj_vel)

        self.proj_vel = np.where(np.logical_and(self.dist_edge > self.l, np.logical_not(self.break_mainline)), self.proj_vel, 0.0) 

        if (self.has_backup):
            self.proj_vel += np.where(self.dist_edge > self.l_backup, self.proj_vel, 0.0)

        # Subtract force in both directions prev and next.
        F = np.zeros_like(self.F)
        F[1:] = -self.damp_kelvin_voigt * self.proj_vel[:, None] * self.d_edge
        F[:-1] += self.damp_kelvin_voigt * self.proj_vel[:, None] * self.d_edge

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
    
