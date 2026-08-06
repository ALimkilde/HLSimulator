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

        # The friction slider, per spring. hyst_tension is a fraction of kl, the
        # tension the webbing carries at 100% strain, so the force the friction
        # holds does not depend on how finely the line is meshed. stick_max does,
        # in step with the edges, so that the stretch it takes to reverse a whole
        # line does not.
        self.k_hyst = self.hyst_tension * self.kl / (self.hyst_strain * self.l)
        self.stick_max = self.hyst_strain * self.l

        if (self.has_backup):
            self.k_hyst_backup = (self.hyst_tension * self.kl_backup
                                  / (self.hyst_strain * self.l_backup))
            self.stick_max_backup = self.hyst_strain * self.l_backup

    # The stiffest an edge can get, springs and friction together. Sets the
    # largest frequency the mesh carries, and with it the time step.
    @property
    def stiffest_edge(self):
        k = self.k + self.k_hyst
        if (self.has_backup):
            k = k + self.k_backup + self.k_hyst_backup
        return np.max(k)

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

        # How far each friction slider has slipped - the whole load history
        self.slip = np.zeros(self.n_edges)
        self.slip_backup = np.zeros(self.n_edges)

        # What each edge is actually carrying, springs and friction together.
        # The friction is not a function of position, so this is the only place
        # it can be read off afterwards
        self.tension = np.zeros(self.n_edges)

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

        # Damping is the friction of the fibres sliding over each other inside
        # the weave, modelled as a Jenkins element in parallel with each spring:
        # a stiffness in series with a slider that lets go once it is asked to
        # hold more than hyst_tension. Loading leaves the slider pinned, so the
        # edge carries a little extra tension; reversing has to drag the slider
        # all the way back before it pins the other way, and that lag is what
        # opens a hysteresis loop and takes the energy out. See Iwan (1966).
        #
        # The loop area does not depend on how fast the cycle is run, which is
        # what real webbing does and what a dashpot cannot do: a dashpot damps in
        # proportion to frequency, so no single coefficient can suit both the
        # 0.3 Hz fall and the few hundred Hz the mesh rings at.
        #
        # hyst_tension is the tension the friction holds as a fraction of kl, so
        # a stiff backup gets proportionally more of it than a soft mainline.
        # hyst_strain is the stretch it takes to reverse the slider, which only
        # rounds the corners of the loop; hyst_tension is what sets the damping.
        self.hyst_tension = 0.006
        self.hyst_strain = 0.005

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
        

    # How much of a spring's stretch its friction is still holding on to. The
    # slider lets go as soon as it is asked for more than stick_max, and that
    # clip is the whole model - it is the return map of a perfectly plastic
    # element, and it needs no velocity and no notion of loading or unloading.
    def held_by_friction(self, stretch, slip, stick_max):
        return np.clip(stretch - slip, -stick_max, stick_max)

    # The tension the friction adds to a spring, which can be either sign.
    def friction_tension(self, stretch, slip, stick_max, k_hyst, beta):
        held = k_hyst * self.held_by_friction(stretch, slip, stick_max)

        # A slack spring has no tension pressing its fibres together, so nothing
        # rubs and it carries no friction. A taut one cannot be pulled below zero
        # tension either, because webbing still cannot push. That cap only bites
        # under a strain of hyst_tension, far below anything a rigged line sees.
        return np.where(stretch > 0, np.maximum(held, -beta), 0.0)

    # get_net_forces has to be called first
    def get_hysteresis_forces(self, after_break = True):
        extra = self.friction_tension(self.stretch, self.slip, self.stick_max,
                                      self.k_hyst, self.k * self.stretch)

        if (self.has_backup):
            extra_backup = self.friction_tension(
                self.stretch_backup, self.slip_backup, self.stick_max_backup,
                self.k_hyst_backup, self.backup)

            if after_break:
                extra = np.where(self.break_mainline, extra_backup, extra + extra_backup)
            else:
                extra = extra + extra_backup

        self.tension[:] = self.beta + extra

        scale = np.zeros(self.n_edges)
        np.divide(extra, self.dist_edge, out=scale, where=self.dist_edge > 0)

        # Same sign convention as the spring force it sits beside
        F = np.zeros_like(self.F)
        F[1:] = -self.d_edge * scale[:, None]
        F[:-1] += self.d_edge * scale[:, None]

        return F

    # Where each slider has slipped to. Advanced once per accepted time step,
    # which is why the solver takes fixed steps: an adaptive one retries steps
    # and interpolates between them, and the sliders would then follow the
    # solver's trials rather than the motion of the line.
    # get_net_forces has to be called first
    def advance_load_history(self):
        self.slip[:] = self.slipped_to(self.stretch, self.slip, self.stick_max)
        if (self.has_backup):
            self.slip_backup[:] = self.slipped_to(
                self.stretch_backup, self.slip_backup, self.stick_max_backup)

    # A spring that has gone slack lets its friction go with it, so it starts
    # clean when it next comes tight
    def slipped_to(self, stretch, slip, stick_max):
        return np.where(
            stretch > 0,
            stretch - self.held_by_friction(stretch, slip, stick_max),
            stretch,
        )

    # Let every slider go, so the line carries its springs and nothing else.
    # get_net_forces has to be called first
    def relax_friction(self):
        self.slip[:] = self.stretch
        self.slip_backup[:] = self.stretch_backup

    def get_hysteresis_forces_free(self, after_break = True):
        F = self.get_hysteresis_forces(after_break)

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
    
