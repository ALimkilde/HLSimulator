import numpy as np

class SlacklineRHS:

    def __init__(
        self,
        N,
        dofhandler,
        kl,
        l,
        kl_backup,
        l_backup,
        break_mainline,
        m,
        g,
        rho_air,
        C_D,
        webbing_width,
        update_every,
        pbar,
        project_along_y,
        kl_leash,
        l_leash,
        m_slackliner,
    ):

        self.N = N
        self.dofhandler = dofhandler

        # Store references
        self.l = l
        self.l_backup = l_backup
        self.break_mainline = break_mainline

        self.g = g
        self.m = m
        self.project_along_y = project_along_y

        self.kl_leash = kl_leash
        self.l_leash = l_leash
        self.m_slackliner = m_slackliner

        self.update_every = update_every
        self.pbar = pbar

        # Precompute constants
        self.k = kl / l
        self.k_backup = kl_backup / l_backup

        self.gravity_force = m[:, None] * g
        self.drag_constant = (
            0.5 * rho_air * C_D * webbing_width / 2
        )

        # Progress bookkeeping
        self.last_t = 0.0
        self.last_update = 0.0

        # Workspace
        n_edges = N - 1
        n_masses = N - 2

        self.d_edge = np.empty((n_edges, 2))
        self.dist_edge = np.empty(n_edges)

        self.stretch = np.empty(n_edges)
        self.backup = np.empty(n_edges)
        self.beta = np.empty(n_edges)
        self.scale = np.empty(n_edges)

        self.F = np.empty((n_masses, 2))
        self.drag_coef = np.empty(n_masses)
        self.vel_norm = np.empty(n_masses)

    def __call__(self, t, Z):

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
            self.dofhandler.offset:
            self.dofhandler.offset + 2*self.N
        ].reshape(self.N, 2)

        out[:self.dofhandler.offset] = Z[self.dofhandler.offset:]

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

        if self.dofhandler.with_slackliner:

            z_slack = Z[
                self.dofhandler.start_slackliner:
                self.dofhandler.start_slackliner+2
            ]

            proj, dist, i_prev, i_next, alpha = (
                self.project_along_y(z_slack, pos)
            )

            d = z_slack - proj

            if dist > self.l_leash:

                beta = (dist-self.l_leash)/self.l_leash

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
            self.dofhandler.offset+2:
            self.dofhandler.offset+2*(self.N-1)
        ] = acc.ravel()

        ########################################################
        # Slackliner equation
        ########################################################

        if self.dofhandler.with_slackliner:

            i = self.dofhandler.start_slackliner

            vel_slack = Z[
                i+self.dofhandler.offset:
                i+self.dofhandler.offset+2
            ]

            vel_norm = np.sqrt(
                vel_slack[0]**2 +
                vel_slack[1]**2
            )

            d = proj - z_slack

            if dist > self.l_leash:

                beta = (dist-self.l_leash)/self.l_leash

                F_slack = (
                    self.m_slackliner*self.g
                    + self.kl_leash*beta*d/dist
                )

            else:

                F_slack = self.m_slackliner*self.g

            out[
                i+self.dofhandler.offset:
                i+self.dofhandler.offset+2
            ] = F_slack/self.m_slackliner

        return out
