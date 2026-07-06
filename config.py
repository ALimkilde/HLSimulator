# Experiment
break_mainline = True

# Slackliner
m_slackliner = 89  # Mass if slackliner [kg]
l_leash = 1.3      # Length of leash [m]
l_leg = 1.1        # Length of legs [m] (until harness connection point)

# Webbings and line
# length
L = 60             # Line length [m]
L_backup = L + 2.5 - 1.25

# weight
rho = 0.055        # Density [kg/m] (main) - Joker 
rho_backup = 0.050 # Density [kg/m] (main) - Mamba
# rho = 0.033      # Density [kg/m] (main) - Y2K
# rho_backup = 0.033      # Density [kg/m] (backup) - Y2K

# elasticity
kl = 139*1E3       # Spring constant times length - Joker
kl_backup = 139*1E3       # Spring constant times length - Solid
# kl = 500*1E3     # Spring constant times length - Y2K
kl_leash = 400*1E3 # Spring constant times length - Leash

# Discretization
N = 101             # Discretization
i_leashring = int(N/2)  # id of pt with slackliner hanging/standing
zeta = 0.0055        # Dampening parameter for linear dampening
detect_collision = True


# ODE setting
t0 = 0
t1 = 4.5
