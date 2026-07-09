from dataclasses import dataclass
from functools import cached_property

@dataclass
class webbing:
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
class segment:
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

# Webbings
joker = webbing(stretch_pct = 3.6,  tension_kN = 5, weight_g_m = 54) 
solid = webbing(stretch_pct = 2.5,  tension_kN = 5, weight_g_m = 50) 
y2k   = webbing(stretch_pct = 1.0,  tension_kN = 5, weight_g_m = 33) 

# Webbings and line
L = 275             # Line length [m]
pull_webbing = -1.0

# Segmented setup
# segs = [ segment(joker, solid, L, L_backup, False) ]
# segs = [ 
#         segment(joker, joker, L/4, L/4 + 1.5, False),
#         segment(joker, joker, L/4, L/4 + 1.5, False),  
#         segment(joker, joker, L/4, L/4 + 1.5, False),
#         segment(joker, joker, L/4, L/4 + 1.5, False),  
#        ]

segs = [ 
        segment(y2k, y2k, 50, 53, True),
        segment(y2k, y2k, 50, 53, False),
        segment(y2k, y2k, 50, 53, False),
        segment(y2k, y2k, 50, 53, False),
        segment(joker, joker, 25, 26, False),
       ]

# Experiment
break_mainline = False

# Slackliner
m_slackliner = 89  # Mass if slackliner [kg]
l_leash = 1.3      # Length of leash [m]
l_leg = 1.1        # Length of legs [m] (until harness connection point)

kl_leash = 200*1E3 # Spring constant times length - Leash

# Discretization
N = 101             # Discretization
x_slacker = 130  # x value of slackliner
zeta = 0.001        # Dampening parameter for linear dampening
detect_collision = True


# ODE setting
t0 = 0
t1 = 5
