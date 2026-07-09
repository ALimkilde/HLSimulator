import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import sys
from tqdm import tqdm

from simulation import simulate, add_tension, get_static_position, dofhandler, compute_tension_mainline
from ropeplayer import RopePlayer
 
from config import * # Change when we switch to segmented setups

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


def animate_rope(result, skip=500, keep_initial_in_background = True):
    fig, ax = plt.subplots(figsize=(16, 9))

    ax.set_xlim(-0.05 * L, 1.05 * L)
    ax.set_ylim(np.min(result["y"]), 0.01 * L)
    ax.set_aspect('equal')
    ax.grid(True)

    line, = ax.plot([], [], 'o-', lw=2, ms=4)
    line_slackliner, = ax.plot([], [], 'o-', lw=2, ms=4)

    plt.show(block=False)

    start_wall = time.perf_counter()
    start_sim = result["t"][0]

    f_w = result["f_webbing"]
    f_a1 = result["f_anchor1"]
    f_a2 = result["f_anchor2"]
    f_leash = result["f_leash"]

    for i in range(0, len(result["t"]), skip):
        t = result["t"][i]

        # Wait until wall clock matches simulation time
        while time.perf_counter() - start_wall < (t - start_sim):
            plt.pause(0.00001)

        Z = result["y"][:, i]

        x = Z[0:2*N:2]
        y = Z[1:2*N:2]


        line.set_data(x, y)
        if (result["backup_activated"][i]):
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

def plot_static_position(pos):
    plot_rope(pos, label = 'static pos')
    plt.title(f"Tension = {compute_tension_mainline(pos)/1000} kN")
    print(f"Tension = {compute_tension_mainline(pos)/1000} kN")

def main():

    
    result = simulate()

    # animate_rope(result)
    # player = RopePlayer(result)
    # plt.show()

if __name__ == "__main__":
    main()
