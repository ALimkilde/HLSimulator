import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import sys
from tqdm import tqdm

from slackline_physics import SlacklineSpringModel, Webbing, Segment, Slackliner
from ropeplayer import RopePlayer
 
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

def plot_static_position(model, pos):
    plot_rope(pos, label = 'static pos')
    plt.title(f"Tension = {model.compute_tension_mainline(pos)/1000} kN")
    print(f"Tension = {model.compute_tension_mainline(pos)/1000} kN")


def state_at(result, idx, start = False):
    """Extract quantities at one time index."""

    h = result["y"][IDX_HEIGHT, idx] - l_leg

    return {
        "height": h,
        "distance": np.nan,
        "left": result["f_anchor1"][idx] / 1000,   # N -> kN
        "right": result["f_anchor2"][idx] / 1000,
        "leash": result["f_leash"][idx] / 1000,
        "standing": result["f_standing"] / 1000,
    }



def summarize_results(model, result_leashfall, result_backupfall):
    rows = []

    # ---------------------------------------------------------------
    # Walking (initial state)
    # ---------------------------------------------------------------
    s = state_at(result_leashfall, 0, start = True)

    w = result_leashfall["w_line"]
    print(f"Weight of line: {w}kg")

    rows.append({
        "Situation": "Standing",
        "Slackliner's height (m)": np.nan,
        "Distance from anchor": np.nan,
        "Tension - left side (kN)": s["standing"],
        "Tension - right side (kN)": s["standing"],
        "Tension - leash (kN)": np.nan,
        })

    rows.append({
        "Situation": "Walking",
        "Slackliner's height (m)": s["height"],
        "Distance from anchor": s["distance"],
        "Tension - left side (kN)": s["left"],
        "Tension - right side (kN)": s["right"],
        "Tension - leash (kN)": np.nan,
    })

    # ---------------------------------------------------------------
    # Leash fall (maximum leash force)
    # ---------------------------------------------------------------
    h = np.min(result_leashfall["y"][model.start_slackliner+1,:])
    f_a1 = np.max(result_leashfall["f_anchor1"])/1000
    f_a2 = np.max(result_leashfall["f_anchor2"])/1000
    f_leash = np.max(result_leashfall["f_leash"])/1000

    rows.append({
        "Situation": "Leash fall",
        "Slackliner's height (m)": h,
        "Distance from anchor": np.nan,
        "Tension - left side (kN)": f_a1,
        "Tension - right side (kN)": f_a2,
        "Tension - leash (kN)": f_leash,
    })

    # ---------------------------------------------------------------
    # Backup fall - impact (maximum leash force)
    # ---------------------------------------------------------------
    h = np.min(result_backupfall["y"][model.start_slackliner+1,:])
    f_a1 = np.max(result_backupfall["f_anchor1"])/1000
    f_a2 = np.max(result_backupfall["f_anchor2"])/1000
    f_leash = np.max(result_backupfall["f_leash"])/1000

    rows.append({
        "Situation": "Backup fall",
        "Slackliner's height (m)": h,
        "Distance from anchor": np.nan,
        "Tension - left side (kN)": f_a1,
        "Tension - right side (kN)": f_a2,
        "Tension - leash (kN)": f_leash,
    })

    # ---------------------------------------------------------------
    # Backup fall - settled (final state)
    # ---------------------------------------------------------------
    s = state_at(result_backupfall, -1)

    # rows.append({
    #     "Situation": "Backup fall - settled",
    #     "Slackliner's height (m)": s["height"],
    #     "Distance from anchor": s["distance"],
    #     "Tension - left side (kN)": s["left"],
    #     "Tension - right side (kN)": s["right"],
    #     "Tension - leash (kN)": np.nan,
    # })

    df = pd.DataFrame(rows)

    # nicer formatting
    return df.round({
        "Slackliner's height (m)": 2,
        "Distance from anchor": 2,
        "Tension - left side (kN)": 2,
        "Tension - right side (kN)": 2,
        "Tension - leash (kN)": 2,
    })

# TODO move to data file?
# Webbings
pinktube = Webbing(stretch_pct = 15.4,  tension_kN = 5, weight_g_m = 54) 
joker = Webbing(stretch_pct = 3.6,  tension_kN = 5, weight_g_m = 54) 
solid = Webbing(stretch_pct = 2.5,  tension_kN = 5, weight_g_m = 50) 
y2k   = Webbing(stretch_pct = 1.0,  tension_kN = 5, weight_g_m = 33) 

def main():


    segs = [ 
            Segment(joker, solid, 30, 32, True),
            Segment(joker, solid, 30, 32, False),
            Segment(joker, solid, 40, 43, False),
           ]

    # Slackliner
    slackliner = Slackliner(
            m = 89, 
            l_leg = 1.1, 
            l_leash = 1.3,
            x_coor = 50,
            )

    model = SlacklineSpringModel(
            L = 100,
            N = 101,
            slackliner = slackliner,
            segs = segs,
            T = 50,
            pull_webbing = 1
            )
    
    # TODO split into multiple calls
    result_leashfall, result_backupfall = model.simulate()
    
    # Example
    # table = summarize_results(model, result_leashfall, result_backupfall)
    # table.to_csv("results_more_tense_dont_detect.csv", index=False)
    # with pd.option_context(
    #     "display.max_columns", None,
    #     "display.width", None,
    # ):
    #     print(table)

    plt.plot(result_leashfall["t"], result_leashfall["f_leash"])
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.grid(True)

    # animate_rope(result)
    player1 = RopePlayer(result_leashfall, model)

    # if (result_backupfall is not None):
    #     player2 = RopePlayer(result_backupfall)

    plt.show()

if __name__ == "__main__":
    main()
