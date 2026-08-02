import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import sys
import yaml
from pathlib import Path
from tqdm import tqdm

from slackline_physics import SlacklineSpringModel, Webbing, Segment, Slackliner
from ropeplayer import RopePlayer
 

def plot_static_position(model, pos):
    plot_rope(pos, label = 'static pos')
    plt.title(f"Tension = {model.compute_tension_mainline(pos)/1000} kN")
    print(f"Tension = {model.compute_tension_mainline(pos)/1000} kN")


def state_at(model, result, idx, start = False):
    """Extract quantities at one time index."""

    h = result["y"][model.start_slackliner + 1, idx] - model.slackliner.l_leg

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
    s = state_at(model, result_leashfall, 0, start = True)

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
    s = state_at(model, result_backupfall, -1)

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

# Webbings
with open(Path(__file__).parent / "webbings.yaml") as f:
    webbings = {name: Webbing(**fields) for name, fields in yaml.safe_load(f).items()}

# Simulation settings
def load_settings(path):
    with open(path) as f:
        settings = yaml.safe_load(f)

    segs = [Segment(webbings[s["main"]], webbings[s["backup"]],
                    s["L_main"], s["L_backup"], s["break_mainline"])
            for s in settings["segments"]]

    slackliner = Slackliner(**settings["slackliner"])

    model = SlacklineSpringModel(
            slackliner = slackliner,
            segs = segs,
            **{k: settings[k] for k in
               ("L", "N", "T", "tension_kN", "pull_webbing", "pull_side")
               if k in settings},
            )

    return model, settings.get("plots", True)


def main(settings_path):

    model, plots = load_settings(settings_path)

    # TODO split into multiple calls
    result_leashfall, result_backupfall = model.simulate()
    
    table = summarize_results(model, result_leashfall, result_backupfall)
    # table.to_csv("results_more_tense_dont_detect.csv", index=False)
    with pd.option_context(
        "display.max_columns", None,
        "display.width", None,
    ):
        print(table)

    out = Path(settings_path).with_suffix(".txt")
    out.write_text(f"# Settings ({settings_path})\n"
                   f"{Path(settings_path).read_text()}\n"
                   f"# Results\n"
                   f"{table.to_string(index=False)}\n")
    print(f"Wrote {out}")

    if (not plots):
        return

    plt.plot(result_leashfall["t"], result_leashfall["f_leash"])
    plt.plot(result_backupfall["t"], result_backupfall["f_leash"])
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.grid(True)

    # animate_rope(result)
    player1 = RopePlayer(result_leashfall, model)

    if (result_backupfall is not None):
        player2 = RopePlayer(result_backupfall, model)

    plt.show()

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        sys.exit("usage: main.py <settings.yaml>")
    main(sys.argv[1])
