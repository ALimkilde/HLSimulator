import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
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
        "left": result["f_anchor1"][idx] / 1000,   # N -> kN
        "right": result["f_anchor2"][idx] / 1000,
        "leash": result["f_leash"][idx] / 1000,
        "webbing": result["f_webbing"][idx] / 1000,
        "backup": bool(result["backup_activated"][idx]),
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
        "Situation": "Empty line",
        "Slackliner's height (m)": np.nan,
        "Tension - left side (kN)": s["standing"],
        "Tension - right side (kN)": s["standing"],
        "Tension - leash (kN)": np.nan,
        "Max force webbing (kN)": s["standing"],
        "Pull webbing (m)": model.pull_webbing,
        "Backup activated": False,
        })

    rows.append({
        "Situation": "Walking",
        "Slackliner's height (m)": s["height"],
        "Tension - left side (kN)": s["left"],
        "Tension - right side (kN)": s["right"],
        "Tension - leash (kN)": np.nan,
        "Max force webbing (kN)": s["webbing"],
        "Pull webbing (m)": model.pull_webbing,
        "Backup activated": s["backup"],
    })

    # ---------------------------------------------------------------
    # Leash fall (maximum leash force)
    # ---------------------------------------------------------------
    h = np.min(result_leashfall["y"][model.start_slackliner+1,:])
    f_a1 = np.max(result_leashfall["f_anchor1"])/1000
    f_a2 = np.max(result_leashfall["f_anchor2"])/1000
    f_leash = np.max(result_leashfall["f_leash"])/1000
    f_webbing = np.max(result_leashfall["f_webbing"])/1000

    rows.append({
        "Situation": "Leash fall",
        "Slackliner's height (m)": h,
        "Tension - left side (kN)": f_a1,
        "Tension - right side (kN)": f_a2,
        "Tension - leash (kN)": f_leash,
        "Max force webbing (kN)": f_webbing,
        "Pull webbing (m)": model.pull_webbing,
        "Backup activated": bool(np.any(result_leashfall["backup_activated"])),
    })

    # ---------------------------------------------------------------
    # Backup fall - impact (maximum leash force)
    # ---------------------------------------------------------------
    h = np.min(result_backupfall["y"][model.start_slackliner+1,:])
    f_a1 = np.max(result_backupfall["f_anchor1"])/1000
    f_a2 = np.max(result_backupfall["f_anchor2"])/1000
    f_leash = np.max(result_backupfall["f_leash"])/1000
    f_webbing = np.max(result_backupfall["f_webbing"])/1000

    rows.append({
        "Situation": "Backup fall",
        "Slackliner's height (m)": h,
        "Tension - left side (kN)": f_a1,
        "Tension - right side (kN)": f_a2,
        "Tension - leash (kN)": f_leash,
        "Max force webbing (kN)": f_webbing,
        "Pull webbing (m)": model.pull_webbing,
        "Backup activated": bool(np.any(result_backupfall["backup_activated"])),
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
        "Tension - left side (kN)": 2,
        "Tension - right side (kN)": 2,
        "Tension - leash (kN)": 2,
        "Max force webbing (kN)": 2,
        "Pull webbing (m)": 2,
    })

def save_gif(model, result_leashfall, result_backupfall, out, fps = 25, lead = 3):
    """Write a gif with the leash fall animated above the backup fall.

    The gif starts `lead` seconds before the fall, standing still on the line,
    so the viewer sees the starting point before anything happens.
    """

    results = [result_leashfall, result_backupfall]
    labels = ["Leash fall: ", "Backup fall: "]

    # the animations keep an equal aspect ratio, so a 100m spot is a thin strip
    fig, axes = plt.subplots(2, 1, figsize=(20, 8), layout="constrained")

    # times before 0 land on the first sample, so the countdown holds the
    # standing position
    times = np.arange(-lead, model.t1, 1/fps)

    players = [RopePlayer(r, model, ax=axes[i], times=times, label=labels[i])
               for i, r in enumerate(results)]

    # same y-range for both animations, so the two falls can be compared
    N = model.N
    N_leash = model.N_leash
    ymin = min(np.min(r["y"][1:2*N+2*N_leash:2]) for r in results)
    ymax = max(np.max(r["y"][1:2*N+2*N_leash:2]) for r in results)
    for p in players:
        p.ax.set_ylim(ymin, ymax)

    def update(k):
        for p in players:
            p.draw_frame(k)

    anim = FuncAnimation(fig, update, frames=len(times))

    pbar = tqdm(total=len(times), desc="Writing gif:")
    anim.save(out, writer=PillowWriter(fps=fps),
              progress_callback=lambda i, n: pbar.update(1))
    pbar.close()
    plt.close(fig)


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
               ("L", "N", "T", "tension_kN", "pull_webbing", "pull_side",
                "leash_fall_hanging")
               if k in settings},
            )

    return model, settings


def main(settings_path):

    model, settings = load_settings(settings_path)
    plots = settings.get("plots", True)
    gif = settings.get("gif", False)

    # TODO split into multiple calls
    result_leashfall, result_backupfall = model.simulate()
    
    table = summarize_results(model, result_leashfall, result_backupfall)
    # table.to_csv("results_more_tense_dont_detect.csv", index=False)
    with pd.option_context(
        "display.max_columns", None,
        "display.width", None,
    ):
        print(table)

    out = Path(settings_path).with_suffix(".csv")
    table.to_csv(out, index=False)
    print(f"Wrote {out}")

    if (gif):
        if (result_backupfall is None):
            print("No backup fall simulated, skipping gif")
        else:
            out = Path(gif) if isinstance(gif, str) else Path(settings_path).with_suffix(".gif")
            save_gif(model, result_leashfall, result_backupfall, out)
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
