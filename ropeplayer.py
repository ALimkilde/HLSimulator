import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from config import *
from simulation import dofhandler


class RopePlayer:
    def __init__(self, result, fps=60):
        self.result = result

        display_times = np.arange(result["t"][0],
                                  result["t"][-1],
                                  1/fps)

        self.display_idx = np.unique(
            np.searchsorted(result["t"], display_times)
        )

        self.Nframes = len(self.display_idx)

        self.frame = 0      # index into display_idx
        ...
        self.playing = False
        self.last_time = None

        self.fig, self.ax = plt.subplots(figsize=(16, 9))

        self.ax.set_xlim(-0.05 * L, 1.05 * L)
        self.ax.set_ylim(np.min(result["y"]), 0.01 * L)
        self.ax.set_aspect("equal")
        self.ax.grid(True)

        self.line, = self.ax.plot([], [], "o-", lw=2, ms=4)
        self.line_slackliner, = self.ax.plot([], [], "o-", lw=2, ms=4)

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # timer runs every few ms
        self.anim = FuncAnimation(
            self.fig,
            self.update,
            interval=1000/fps,
            cache_frame_data=True,
            save_count=self.Nframes,
        )

        self.draw_frame(0)

    def draw_frame(self, display_frame):
        i = self.display_idx[display_frame]
    
        Z = self.result["y"][:, i]
    
        x = Z[0:2*N:2]
        y = Z[1:2*N:2]

        self.line.set_data(x, y)

        self.line.set_color(
            "red" if self.result["backup_activated"][i] else "blue"
        )

        xs = [Z[2*i_leashring], Z[dofhandler.start_slackliner]]
        ys = [Z[2*i_leashring+1], Z[dofhandler.start_slackliner+1]]

        self.line_slackliner.set_data(xs, ys)

        t=self.result['t'][i]
        f_w = self.result["f_webbing"][i]
        f_a1 = self.result["f_anchor1"][i]
        f_a2 = self.result["f_anchor2"][i]
        f_leash = self.result["f_leash"][i]
        self.ax.set_title(f"t = {t:.1f}s, F_w = {f_w/1000:.1f}kN, F_l = {f_leash/1000:.1f}kN")

        self.fig.canvas.draw_idle()

    def update(self, _):
        if not self.playing:
            return

        if self.frame >= self.Nframes - 1:
            self.playing = False
            return

        
        self.frame += 1
        self.draw_frame(self.frame)

        # choose next timer interval from timestamps
        dt = self.result["t"][self.frame] - self.result["t"][self.frame-1]
        self.anim.event_source.interval = max(1, dt * 1000)

    def on_key(self, event):

        if event.key == " ":
            self.playing = not self.playing

        elif event.key == "right":
            self.frame = min(self.frame + 10, self.Nframes - 1)
            self.draw_frame(self.frame)

        elif event.key == "left":
            self.frame = max(self.frame - 10, 0)
            self.draw_frame(self.frame)

        elif event.key == "shift+right":
            self.frame = min(self.frame + 100, self.Nframes - 1)
            self.draw_frame(self.frame)

        elif event.key == "shift+left":
            self.frame = max(self.frame - 100, 0)
            self.draw_frame(self.frame)

        elif event.key == "home":
            self.frame = 0
            self.draw_frame(self.frame)

        elif event.key == "end":
            self.frame = self.Nframes - 1
            self.draw_frame(self.frame)
