import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import numpy as np

from slackline_physics import project_along_y, interpolate

class RopePlayer:
    def __init__(self, result, model, fps=600):

        self.model = model
        N = model.N_line
                               
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

        self.ax.set_xlim(-0.05 * self.model.L, 1.05 * self.model.L)
        self.ax.set_ylim(np.min(result["y"][1:2*N+2:2]), np.max(result["y"][1:2*N+2:2]))
        self.ax.set_aspect("equal")
        self.ax.grid(True)


        self.line = LineCollection([], colors="blue")
        self.ax.add_collection(self.line)

        self.markers = self.ax.scatter([], [], color="black", s=20)
        
        self.line_slackliner, = self.ax.plot([], [], "o-", lw=2, ms=4)
        self.center_marker, = self.ax.plot(
            [], [],
            "o",
            color="green",   # center point color
            markersize=5,
            zorder=10,
        )

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
        N = self.model.N_line
        i = self.display_idx[display_frame]
    
        Z = self.result["y"][:, i]
    
        x = Z[0:2*N:2]
        y = Z[1:2*N:2]

        points = np.column_stack((x, y))
        segments = np.stack([points[:-1], points[1:]], axis=1)
        
        self.line.set_segments(segments)
        
        # bool vector of length N-1
        active = self.result["backup_activated_segments"][:,i]  # adjust name
        
        colors = np.where(active, "red", "blue")
        self.line.set_color(colors)

        pos = Z[:2*N].reshape(N, 2)
        start_slackliner = 2*N
        p_slacker = Z[start_slackliner:start_slackliner+2]
        proj, _, _, _, _ = project_along_y(p_slacker, pos)
        xs = [proj[0], p_slacker[0]]
        ys = [proj[1], p_slacker[1]]

        self.line_slackliner.set_data(xs, ys)
 
        self.markers.set_offsets(points)

        # Highlight center point
        center = N // 2
        self.center_marker.set_data([x[center]], [y[center]])

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
            self.frame = min(self.frame + 1, self.Nframes - 1)
            self.draw_frame(self.frame)

        elif event.key == "left":
            self.frame = max(self.frame - 1, 0)
            self.draw_frame(self.frame)

        elif event.key == "shift+right":
            self.frame = min(self.frame + 10, self.Nframes - 1)
            self.draw_frame(self.frame)

        elif event.key == "shift+left":
            self.frame = max(self.frame - 10, 0)
            self.draw_frame(self.frame)

        elif event.key == "home":
            self.frame = 0
            self.draw_frame(self.frame)

        elif event.key == "end":
            self.frame = self.Nframes - 1
            self.draw_frame(self.frame)
