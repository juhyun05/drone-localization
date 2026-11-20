"""Matplotlib view with click-to-capture and yaw adjustment."""

import math
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np

from localization.camera_simulator import Pose, SyntheticCamera
from localization.map_matching import MapMatcher, angle_error
from localization.tracking_manager import TrackingManager
from simulation.drone_trajectory import capture_sample
from simulation.playback import latest_due_index


def draw_map(axes, camera):
    raster = camera.raster
    width, height = raster.size
    left, top = raster.uv_to_xy((-0.5, -0.5))
    right, bottom = raster.uv_to_xy((width - 0.5, height - 0.5))
    axes.imshow(cv2.cvtColor(camera.scene, cv2.COLOR_BGR2RGB),
                extent=(left, right, bottom, top), interpolation="nearest")
    axes.set(xlim=(0, raster.campus.width_m), ylim=(0, raster.campus.height_m),
             xlabel="x east (m)", ylabel="y north (m)", title="KAIST model")


def draw_pose(axes, x_m, y_m, yaw_deg, *, color, label, marker, length=0.14, zorder=5,
              **marker_style):
    """Draw a position and its north-referenced heading on a map axes."""
    heading = math.radians(yaw_deg)
    marker_style.setdefault("edgecolors" if marker_style.get("facecolors") == "none"
                            else "color", color)
    axes.scatter(x_m, y_m, marker=marker, label=label, zorder=zorder, **marker_style)
    axes.quiver(x_m, y_m, -math.sin(heading) * length, math.cos(heading) * length,
                angles="xy", scale_units="xy", scale=1, color=color, zorder=zorder - 1)


def draw_footprint(axes, xy, yaw_deg, span_m, aspect, color):
    """Draw the planar footprint for a nadir camera, with aspect equal to height/width."""
    heading = math.radians(yaw_deg)
    cos_yaw, sin_yaw = math.cos(heading), math.sin(heading)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    corners = np.array([[-1, -aspect], [1, -aspect], [1, aspect], [-1, aspect],
                        [-1, -aspect]]) * (span_m / 2)
    points = corners @ rotation.T + xy
    axes.plot(*points.T, color=color, linewidth=1.2, alpha=0.85)


class MapViewer:
    def __init__(self, camera: SyntheticCamera, matcher: MapMatcher, pose: Pose,
                 interactive: bool = True):
        self.camera = camera
        self.matcher = matcher
        self.pose = pose
        self.figure, (self.map_axes, self.frame_axes) = plt.subplots(1, 2, figsize=(12, 6))
        self.figure.subplots_adjust(left=0.08, right=0.96, bottom=0.20, top=0.90, wspace=0.20)
        self.status = self.figure.text(0.08, 0.08, "", fontsize=10)
        if interactive:
            self.figure.text(0.08, 0.04, "Click the map to capture. Adjust yaw to rotate the camera.")
            self.yaw_slider = Slider(self.figure.add_axes([0.20, 0.13, 0.62, 0.025]),
                                     "Yaw (deg)", 0, 360, valinit=pose.yaw_deg % 360,
                                     dragging=False)
            self.yaw_slider.on_changed(self._change_yaw)
            self.figure.canvas.mpl_connect("button_press_event", self._click)
        self.update(pose)

    def _click(self, event):
        if event.inaxes is self.map_axes and event.button == 1:
            self.update(Pose(event.xdata, event.ydata, self.pose.yaw_deg))

    def _change_yaw(self, yaw):
        self.update(Pose(self.pose.x_m, self.pose.y_m, yaw))

    def update(self, pose: Pose):
        """Capture a new frame, estimate independently, then draw both poses."""
        self.pose = pose
        frame = self.camera.capture(pose)
        estimate = self.matcher.estimate(frame)
        self.map_axes.clear()
        self.frame_axes.clear()
        draw_map(self.map_axes, self.camera)
        draw_pose(self.map_axes, pose.x_m, pose.y_m, pose.yaw_deg,
                  color="#172459", label="Ground truth", marker="o")
        status = (f"Truth ({pose.x_m:.3f}, {pose.y_m:.3f}) m, {pose.yaw_deg:.1f} deg"
                  f"  |  IoU {estimate.score:.3f}, margin {estimate.margin:.3f}")
        if estimate.success:
            draw_pose(self.map_axes, estimate.x_m, estimate.y_m, estimate.yaw_deg,
                      color="#aa1c76", label="Estimated pose", marker="x")
            error = math.hypot(estimate.x_m - pose.x_m, estimate.y_m - pose.y_m)
            yaw_error = angle_error(estimate.yaw_deg, pose.yaw_deg)
            status += (f"\nEstimate ({estimate.x_m:.3f}, {estimate.y_m:.3f}) m, "
                       f"{estimate.yaw_deg:.1f} deg  |  Errors {error:.3f} m, {yaw_error:.1f} deg")
        else:
            status += f"\nPose rejected ({estimate.reason})"
        self.map_axes.legend(loc="lower right")
        self.frame_axes.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), interpolation="nearest")
        self.frame_axes.set_title("Synthetic camera frame")
        self.frame_axes.set_axis_off()
        self.status.set_text(status)
        self.figure.canvas.draw_idle()
        return estimate

class TrackingViewer:
    """Play timed captures with one worker and no backlog of pending frames."""

    def __init__(self, raster, config, matcher, samples, seed=42, initial_xy=None):
        self.raster, self.config, self.matcher = raster, config, matcher
        self.samples, self.seed, self.initial_xy = samples, seed, initial_xy
        self.timestamps = [sample.timestamp_s for sample in samples]
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.figure, (self.map_axes, frame_axes) = plt.subplots(1, 2, figsize=(12, 6.5))
        self.figure.subplots_adjust(bottom=0.24)
        self.status = self.figure.text(0.08, 0.10, "", fontsize=10)
        self.blank = np.zeros((config.height_px, config.width_px, 3), np.uint8)
        self.image = frame_axes.imshow(self.blank)
        frame_axes.set(title="Synthetic camera frame")
        frame_axes.set_axis_off()
        self.lines = [self.map_axes.plot([], [], marker=marker, markersize=3, color=color, label=label)[0]
                      for marker, color, label in (("o", "#182858", "Ground truth"),
                                                    ("x", "#ad246d", "Measurement"),
                                                    (".", "#007e72", "Filter estimate"))]
        self.play_button = Button(self.figure.add_axes([0.08, 0.025, 0.1, 0.055]), "Play")
        self.reset_button = Button(self.figure.add_axes([0.2, 0.025, 0.1, 0.055]), "Reset")
        self.play_button.on_clicked(self._toggle)
        self.reset_button.on_clicked(self._reset)
        self._reset()
        draw_map(self.map_axes, self.camera)
        self.map_axes.legend(loc="lower right")
        self.timer = self.figure.canvas.new_timer(interval=25)
        self.timer.add_callback(self._tick)
        self.figure.canvas.mpl_connect("close_event", self._close)
        self.timer.start()

    def _toggle(self, _event):
        if self.play_started is not None:
            self.elapsed += perf_counter() - self.play_started
            self.play_started = None
            self.play_button.label.set_text("Play")
        elif self.index < len(self.samples) - 1:
            self.play_started = perf_counter()
            self.play_button.label.set_text("Pause")
            self._tick()
        self.figure.canvas.draw_idle()

    def _reset(self, _event=None):
        self.play_started, self.elapsed = None, 0.0
        self.index, self.skipped = -1, 0
        self.draw_ms = 0.0
        self.camera = SyntheticCamera(self.raster, self.config, self.seed)
        self.tracker = TrackingManager(self.matcher, initial_xy=self.initial_xy)
        self.history = [[], [], []]
        for line in self.lines:
            line.set_data([], [])
        self.image.set_data(self.blank)
        self.status.set_text("UNINITIALIZED  |  Press Play to start")
        self.play_button.label.set_text("Play")
        self.figure.canvas.draw_idle()

    def _close(self, _event=None):
        self.timer.stop()
        self.pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _process(tracker, camera, sample):
        frame, timestamp = capture_sample(camera, sample)
        return frame, tracker.step(frame, timestamp)

    def _tick(self):
        if self.pending is not None:
            future, tracker, index = self.pending
            if not future.done():
                return
            self.pending = None
            frame, result = future.result()
            if tracker is self.tracker:
                self._render(self.samples[index], frame, result)
                if index == len(self.samples) - 1 and self.play_started is not None:
                    self._toggle(None)
        if self.play_started is None:
            return
        elapsed = self.elapsed + perf_counter() - self.play_started
        index = latest_due_index(self.timestamps, elapsed, self.index)
        if index is not None:
            self.skipped += index - self.index - 1
            self.index = index
            future = self.pool.submit(self._process, self.tracker, self.camera, self.samples[index])
            self.pending = future, self.tracker, index

    def _render(self, sample, frame, result):
        start = perf_counter()
        truth = (sample.pose.x_m, sample.pose.y_m)
        match = result.measurement
        observed = (match.x_m, match.y_m) if match and match.success else (np.nan, np.nan)
        filtered = result.xy_m if result.xy_m is not None else (np.nan, np.nan)
        for line, history, xy in zip(self.lines, self.history, (truth, observed, filtered)):
            history.append(xy)
            line.set_data(*zip(*history))
        error = f"{math.dist(truth, filtered):.3f} m" if result.xy_m is not None else "n/a"
        sigma = f"{result.sigma_m:.3f} m" if result.sigma_m is not None else "n/a"
        yaw = (f"{angle_error(sample.pose.yaw_deg, result.yaw_deg):.1f} deg"
               if result.yaw_deg is not None else "n/a")
        if result.yaw_deg is not None and not result.measurement_used:
            yaw += " (last observation)"
        observation = "fused" if result.measurement_used else result.reason
        self.image.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.status.set_text(
            f"t={sample.timestamp_s:.2f} s  |  {result.state.value} / {result.attempt}  |  {observation}\n"
            f"Position error {error}  |  yaw error {yaw}  |  position sigma {sigma}\n"
            f"Tracking {result.total_ms:.1f} ms  |  last GUI draw {self.draw_ms:.1f} ms"
            f"  |  skipped captures {self.skipped}"
        )
        self.figure.canvas.draw()
        self.draw_ms = (perf_counter() - start) * 1000
