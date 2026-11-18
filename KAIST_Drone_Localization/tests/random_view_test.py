"""Display and export paired camera views while matching in a worker thread."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from main import main

    raise SystemExit(main(["--random-test", *sys.argv[1:]]))

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import Button, TextBox

from localization.random_view import BlurConfig, RandomViewExperiment, ViewComparison, write_json
from visualization.map_viewer import draw_map, draw_pose

COLORS = ("#1756a1", "#bc2939", "#177344")


class RandomViewViewer:
    def __init__(self, experiment: RandomViewExperiment, blur: BlurConfig, interactive=True):
        self.experiment = experiment
        self.result = None
        self.figure = plt.figure(figsize=(13, 8))
        grid = self.figure.add_gridspec(3, 2, width_ratios=(1.25, 1),
                                      height_ratios=(1, 1, 0.61), hspace=0.40, wspace=0.16,
                                      left=0.08, right=0.95, top=0.92,
                                      bottom=0.16 if interactive else 0.08)
        self.map_axes = self.figure.add_subplot(grid[:2, 0])
        self.original_axes = self.figure.add_subplot(grid[0, 1])
        self.blurred_axes = self.figure.add_subplot(grid[1, 1])
        self.table_axes = self.figure.add_subplot(grid[2, :])
        self.table_axes.set_axis_off()
        self.heading = self.figure.suptitle("Random Camera View & Blur Robustness", fontsize=17)
        self.status = self.figure.text(0.08, 0.082 if interactive else 0.033, "", fontsize=9)
        self._draw_frames()
        if interactive:
            self.pool = ThreadPoolExecutor(max_workers=1)
            self.pending = None
            self.widgets = []
            self.seed_box = self._text(0.10, 0.135, "Seed ", str(experiment.seed))
            self.kernel_box = self._text(0.305, 0.055, "Kernel ", str(blur.kernel_size))
            self.sigma_box = self._text(0.43, 0.06, "Sigma ", str(blur.sigma))
            self.save_box = self._text(0.69, 0.16, "PNG ", "random_test.png")
            self._button(0.08, 0.105, "Random Test", self._random)
            self._button(0.193, 0.079, "Reseed", self._reseed)
            self._button(0.51, 0.105, "Apply Blur", self._reblur)
            self._button(0.86, 0.09, "Save Result", self._save)
            self.timer = self.figure.canvas.new_timer(interval=50)
            self.timer.add_callback(self._poll)
            self.figure.canvas.mpl_connect("close_event", self._close)
            self.timer.start()
            self._random()

    def _text(self, x, width, label, value):
        widget = TextBox(self.figure.add_axes([x, 0.113, width, 0.032]), label, initial=value)
        self.widgets.append(widget)
        return widget

    def _button(self, x, width, label, callback):
        button = Button(self.figure.add_axes([x, 0.027, width, 0.037]), label)
        button.on_clicked(callback)
        self.widgets.append(button)

    def _blur_settings(self):
        return BlurConfig(int(self.kernel_box.text), float(self.sigma_box.text))

    def _message(self, message):
        self.status.set_text(message)
        self.figure.canvas.draw_idle()

    def _submit(self, operation):
        if self.pending is not None:
            return
        try:
            blur = self._blur_settings()
            self.pending = operation(blur)
        except ValueError as exc:
            self._message(str(exc))
            return
        self._message("Running independent global matching...")

    def _random(self, _event=None):
        self._submit(lambda blur: self.pool.submit(self.experiment.run, blur))

    def _reseed(self, _event):
        def submit(blur):
            seed = int(self.seed_box.text)
            if seed < 0:
                raise ValueError("seed must be nonnegative")
            return self.pool.submit(self._restart, seed, blur)
        self._submit(submit)

    def _restart(self, seed, blur):
        self.experiment.reseed(seed)
        return self.experiment.run(blur)

    def _reblur(self, _event):
        if self.result is not None:
            self._submit(lambda blur: self.pool.submit(self.experiment.reblur, self.result, blur))

    def _poll(self):
        if self.pending is None or not self.pending.done():
            return
        finished, self.pending = self.pending, None
        try:
            self.show_result(finished.result())
        except (ValueError, cv2.error) as exc:
            self._message(str(exc))

    def _close(self, _event=None):
        self.timer.stop()
        self.pool.shutdown(wait=False, cancel_futures=True)

    def _save(self, _event):
        if self.pending is None and self.result is not None:
            try:
                path = self.save(self.save_box.text)
                self._message(f"Saved {path.name} and {path.with_suffix('.json').name}")
            except (OSError, ValueError) as exc:
                self._message(str(exc))

    def save(self, path, data_path=None):
        if self.result is None:
            raise ValueError("no result available to save")
        path = Path(path)
        if path.suffix.lower() != ".png":
            raise ValueError("save path must end in .png")
        data_path = Path(data_path) if data_path is not None else path.with_suffix(".json")
        if path.resolve() == data_path.resolve():
            raise ValueError("PNG and JSON paths must differ")
        self.figure.savefig(path, dpi=150)
        write_json(data_path, self.result.record())
        return path

    def _draw_frames(self, frames=(None, None)):
        variant_title = ("Blurred Camera View" if self.result is None or self.result.lighting_case is None
                         else self.result.variant_label + " view")
        for axes, title, frame in zip((self.original_axes, self.blurred_axes),
                                      ("Original Camera View", variant_title), frames):
            axes.clear()
            axes.set(title=title)
            axes.set_axis_off()
            if frame is not None:
                axes.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), interpolation="nearest")

    def show_result(self, result: ViewComparison):
        """Render a completed pair on the calling (GUI) thread."""
        self.result = result
        sample = result.sample
        truth = sample.pose
        axes = self.map_axes
        axes.clear()
        draw_map(axes, self.experiment.camera)
        footprint = sample.footprint_xy_m
        axes.add_patch(Polygon(footprint, closed=True, fill=False, edgecolor=COLORS[0],
                               linestyle="--", linewidth=1.5, label="Camera footprint"))
        campus = self.experiment.camera.raster.campus
        axes.set(xlim=(min(0, footprint[:, 0].min()) - 0.04,
                       max(campus.width_m, footprint[:, 0].max()) + 0.04),
                 ylim=(min(0, footprint[:, 1].min()) - 0.04,
                       max(campus.height_m, footprint[:, 1].max()) + 0.04))
        draw_pose(axes, truth.x_m, truth.y_m, truth.yaw_deg, color=COLORS[0],
                  label="Ground truth", marker="o", length=0.17, zorder=7,
                  s=185, linewidths=1.9, facecolors="none")
        for score, color, size, length, name in zip(
            (result.original, result.blurred), COLORS[1:], (125, 45), (0.135, 0.105),
            ("Original estimate", result.variant_label + " estimate"),
        ):
            estimate = score.estimate
            if estimate.success:
                axes.plot([truth.x_m, estimate.x_m], [truth.y_m, estimate.y_m],
                          color=color, alpha=0.6, linewidth=1)
                draw_pose(axes, estimate.x_m, estimate.y_m, estimate.yaw_deg, color=color,
                          label=name, marker="x", length=length, zorder=7, s=size, linewidths=1.9)
        axes.legend(loc="lower left", fontsize=8, framealpha=0.94)
        self._draw_frames((sample.frame, result.blurred_frame))
        self._draw_table(result)
        self._message(f"Seed {sample.seed}  |  View {sample.index}  |  Building coverage {sample.coverage:.1%}"
                      f"  |  Draws {sample.attempts}  |  Gaussian {result.blur.kernel_size} x "
                      f"{result.blur.kernel_size}, sigma {result.blur.sigma:g}")

    def _draw_table(self, result):
        axes = self.table_axes
        axes.clear()
        axes.set_axis_off()
        truth = result.sample.pose
        rows = [["Ground truth", f"({truth.x_m:.3f}, {truth.y_m:.3f})", f"{truth.yaw_deg:.1f}",
                 "—", "—", "—", "—", "—", "Reference"]]
        for name, score in (("Original", result.original), (result.variant_label, result.blurred)):
            estimate = score.estimate
            values = ([f"({estimate.x_m:.3f}, {estimate.y_m:.3f})", f"{estimate.yaw_deg:.1f}",
                       f"{score.position_error_m * 100:.2f}", f"{score.yaw_error_deg:.1f}"]
                      if estimate.success else ["N/A"] * 4)
            status = "SUCCESS" if estimate.success else f"FAILURE\n{estimate.reason}"
            rows.append([name, *values, f"{estimate.score:.3f}", f"{estimate.margin:.3f}",
                         f"{score.processing_ms:.1f}", status])
        table = axes.table(cellText=rows, colLabels=["View", "X, Y (m)", "Yaw (deg)",
                                                    "Error (cm)", "Yaw error (deg)", "IoU",
                                                    "Margin", "Match (ms)", "Status"],
                           colWidths=[0.12, 0.14, 0.08, 0.10, 0.12, 0.06, 0.07, 0.09, 0.22],
                           cellLoc="center", bbox=(0, 0, 1, 1))
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#d7dfe7")
            cell.set_facecolor("#eaf0f6" if row == 0 else "#ffffff")
            if row:
                cell.get_text().set_color(COLORS[row - 1])


