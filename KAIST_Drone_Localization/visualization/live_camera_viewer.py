"""Camera-only and localization viewers for live HTTP and recorded video."""

from collections import deque
from time import perf_counter
import math

import cv2
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

from localization.camera_interface import INPUT_GAP_S
from .map_viewer import draw_footprint, draw_map, draw_pose


class LiveCameraViewer:
    STATUS_REFRESH_S = 0.25

    def __init__(self, source, localizer=None, map_camera=None, *,
                 scheduler=None, performance=None, performance_monitor=False,
                 gui_interval_ms=50, geometry_note="", ground_truth=None, stale_s=INPUT_GAP_S,
                 result_age_s=5.0, nominal_span_m=None, physical_per_map=None):
        self.source, self.localizer, self.map_camera = source, localizer, map_camera
        self.scheduler, self.performance = scheduler, performance
        self.performance_monitor = performance_monitor
        self.ground_truth, self.stale_s = ground_truth, stale_s
        self.geometry_note = geometry_note
        self.result_age_s = result_age_s
        self.nominal_span_m = nominal_span_m
        self.physical_per_map = physical_per_map
        self.stream_status = getattr(source, "status", None)
        self.frame = self.result = None
        self.result_event = None
        self.path = deque(maxlen=300)
        self.displayed_frame = None
        self.displaying_diagnostic = False
        self.preview_frame = None
        self.displayed_pose_frame = None
        self.done = False
        self.error = ""
        self.closed = False
        self.timing_line = None
        graph = bool(performance_monitor and localizer)
        self.figure = plt.figure(figsize=(12 if localizer else 7, 9 if graph else 7.4))
        grid = self.figure.add_gridspec(3 if graph else 2, 2 if localizer else 1,
                                      height_ratios=(4, 3.5, 1.2) if graph else (4, 3 if localizer else 2),
                                      left=0.07, right=0.96, bottom=0.09, top=0.92, wspace=0.23)
        self.map_axes = self.figure.add_subplot(grid[:, 0]) if localizer else None
        camera_axes = self.figure.add_subplot(grid[0, -1])
        status_axes = self.figure.add_subplot(grid[1, -1])
        camera_axes.set_title("Camera view")
        camera_axes.set_axis_off()
        self.blank = np.zeros((240, 320, 3), np.uint8)
        self.image = camera_axes.imshow(self.blank, interpolation="nearest")
        status_axes.set_axis_off()
        # Socket errors can contain text localized by the operating system.
        installed_fonts = set(font_manager.get_font_names())
        status_font = next((family for family in ("Malgun Gothic", "Apple SD Gothic Neo",
                                                  "Noto Sans CJK KR", "NanumGothic")
                            if family in installed_fonts), plt.rcParams["font.family"][0])
        self.status = status_axes.text(0, 1, "Waiting for frames", va="top", fontsize=9,
                                       fontfamily=status_font)
        if graph:
            axes = self.figure.add_subplot(grid[2, -1])
            axes.set(xlim=(1, 60), ylim=(0, 1), ylabel="Process (ms)")
            axes.tick_params(labelsize=8)
            axes.grid(alpha=0.2)
            self.timing_line, = axes.plot([], [], color="#007b70", linewidth=1.4)
        if self.map_axes is not None:
            draw_map(self.map_axes, self.map_camera)
        canvas = self.figure.canvas
        self._blit_preview = canvas.supports_blit and not performance_monitor
        self._background = None
        self._status_background = None
        self._status_refresh_at = 0.0
        if self._blit_preview:
            self.image.set_animated(True)
            self.status.set_animated(True)
            canvas.mpl_connect("draw_event", self._cache_background)
            canvas.mpl_connect("resize_event", self._forget_background)
        self.timer = self.figure.canvas.new_timer(interval=gui_interval_ms)
        self.timer.add_callback(self.tick)
        self.figure.canvas.mpl_connect("close_event", self.close)
        self.timer.start()

    def tick(self):
        now = perf_counter()
        stats = self.stream_status() if self.stream_status else None
        self.preview_frame = self.source.peek()
        if self.scheduler is not None:
            self.done = self.scheduler.done.is_set()
            self.error = self.scheduler.error or getattr(self.source, "error", "")
            event = self.scheduler.latest_processed
            if event is not None and (not stats or stats.connection == event.frame.connection):
                frame = event.frame
                if self.frame is not None and (self.frame.connection != frame.connection
                                               or self.frame.image.shape != frame.image.shape
                                               or self.frame.input_epoch != frame.input_epoch):
                    self.path.clear()
                self.frame, self.result = frame, event.result
                self.result_event = event
        started = perf_counter()
        self._render(now, stats)
        if self.performance_monitor and self.performance is not None:
            self.performance.add_gui_time((perf_counter() - started) * 1000)

    def _render(self, now, stats):
        frame = self.frame
        last_age = (max(0.0, now - stats.last_received_at)
                    if stats and stats.last_received_at is not None else None)
        result_age = (max(0.0, now - frame.received_at)
                      if frame and frame.received_at is not None else None)
        connected = (not stats or (stats.state == "streaming" and last_age is not None
                                   and last_age < self.stale_s))
        recent_result = result_age is None or result_age < self.result_age_s or (not stats and self.done)
        current = bool(frame is not None and connected and not self.error
                       and recent_result and (not stats or frame.connection == stats.connection))
        if current and stats and self.preview_frame and self.preview_frame.connection == stats.connection:
            current = self.preview_frame.image.shape == frame.image.shape
        if not connected:
            self.path.clear()
        mode = stats.state if stats else ("end of file" if self.done else "video")
        lines = [f"Input: {mode}" + (" (stale)" if not connected and mode == "streaming" else "")]
        if stats:
            drops = "unknown" if stats.source_gaps is None else str(stats.source_gaps)
            lines.append(f"Camera {stats.receive_fps:.1f} FPS"
                         f"  |  Sender gaps {drops}  |  Bad JPEG {stats.decode_errors}")
            if last_age is not None:
                lines.append(f"Time since latest receive {last_age:.2f} s")
            if stats.error:
                lines.append(stats.error[:85])
        if self.error:
            lines.append(self.error[:85])
        diagnostic = self.result.diagnostic if current and self.result is not None else None
        shown = self.preview_frame if self.preview_frame is not None else frame
        if diagnostic is not None:
            shown = frame
        if (not connected or self.error or
                (stats and shown and diagnostic is None and (shown.connection != stats.connection
                                                             or now - shown.received_at >= self.stale_s))):
            shown = None
        changed = shown is not self.displayed_frame or (diagnostic is not None) != self.displaying_diagnostic
        if changed:
            self.displayed_frame = shown
            self.displaying_diagnostic = diagnostic is not None
            pixels = diagnostic if diagnostic is not None else shown.image if shown else None
            color = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB) if pixels is not None else self.blank
            previous_shape = self.image.get_array().shape[:2]
            self.image.set_data(color)
            height, width = color.shape[:2]
            if color.shape[:2] != previous_shape:
                self.image.set_extent((-0.5, width - 0.5, height - 0.5, -0.5))
                self._forget_background()
        if shown:
            height, width = shown.image.shape[:2]
            lines.append(f"Frame #{shown.frame_id}  |  {width} x {height}  |  Decode {shown.decode_ms:.1f} ms")
        if self.localizer:
            if self.performance is not None:
                fps, skipped, stale, global_ms, local_ms, recent = self.performance.display(
                    now, stale_s=self.stale_s)
                lines.append(f"Localization {fps:.1f} FPS  |  Skipped {skipped}  |  Stale {stale}")
                lines.append(f"Global {self._milliseconds(global_ms)}  |  "
                             f"Local {self._milliseconds(local_ms)}")
                latest = self.scheduler.latest if self.scheduler is not None else None
                if latest is not None and latest.status == "stale":
                    lines.append(f"Rejected frame #{latest.frame.frame_id}  |  "
                                 f"Frame age {latest.frame_age_ms:.1f} ms")
                if self.timing_line is not None and recent:
                    self.timing_line.set_data(range(61 - len(recent), 61), recent)
                    self.timing_line.axes.set_ylim(0, max(1.0, max(recent) * 1.1))
            if not stats:
                receive_fps = getattr(self.source, "receive_fps", 0)
                lines.append(f"Replay input {receive_fps:.1f} FPS")
            pose_frame = frame if current and self.result is not None else None
            pose_changed = pose_frame is not self.displayed_pose_frame
            if pose_changed:
                self.displayed_pose_frame = pose_frame
                for artist in self._pose_artists():
                    artist.remove()
            if self.geometry_note:
                lines.append(self.geometry_note)
            if pose_frame is not None:
                event = self.result_event
                age = event.frame_age_ms if event is not None and event.frame is frame else None
                lines.append(f"Localization frame #{frame.frame_id}  |  Frame age {self._milliseconds(age)}")
                lighting = self.result.lighting
                if lighting is not None:
                    score_mode = "Weighted IoU" if lighting.shadow_aware else "IoU"
                    lines.append(f"Lighting {lighting.method}  |  {score_mode}  |  "
                                 f"Prep {lighting.preprocess_ms:.1f} ms  |  Match {lighting.match_ms:.1f} ms")
                    if lighting.shadow_ratio is not None:
                        lines.append(f"Effective coverage {lighting.effective_coverage:.0%}"
                                     f"  |  Shadow ratio {lighting.shadow_ratio:.0%}")
                self._render_pose(lines, self.result, frame.frame_id, pose_changed)
                if self.result.measurement is None and event is not None and self.scheduler is not None:
                    attempted = self.scheduler.latest_attempt
                    if attempted is not None and attempted[0] == event.continuity and attempted[1] <= frame.frame_id:
                        _, _, estimate = attempted
                        lines.append(f"Last search {estimate.reason}  |  IoU {estimate.score:.3f}")
                if self._blit_preview and pose_changed:
                    for artist in self._pose_artists():
                        artist.set_animated(True)
            else:
                lines.append("No current localization result")
        self.status.set_text("\n".join(lines))
        self._draw(now)

    def _pose_artists(self):
        if self.map_axes is None:
            return ()
        return (*self.map_axes.lines, *self.map_axes.collections)

    def _draw_animated(self, now):
        canvas = self.figure.canvas
        if self._status_background is None or now >= self._status_refresh_at:
            canvas.restore_region(self._background)
            self.figure.draw_artist(self.status)
            self._status_background = canvas.copy_from_bbox(self.figure.bbox)
            self._status_refresh_at = now + self.STATUS_REFRESH_S
        else:
            canvas.restore_region(self._status_background)
        self.figure.draw_artist(self.image)
        for artist in sorted(self._pose_artists(), key=lambda artist: artist.get_zorder()):
            self.figure.draw_artist(artist)

    def _cache_background(self, event):
        if event.canvas.is_saving():
            return
        self._background = event.canvas.copy_from_bbox(self.figure.bbox)
        self._status_background = None
        self._draw_animated(perf_counter())

    def _forget_background(self, _event=None):
        self._background = None
        self._status_background = None

    def _draw(self, now):
        canvas = self.figure.canvas
        if self.performance_monitor:
            canvas.draw()
        elif self._background is not None:
            self._draw_animated(now)
            canvas.blit(self.figure.bbox)
        else:
            canvas.draw_idle()

    @staticmethod
    def _milliseconds(value):
        return f"{value:.1f} ms" if value is not None else "—"

    def _render_pose(self, lines, result, frame_id, draw):
        estimate, tracking = result.measurement, result.tracking
        state = tracking.state.value if tracking else "GLOBAL"
        reason = tracking.reason if tracking else estimate.reason
        lines.append(f"{state}  |  {reason}  |  Processing {result.processing_ms:.1f} ms")
        if estimate:
            show = lambda value: f"{value:.3f}" if value is not None else "—"
            lines.append(f"Match {estimate.score:.3f}  |  Margin {estimate.margin:.3f}"
                         f"  |  Building {show(estimate.building_score)}  |  Road {show(estimate.road_score)}")
            if estimate.road_coverage is not None:
                lines.append(f"Effective coverage B {estimate.building_coverage:.1%}"
                             f"  |  R {estimate.road_coverage:.1%}"
                             f"  |  Road mask {self._milliseconds(result.road_ms)}")
        if tracking:
            xy, yaw = result.projected_xy_m or tracking.xy_m, tracking.yaw_deg
            span = tracking.span_m
            used = tracking.measurement_used
            label = tracking.observation_source or state
            if tracking.sigma_m is not None:
                lines.append(f"Position uncertainty (σ) {tracking.sigma_m:.3f} m")
            if result.projected_xy_m is not None:
                label += " (output-time XY)"
        else:
            used = estimate.success
            xy = (estimate.x_m, estimate.y_m) if used else None
            yaw = estimate.yaw_deg
            span = estimate.span_m
            label = estimate.observation_source or "Measurement"
        if xy is None:
            return
        if draw:
            if not self.path or self.path[-1][0] != frame_id:
                self.path.append((frame_id, xy))
            xs, ys = zip(*(point for _, point in self.path))
            color = "#007b70" if used else "#b35900"
            self.map_axes.plot(xs, ys, color="#007b70", linewidth=1.3, alpha=0.65)
            draw_pose(self.map_axes, *xy, yaw, color=color, label=label, marker="o")
            if used and tracking:
                self.map_axes.plot(estimate.x_m, estimate.y_m, marker="+", color="#264f83",
                                   linestyle="none", markersize=8)
            if used and span is not None and result.frame_aspect is not None:
                draw_footprint(self.map_axes, tracking.xy_m if tracking else xy, yaw,
                               span, result.frame_aspect, color)
        yaw_label = "Yaw" if used else "Last observed yaw"
        lines.append(f"{label} ({xy[0]:.3f}, {xy[1]:.3f}) m  |  {yaw_label} {yaw:.1f}°")
        if span is not None:
            detail = f"Span {span:.3f} map m" if used else f"Last span {span:.3f} map m"
            if self.nominal_span_m is not None:
                detail += f"  |  Scale {span / self.nominal_span_m:.2f}x"
            if self.physical_per_map is not None:
                detail += f"  |  {span * self.physical_per_map * 100:.1f} cm on print"
            lines.append(detail)
        if used and self.ground_truth:
            error = math.dist((estimate.x_m, estimate.y_m), self.ground_truth)
            lines.append(f"Measurement error at fixed reference: {error:.3f} m")

    def close(self, _event=None):
        if self.closed:
            return
        self.closed = True
        self.timer.stop()
