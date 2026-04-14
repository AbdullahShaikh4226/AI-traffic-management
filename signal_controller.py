"""
signal_controller.py
====================
4-way intersection signal controller.

Logic
-----
  - Only ONE lane is GREEN at a time (standard traffic rule).
  - Each lane gets a recommended green time based on its vehicle grade total:
        green_time = clamp(BASE + total_grade × SECONDS_PER_GRADE_PT,
                           MIN_GREEN, MAX_GREEN)
  - After a lane's green expires, the controller picks the next highest-priority
    lane that has waited at least MIN_RED_TIME (fairness guard).
  - Emergency override: if any lane detects an emergency vehicle, that lane
    immediately gets GREEN for EMERGENCY_OVERRIDE_TIME seconds regardless of
    the current active lane.

State per lane
--------------
  phase          : "GREEN" | "RED"
  green_time_rec : recommended green time (seconds)
  time_remaining : seconds left in current phase
  total_grade    : sum of vehicle grades in last frame
  vehicle_count  : number of vehicles in last frame
  emergency      : bool
  last_red_start : timestamp when this lane last went RED
"""

import time
import logging
from dataclasses import dataclass, field
from config import Config

log = logging.getLogger("SignalController")


@dataclass
class LaneState:
    lane_id        : int
    lane_name      : str
    phase          : str   = "RED"
    green_time_rec : float = 10.0
    time_remaining : float = 0.0
    total_grade    : int   = 0
    vehicle_count  : int   = 0
    emergency      : bool  = False
    last_red_start : float = field(default_factory=time.time)


class IntersectionSignalController:
    """
    Controls a 4-way (N-lane) intersection.
    Call  update(lane_detections)  every frame.
    """

    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.n      = len(cfg.LANE_NAMES)
        self._now   = time.time

        self.lanes: list[LaneState] = [
            LaneState(lane_id=i, lane_name=cfg.LANE_NAMES[i])
            for i in range(self.n)
        ]

        # Start with lane 0 GREEN
        self._active_lane     : int   = 0
        self._phase_start     : float = time.time()
        self._emergency_active: bool  = False
        self._emergency_lane  : int   = -1
        self._emergency_start : float = 0.0

        self.lanes[0].phase = "GREEN"
        log.info(f"Intersection started — {cfg.LANE_NAMES[0]} is GREEN first.")

    # ─────────────────────────────────────────────────────────────────────────
    def update(self, lane_detections: list[list[dict]]) -> list[LaneState]:
        """
        Args:
            lane_detections: list of length n_lanes.
                             Each element is the list of detection dicts
                             from that lane's camera frame.
        Returns:
            Updated list of LaneState (one per lane).
        """
        now = time.time()

        # ── Update per-lane metrics ───────────────────────────────────────────
        for i, dets in enumerate(lane_detections):
            lane = self.lanes[i]
            lane.total_grade   = sum(d["grade"] for d in dets)
            lane.vehicle_count = len(dets)
            lane.emergency     = any(d.get("emergency") for d in dets)

        # ── Emergency override check ──────────────────────────────────────────
        emergency_lanes = [i for i, l in enumerate(self.lanes) if l.emergency]

        if emergency_lanes and not self._emergency_active:
            # Pick highest-grade emergency lane if multiple
            best_em = max(
                emergency_lanes,
                key=lambda i: self.lanes[i].total_grade
            )
            self._trigger_emergency(best_em, now)

        if self._emergency_active:
            elapsed = now - self._emergency_start
            if elapsed >= self.cfg.EMERGENCY_OVERRIDE_TIME:
                self._end_emergency(now)
                log.info("Emergency override ended — resuming normal cycle.")

        # ── Normal cycle ──────────────────────────────────────────────────────
        if not self._emergency_active:
            active  = self.lanes[self._active_lane]
            elapsed = now - self._phase_start

            if elapsed >= active.green_time_rec:
                # This lane's green time is up → pick next
                self._rotate_to_next(now)

        # ── Recompute green time recommendations & time_remaining ─────────────
        for i, lane in enumerate(self.lanes):
            lane.green_time_rec = self._calc_green_time(lane.total_grade)
            if i == self._active_lane:
                if self._emergency_active:
                    rem = max(
                        0,
                        self.cfg.EMERGENCY_OVERRIDE_TIME - (now - self._emergency_start)
                    )
                else:
                    rem = max(0, lane.green_time_rec - (now - self._phase_start))
                lane.time_remaining = rem
            else:
                lane.time_remaining = 0.0

        return self.lanes

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _calc_green_time(self, total_grade: int) -> float:
        t = self.cfg.BASE_GREEN_TIME + total_grade * self.cfg.SECONDS_PER_GRADE_PT
        return float(max(self.cfg.MIN_GREEN_TIME, min(t, self.cfg.MAX_GREEN_TIME)))

    def _set_green(self, lane_idx: int, now: float):
        """Set one lane to GREEN, all others to RED."""
        for i, lane in enumerate(self.lanes):
            if i == lane_idx:
                lane.phase = "GREEN"
            else:
                if lane.phase == "GREEN":          # was previously green
                    lane.last_red_start = now
                lane.phase = "RED"

        self._active_lane = lane_idx
        self._phase_start = now

        lane_name = self.lanes[lane_idx].lane_name
        grade     = self.lanes[lane_idx].total_grade
        gt        = self.lanes[lane_idx].green_time_rec
        log.info(
            f"→ GREEN: {lane_name:6s} | "
            f"vehicles:{self.lanes[lane_idx].vehicle_count}  "
            f"grade:{grade}  green_time_rec:{gt:.1f}s"
        )

    def _rotate_to_next(self, now: float):
        """Pick the waiting lane with the highest total_grade."""
        candidates = []
        for i, lane in enumerate(self.lanes):
            if i == self._active_lane:
                continue
            waited = now - lane.last_red_start
            if waited >= self.cfg.MIN_RED_TIME:
                candidates.append(i)

        if not candidates:
            # No lane has waited long enough — just continue current green
            return

        # Highest grade wins; tie-break by longest wait
        best = max(
            candidates,
            key=lambda i: (self.lanes[i].total_grade, now - self.lanes[i].last_red_start)
        )

        # Mark current lane as RED
        self.lanes[self._active_lane].last_red_start = now
        self.lanes[self._active_lane].phase = "RED"

        self._set_green(best, now)

    def _trigger_emergency(self, lane_idx: int, now: float):
        self._emergency_active = True
        self._emergency_lane   = lane_idx
        self._emergency_start  = now
        self._set_green(lane_idx, now)
        log.warning(
            f"🚨 EMERGENCY OVERRIDE → {self.lanes[lane_idx].lane_name} "
            f"(lane {lane_idx}) — GREEN for {self.cfg.EMERGENCY_OVERRIDE_TIME}s"
        )

    def _end_emergency(self, now: float):
        self._emergency_active = False
        # Mark emergency lane as RED and rotate normally
        self.lanes[self._active_lane].last_red_start = now
        self.lanes[self._active_lane].phase = "RED"
        self._rotate_to_next(now)
