"""
logger.py
=========
Dual-output session logger.

Every LOG_INTERVAL_FRAMES frames:
  • Prints a formatted signal summary to console
  • Appends a row to in-memory record list

On session end (save()):
  • Writes  logs/session_<timestamp>.json
  • Writes  logs/session_<timestamp>.csv
  • Prints a final summary table to console
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

from signal_controller import LaneState
from config import Config

log = logging.getLogger("SessionLogger")


class SessionLogger:
    def __init__(self, cfg: Config):
        self.cfg         = cfg
        self._records    : list[dict] = []
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.n_lanes     = len(cfg.LANE_NAMES)

        out = Path(cfg.LOG_OUTPUT_DIR)
        out.mkdir(parents=True, exist_ok=True)

        self.json_path = out / f"session_{self._session_id}.json"
        self.csv_path  = out / f"session_{self._session_id}.csv"

    # ─────────────────────────────────────────────────────────────────────────
    def record(
        self,
        frame_idx  : int,
        lane_states: list[LaneState],
        flow_rates : list[float],
    ):
        """Call every LOG_INTERVAL_FRAMES frames."""
        ts = datetime.now().isoformat(timespec="seconds")

        row: dict = {
            "frame"    : frame_idx,
            "timestamp": ts,
        }

        has_emergency = False
        active_lane   = "—"

        for ls, fr in zip(lane_states, flow_rates):
            pfx = ls.lane_name
            row[f"{pfx}_phase"]          = ls.phase
            row[f"{pfx}_vehicles"]       = ls.vehicle_count
            row[f"{pfx}_grade"]          = ls.total_grade
            row[f"{pfx}_green_time_rec"] = round(ls.green_time_rec, 2)
            row[f"{pfx}_time_remaining"] = round(ls.time_remaining, 2)
            row[f"{pfx}_flow_vpm"]       = round(fr, 3)
            row[f"{pfx}_emergency"]      = ls.emergency

            if ls.phase == "GREEN":
                active_lane = ls.lane_name
            if ls.emergency:
                has_emergency = True

        row["any_emergency"] = has_emergency
        row["active_lane"]   = active_lane

        self._records.append(row)

        # ── Console print ─────────────────────────────────────────────────────
        self._console_print(frame_idx, lane_states, active_lane, has_emergency)

    # ─────────────────────────────────────────────────────────────────────────
    def _console_print(
        self,
        frame_idx   : int,
        lane_states : list[LaneState],
        active_lane : str,
        emergency   : bool,
    ):
        em_tag = "  🚨 EMERGENCY" if emergency else ""
        print(f"\n[Frame {frame_idx:05d}] Active: {active_lane}{em_tag}")
        print(f"  {'Lane':<8} {'Phase':<12} {'Veh':>4} {'Grade':>6} "
              f"{'Green Rec':>10} {'Remaining':>10}")
        print("  " + "-"*56)
        for ls in lane_states:
            em = " (!)" if ls.emergency else ""
            print(
                f"  {ls.lane_name:<8} {ls.phase:<12} {ls.vehicle_count:>4} "
                f"{ls.total_grade:>6} {ls.green_time_rec:>9.1f}s "
                f"{ls.time_remaining:>9.1f}s{em}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    def save(self):
        if not self._records:
            log.warning("No records to save.")
            return

        # ── JSON ──────────────────────────────────────────────────────────────
        with open(self.json_path, "w") as f:
            json.dump(
                {
                    "session_id"  : self._session_id,
                    "n_lanes"     : self.n_lanes,
                    "total_logged": len(self._records),
                    "records"     : self._records,
                },
                f, indent=2
            )

        # ── CSV ───────────────────────────────────────────────────────────────
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._records[0].keys())
            writer.writeheader()
            writer.writerows(self._records)

        # ── Final summary ─────────────────────────────────────────────────────
        em_count  = sum(1 for r in self._records if r["any_emergency"])
        avg_flows = {}
        for name in self.cfg.LANE_NAMES:
            key = f"{name}_flow_vpm"
            vals= [r[key] for r in self._records if key in r]
            avg_flows[name] = sum(vals)/len(vals) if vals else 0.0

        print("\n" + "="*58)
        print("  SESSION SUMMARY")
        print("="*58)
        print(f"  Session ID       : {self._session_id}")
        print(f"  Frames logged    : {len(self._records)}")
        print(f"  Emergency frames : {em_count}")
        print(f"  Avg flow by lane :")
        for name, fr in avg_flows.items():
            print(f"    {name:<8} : {fr:.2f} VPM")
        print(f"\n  JSON → {self.json_path}")
        print(f"  CSV  → {self.csv_path}")
        print("="*58 + "\n")
