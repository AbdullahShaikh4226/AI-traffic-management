"""
main.py — Smart Traffic Management System (4-way intersection)
==============================================================
Runs one capture per lane in parallel, processes detections,
drives the signal controller, and displays a 2×2 tiled view.

Press Q to quit.
"""

import logging
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from config import Config
from detector import run_detection
from signal_controller import IntersectionSignalController
from overlay import draw_lane_frame, build_tile_grid
from logger import SessionLogger

# ─── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Main")


# ─────────────────────────────────────────────────────────────────────────────
def open_captures(sources: list) -> list[cv2.VideoCapture]:
    caps = []
    for src in sources:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video source: {src}\n"
                "Check VIDEO_SOURCES in config.py"
            )
        caps.append(cap)
    log.info(f"Opened {len(caps)} camera/video source(s).")
    return caps


# ─────────────────────────────────────────────────────────────────────────────
def main():
    cfg = Config()

    log.info(f"Emergency model : {cfg.EMERGENCY_MODEL_PATH}")
    log.info(f"General model   : {cfg.GENERAL_MODEL_PATH}")

    # ── Load models ───────────────────────────────────────────────────────────
    log.info("Loading models...")
    general_model   = YOLO(cfg.GENERAL_MODEL_PATH)
    emergency_model = YOLO(cfg.EMERGENCY_MODEL_PATH)
    log.info("Models loaded.")

    # ── Open video sources ────────────────────────────────────────────────────
    caps = open_captures(cfg.VIDEO_SOURCES)
    n    = len(caps)

    # ── Per-lane tracker state (ByteTrack persists across frames per model
    #    instance, but we need per-lane track-id → vehicle-type memory)
    tracker_states = [{} for _ in range(n)]

    # ── Controllers ───────────────────────────────────────────────────────────
    signal_ctrl  = IntersectionSignalController(cfg)
    session_log  = SessionLogger(cfg)

    # ── Per-lane flow metrics ─────────────────────────────────────────────────
    lane_vehicle_pools = [dict() for _ in range(n)]   # track_id → data
    start_time  = time.time()
    frame_idx   = 0

    log.info("Starting 4-way intersection — press Q to quit.")

    while True:
        frames     : list = []
        all_dets   : list = []
        lane_emergency: list[bool] = []
        any_feed_alive = False

        # ── Read one frame per lane ───────────────────────────────────────────
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                # Loop video for demo; replace with `break` for live cameras
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    frames.append(
                        _blank_frame(cfg.TILE_WIDTH, cfg.TILE_HEIGHT,
                                     cfg.LANE_NAMES[i])
                    )
                    all_dets.append([])
                    lane_emergency.append(False)
                    continue

            any_feed_alive = True

            # ── Detection ─────────────────────────────────────────────────────
            dets, em_flag = run_detection(
                frame,
                general_model,
                emergency_model,
                cfg,
                tracker_states[i],
            )

            # Track unique vehicles for flow rate
            for d in dets:
                tid = d["track_id"]
                if tid != -1:
                    lane_vehicle_pools[i][tid] = d

            frames.append(frame)
            all_dets.append(dets)
            lane_emergency.append(em_flag)

        if not any_feed_alive:
            log.info("All video sources ended.")
            break

        frame_idx += 1

        # ── Signal controller ─────────────────────────────────────────────────
        lane_states = signal_ctrl.update(all_dets)

        # ── Per-lane flow rates ───────────────────────────────────────────────
        elapsed_min = (time.time() - start_time) / 60.0
        flow_rates = [
            len(pool) / elapsed_min if elapsed_min > 0 else 0.0
            for pool in lane_vehicle_pools
        ]

        # ── Draw overlays ─────────────────────────────────────────────────────
        annotated = [
            draw_lane_frame(frames[i], all_dets[i], lane_states[i], flow_rates[i])
            for i in range(n)
        ]

        # ── Tile display ──────────────────────────────────────────────────────
        grid = build_tile_grid(
            annotated, lane_states, cfg.TILE_WIDTH, cfg.TILE_HEIGHT
        )
        cv2.imshow("4-Way Traffic Management — Q to quit", grid)

        # ── Logging ───────────────────────────────────────────────────────────
        if frame_idx % cfg.LOG_INTERVAL_FRAMES == 0:
            session_log.record(
                frame_idx  = frame_idx,
                lane_states= lane_states,
                flow_rates = flow_rates,
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            log.info("User quit.")
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
    session_log.save()


# ─────────────────────────────────────────────────────────────────────────────
def _blank_frame(w: int, h: int, label: str):
    """Black frame shown when a feed has ended."""
    import numpy as np
    f = __import__("numpy").zeros((h, w, 3), dtype="uint8")
    cv2.putText(f, f"{label}: feed ended", (20, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80,80,80), 2)
    return f


if __name__ == "__main__":
    main()
