"""
overlay.py
==========
  draw_lane_frame()   — draws boxes + per-lane signal HUD on one camera frame
  build_tile_grid()   — stitches 4 lane frames into a 2×2 display grid
"""

import cv2
import numpy as np
from signal_controller import LaneState


# ─── Palette ─────────────────────────────────────────────────────────────────
CLR_BG        = (18, 18, 18)
CLR_WHITE     = (240, 240, 240)
CLR_GREEN     = (60,  210, 90)
CLR_RED       = (55,   55, 220)
CLR_YELLOW    = (40,  210, 230)
CLR_EMERGENCY = (30,   30, 235)
CLR_VEHICLE   = (55,  195, 100)
CLR_EM_BOX    = (30,   50, 230)

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────────────────────────────────────
def draw_lane_frame(
    frame      : np.ndarray,
    detections : list[dict],
    lane_state : LaneState,
    flow_rate  : float,
) -> np.ndarray:
    """Draw bounding boxes and per-lane HUD onto a single camera frame."""
    frame = _draw_boxes(frame, detections)
    frame = _draw_hud(frame, lane_state, flow_rate)
    if lane_state.emergency:
        frame = _draw_emergency_stripe(frame)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
def build_tile_grid(
    frames     : list[np.ndarray],
    lane_states: list[LaneState],
    tile_w     : int,
    tile_h     : int,
) -> np.ndarray:
    """
    Resize 4 frames to tile_w×tile_h and arrange in a 2×2 grid.
    Returns a single (2*tile_h) × (2*tile_w) BGR image.
    """
    assert len(frames) == 4, "build_tile_grid expects exactly 4 frames"

    resized = [cv2.resize(f, (tile_w, tile_h)) for f in frames]

    # Draw lane name + phase badge on each tile corner
    for i, (img, ls) in enumerate(zip(resized, lane_states)):
        _draw_tile_badge(img, ls)

    top    = np.hstack(resized[:2])
    bottom = np.hstack(resized[2:])
    return np.vstack([top, bottom])


# ─────────────────────────────────────────────────────────────────────────────
#  Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_boxes(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        is_em  = det.get("emergency", False)
        color  = CLR_EM_BOX if is_em else CLR_VEHICLE
        thick  = 3 if is_em else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        if is_em:
            label = f"!! {det['vehicle_type'].upper()}"
        else:
            label = f"ID:{det['track_id']} {det['vehicle_type']} G:{det['grade']}"

        (tw, th), _ = cv2.getTextSize(label, FONT_SMALL, 0.48, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 3),
                    FONT_SMALL, 0.48, CLR_BG if is_em else CLR_WHITE, 1)
    return frame


def _draw_hud(
    frame     : np.ndarray,
    ls        : LaneState,
    flow_rate : float,
) -> np.ndarray:
    """Draws the signal HUD panel top-left of the frame."""
    PX, PY, PW, PH = 8, 8, 310, 200

    overlay = frame.copy()
    cv2.rectangle(overlay, (PX, PY), (PX + PW, PY + PH), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Lane name
    cv2.putText(frame, f"{ls.lane_name.upper()} LANE",
                (PX + 10, PY + 28), FONT, 0.7, CLR_WHITE, 2)

    cv2.line(frame, (PX+8, PY+36), (PX+PW-8, PY+36), (70,70,70), 1)

    # Phase indicator
    phase_color = {
        "GREEN": CLR_GREEN,
        "RED"  : CLR_RED,
    }.get(ls.phase, CLR_EMERGENCY)
    phase_label = "!! EMERGENCY" if ls.emergency else ls.phase

    cv2.putText(frame, f"Signal : {phase_label}",
                (PX+10, PY+60), FONT_SMALL, 0.6, phase_color, 2)

    # Time-remaining bar
    bx, by = PX + 10, PY + 72
    bw, bh = PW - 20, 10
    total_t = ls.green_time_rec if ls.phase == "GREEN" else 1.0
    ratio   = 1 - (ls.time_remaining / total_t) if total_t > 0 else 0
    ratio   = max(0, min(ratio, 1))

    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (50,50,50), -1)
    cv2.rectangle(frame, (bx, by), (bx + int(bw*ratio), by+bh), phase_color, -1)

    cv2.putText(frame, f"{ls.time_remaining:.1f}s left",
                (bx, by + bh + 16), FONT_SMALL, 0.52, CLR_WHITE, 1)

    # Metrics
    lines = [
        (f"Vehicles    : {ls.vehicle_count}",          CLR_WHITE),
        (f"Grade total : {ls.total_grade}",             CLR_YELLOW),
        (f"Rec. green  : {ls.green_time_rec:.1f}s",     CLR_GREEN),
        (f"Flow rate   : {flow_rate:.2f} VPM",          CLR_WHITE),
    ]
    for i, (txt, clr) in enumerate(lines):
        cv2.putText(frame, txt, (PX+10, PY+112 + i*22),
                    FONT_SMALL, 0.53, clr, 1)

    return frame


def _draw_emergency_stripe(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-48), (w, h), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    txt = "  EMERGENCY OVERRIDE ACTIVE  "
    (tw, _), _ = cv2.getTextSize(txt, FONT_SMALL, 0.6, 2)
    cv2.putText(frame, txt, ((w-tw)//2, h-14),
                FONT_SMALL, 0.6, CLR_WHITE, 2)
    return frame


def _draw_tile_badge(frame: np.ndarray, ls: LaneState):
    """Small coloured lane badge in the bottom-right corner of a tile."""
    h, w = frame.shape[:2]
    color = CLR_GREEN if ls.phase == "GREEN" else CLR_RED
    if ls.emergency:
        color = CLR_EMERGENCY
    label = f"{ls.lane_name} [{ls.phase}]"
    (tw, th), _ = cv2.getTextSize(label, FONT_SMALL, 0.55, 2)
    x, y = w - tw - 12, h - 12
    cv2.rectangle(frame, (x-4, y-th-4), (x+tw+4, y+4), (20,20,20), -1)
    cv2.putText(frame, label, (x, y), FONT_SMALL, 0.55, color, 2)
