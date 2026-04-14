"""
config.py
=========
Single file for ALL tuneable parameters.

BEFORE RUNNING:
  1. Set VENV_PATH to your venv folder
  2. Set VIDEO_SOURCES to your 4 video files or camera indices
  3. Optionally override EMERGENCY_MODEL_PATH if auto-resolve fails
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:

    # ─────────────────────────────────────────────────────────────────────────
    #  VENV — path to your Python virtual environment
    #  The system will search inside it for emergency_model.pt automatically
    # ─────────────────────────────────────────────────────────────────────────
    VENV_PATH: str = r"C:\Users\abdul\programingcodes\Python\ASEPTC\traffic_system\Tvenv"

    # Override this if auto-resolve fails (leave empty for auto)
    EMERGENCY_MODEL_PATH: str = r"C:\Users\abdul\programingcodes\Python\ASEPTC\traffic_system\emergency_model.pt"

    # General COCO detection model (downloaded automatically if missing)
    GENERAL_MODEL_PATH: str = "yolov8n.pt"

    # ─────────────────────────────────────────────────────────────────────────
    #  4-WAY INTERSECTION
    #  One video source per lane. Use int (0,1,2,3) for webcams or file paths.
    # ─────────────────────────────────────────────────────────────────────────
    LANE_NAMES: list = field(default_factory=lambda: [
        "North", "South", "East", "West"
    ])
    VIDEO_SOURCES: list = field(default_factory=lambda: [
        r"C:\Users\abdul\programingcodes\Python\ASEPTC\traffic_system\data\traffic1.mp4",   # Lane 0 — North
        r"C:\Users\abdul\programingcodes\Python\ASEPTC\traffic_system\data\traffic1.mp4",   # Lane 1 — South
        r"C:\Users\abdul\programingcodes\Python\ASEPTC\traffic_system\data\traffic1.mp4",   # Lane 2 — East
        r"C:\Users\abdul\programingcodes\Python\ASEPTC\traffic_system\data\traffic1.mp4",   # Lane 3 — West
#        r"data\south.mp4",   # Lane 1 — South
#        r"data\east.mp4",    # Lane 2 — East
 #       r"data\west.mp4",    # Lane 3 — West
    ])

    # ─────────────────────────────────────────────────────────────────────────
    #  DETECTION THRESHOLDS
    # ─────────────────────────────────────────────────────────────────────────
    GENERAL_CONF  : float = 0.40   # Confidence for general model
    GENERAL_IOU   : float = 0.50   # IOU for ByteTrack
    EMERGENCY_CONF: float = 0.50   # Higher = fewer false emergency alarms

    # ─────────────────────────────────────────────────────────────────────────
    #  COCO CLASS IDs  →  vehicle label (standard YOLOv8 COCO indices)
    # ─────────────────────────────────────────────────────────────────────────
    VEHICLE_CLASSES: dict = field(default_factory=lambda: {
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    })

    # ─────────────────────────────────────────────────────────────────────────
    #  VEHICLE PRIORITY GRADES
    #  Higher grade = more green time allocated to that lane
    # ─────────────────────────────────────────────────────────────────────────
    VEHICLE_GRADES: dict = field(default_factory=lambda: {
        "emergency" : 10,
        "ambulance" : 10,
        "fire_truck": 10,
        "firetruck" : 10,
        "police"    : 9,
        "truck"     : 7,
        "bus"       : 6,
        "car"       : 5,
        "motorcycle": 4,
        "bicycle"   : 2,
    })

    # ─────────────────────────────────────────────────────────────────────────
    #  SIGNAL TIMING  (all values in seconds)
    # ─────────────────────────────────────────────────────────────────────────
    MIN_GREEN_TIME         : float = 5.0    # Floor for any lane's green phase
    MAX_GREEN_TIME         : float = 60.0   # Cap — prevent one lane hogging
    BASE_GREEN_TIME        : float = 10.0   # Green time with zero vehicles
    SECONDS_PER_GRADE_PT   : float = 0.8    # Added seconds per total-grade point
    EMERGENCY_OVERRIDE_TIME: float = 30.0   # Green duration on emergency
    MIN_RED_TIME           : float = 8.0    # A lane must wait this long before
                                            # it can be chosen as green again

    # ─────────────────────────────────────────────────────────────────────────
    #  LOGGING
    # ─────────────────────────────────────────────────────────────────────────
    LOG_INTERVAL_FRAMES: int = 30           # Write a log row every N frames
    LOG_OUTPUT_DIR     : str = "logs"

    # ─────────────────────────────────────────────────────────────────────────
    #  DISPLAY — 2×2 tiled window dimensions per lane cell
    # ─────────────────────────────────────────────────────────────────────────
    TILE_WIDTH : int = 640
    TILE_HEIGHT: int = 360

    # ─────────────────────────────────────────────────────────────────────────
    def __post_init__(self):
        """Auto-resolve emergency model path from venv if not set."""
        if self.EMERGENCY_MODEL_PATH:
            return   # User provided explicit path — use it

        search_dirs = [
            Path(self.VENV_PATH),
            Path(self.VENV_PATH) / "Scripts",
            Path(self.VENV_PATH) / "bin",
            Path(self.VENV_PATH) / "Lib" / "site-packages",
            Path("."),   # current working directory fallback
        ]

        for d in search_dirs:
            candidate = d / "emergency_model.pt"
            if candidate.exists():
                self.EMERGENCY_MODEL_PATH = str(candidate)
                return

        raise FileNotFoundError(
            "\n\n[Config Error] emergency_model.pt not found.\n"
            f"  Searched inside venv : {self.VENV_PATH}\n"
            f"  Also checked cwd     : {Path('.').resolve()}\n"
            "  Fix: set EMERGENCY_MODEL_PATH explicitly in config.py\n"
            "  e.g.  EMERGENCY_MODEL_PATH = r'C:\\path\\to\\emergency_model.pt'\n"
        )
