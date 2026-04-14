# Smart Traffic Management System — 4-Way Intersection

## Project Structure

```
traffic_system/
│
├── main.py                ← Entry point — runs the 4-lane loop
├── config.py              ← ALL settings in one place (edit this first)
├── detector.py            ← Dual-model detection pipeline (1 frame → detections)
├── signal_controller.py   ← 4-way signal logic + emergency override
├── overlay.py             ← OpenCV drawing — boxes, HUD, 2×2 tile grid
├── logger.py              ← Console print + JSON + CSV session log
│
├── emergency_model.pt     ← Your fine-tuned model (or set path in config.py)
├── data/
│   ├── north.mp4
│   ├── south.mp4
│   ├── east.mp4
│   └── west.mp4
└── logs/                  ← Auto-created; session JSON + CSV saved here
```

---

## Setup (Step by Step)

### 1 — Install dependencies
```bash
pip install ultralytics opencv-python numpy
```

### 2 — Set your venv path in config.py
```python
VENV_PATH = r"C:\Users\abdul\programingcodes\opencv\.venv"
```
The system searches for `emergency_model.pt` inside your venv automatically.
If it can't find it, set the path explicitly:
```python
EMERGENCY_MODEL_PATH = r"C:\Users\abdul\programingcodes\opencv\.venv\emergency_model.pt"
```

### 3 — Set your video sources
```python
VIDEO_SOURCES = [
    r"data\north.mp4",   # Lane 0
    r"data\south.mp4",   # Lane 1
    r"data\east.mp4",    # Lane 2
    r"data\west.mp4",    # Lane 3
]
```
Use integer indices (0, 1, 2, 3) for live webcams.

### 4 — Run
```bash
python main.py
```
Press **Q** to quit. Logs saved to `logs/` automatically.

---

## Signal Logic

### Green time formula
```
total_grade  = Σ vehicle_grade for all detections in that lane
green_time   = clamp(BASE + total_grade × 0.8s,  MIN=5s, MAX=60s)
```

Only **one lane** is GREEN at a time. After it expires, the controller picks the
waiting lane with the **highest total_grade** that has been RED for at least
`MIN_RED_TIME` (default 8s) — this prevents lane starvation.

### Emergency override
1. Fine-tuned model fires above 0.5 confidence → that lane immediately goes **GREEN**
2. Held green for `EMERGENCY_OVERRIDE_TIME` (default 30s)
3. Red stripe appears on that lane's tile in the display
4. Warning printed to console + logged to file

### Vehicle Grades (config.py)
| Vehicle        | Grade |
|----------------|-------|
| Ambulance      | 10    |
| Fire Truck     | 10    |
| Police         | 9     |
| Truck          | 7     |
| Bus            | 6     |
| Car            | 5     |
| Motorcycle     | 4     |
| Bicycle        | 2     |

---

## Log Output

Every 30 frames, a console table is printed:
```
[Frame 00030] Active: North
  Lane     Phase        Veh  Grade  Green Rec  Remaining
  --------------------------------------------------------
  North    GREEN          8     46      46.8s      12.3s
  South    RED            3     17      23.6s       0.0s
  East     RED            0      0      10.0s       0.0s
  West     RED            5     28      32.4s       0.0s
```

At session end, two files are written to `logs/`:
- `session_YYYYMMDD_HHMMSS.json` — full frame-by-frame record
- `session_YYYYMMDD_HHMMSS.csv`  — same data, spreadsheet-friendly

---

## Planned Upgrades
- [ ] Lane-wise grading with ROI polygon zones
- [ ] Jetson Nano / edge export (TensorRT)
- [ ] RL-based green-time optimizer
- [ ] Google Maps API — ambulance path clearing
- [ ] Multi-junction coordination
