"""
detector.py
===========
Runs the dual-model detection pipeline on a single frame.

  - general_model  : YOLOv8n COCO (vehicles + ByteTrack IDs)
  - emergency_model: Fine-tuned model (ambulance, fire truck, police)

Returns a list of detection dicts ready for the signal controller and overlay.
"""

import numpy as np
from ultralytics import YOLO
from config import Config


def run_detection(
    frame          : np.ndarray,
    general_model  : YOLO,
    emergency_model: YOLO,
    cfg            : Config,
    tracker_state  : dict,     # persistent state dict per lane (pass {} initially)
) -> tuple[list[dict], bool]:
    """
    Returns
    -------
    detections   : list[dict]
        Each dict has keys:
          track_id, vehicle_type, grade, box (x1,y1,x2,y2 int),
          emergency (bool), conf (float)
    emergency_flag : bool
        True if at least one emergency vehicle was detected this frame.
    """
    detections: list[dict] = []
    emergency_flag = False

    # ── 1. General vehicle detection + ByteTrack ─────────────────────────────
    results = general_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=cfg.GENERAL_CONF,
        iou=cfg.GENERAL_IOU,
        verbose=False,
    )

    if results[0].boxes.id is not None:
        boxes     = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        confs     = results[0].boxes.conf.cpu().numpy()

        for box, cls_id, track_id, conf in zip(boxes, class_ids, track_ids, confs):
            if cls_id not in cfg.VEHICLE_CLASSES:
                continue

            v_type = cfg.VEHICLE_CLASSES[cls_id]
            grade  = cfg.VEHICLE_GRADES.get(v_type, 0)

            # Persist type across frames (ByteTrack may lose class briefly)
            if track_id not in tracker_state:
                tracker_state[track_id] = {"type": v_type, "grade": grade}

            detections.append({
                "track_id"    : int(track_id),
                "vehicle_type": tracker_state[track_id]["type"],
                "grade"       : tracker_state[track_id]["grade"],
                "box"         : box.astype(int),
                "emergency"   : False,
                "conf"        : float(conf),
            })

    # ── 2. Emergency vehicle detection ───────────────────────────────────────
    em_results = emergency_model.predict(
        frame,
        conf=cfg.EMERGENCY_CONF,
        verbose=False,
    )

    if em_results[0].boxes is not None and len(em_results[0].boxes):
        em_boxes  = em_results[0].boxes.xyxy.cpu().numpy()
        em_cls_ids= em_results[0].boxes.cls.cpu().numpy().astype(int)
        em_confs  = em_results[0].boxes.conf.cpu().numpy()
        names     = em_results[0].names

        for em_box, em_cls, em_conf in zip(em_boxes, em_cls_ids, em_confs):
            em_label = names.get(int(em_cls), "emergency")
            em_grade = cfg.VEHICLE_GRADES.get(
                em_label.lower().replace(" ", "_"), 10
            )
            emergency_flag = True

            # Try to merge with an existing general detection via IoU
            matched = False
            for det in detections:
                if _iou(det["box"], em_box.astype(int)) > 0.35:
                    det["emergency"]    = True
                    det["vehicle_type"] = em_label
                    det["grade"]        = em_grade
                    matched = True
                    break

            if not matched:
                detections.append({
                    "track_id"    : -1,
                    "vehicle_type": em_label,
                    "grade"       : em_grade,
                    "box"         : em_box.astype(int),
                    "emergency"   : True,
                    "conf"        : float(em_conf),
                })

    return detections, emergency_flag


# ─────────────────────────────────────────────
def _iou(a: np.ndarray, b: np.ndarray) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter  = max(0, xB - xA) * max(0, yB - yA)
    areaA  = (a[2]-a[0]) * (a[3]-a[1])
    areaB  = (b[2]-b[0]) * (b[3]-b[1])
    union  = areaA + areaB - inter
    return inter / union if union > 0 else 0.0
