from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from speed_estimator import SpeedEstimate, SpeedEstimator


@dataclass
class VehicleTrack:
    track_id: int
    cls_id: int
    conf: float
    xyxy: Tuple[int, int, int, int]


class TrafficAnalyzer:
    def __init__(
        self,
        yolo_det_path: str = "yolo11n.pt",
        yolo_seg_path: str = "yolo11n-seg.pt",
        yolo_pose_path: str = "yolo11n-pose.pt",
        device: str = "cpu",
        line_orientation: str = "horizontal",
        line_y_ratio: float = 0.5,
        line_x_ratio: float = 0.5,
        meters_per_pixel: float = 0.05,
    ) -> None:
        self.device = device
        self._det_path = yolo_det_path
        self._seg_path = yolo_seg_path
        self._pose_path = yolo_pose_path

        self.line_orientation = line_orientation
        self.line_y_ratio = float(line_y_ratio)
        self.line_x_ratio = float(line_x_ratio)
        self._line_color = (0, 255, 255)

        self.frame_idx = 0
        self.video_fps = 30.0
        self.speed_estimator = SpeedEstimator(fps=self.video_fps, meters_per_pixel=meters_per_pixel)

        self.latest_vehicle_tracks: List[VehicleTrack] = []
        self.latest_speeds: Dict[int, SpeedEstimate] = {}
        self.vehicle_count = 0
        self._counted_ids = set()
        self._line_side_prev: Dict[int, float] = {}

        self._detector = None
        self._segmenter = None
        self._pose = None
        self._ocr = None

    def set_frame_index(self, idx: int) -> None:
        self.frame_idx = int(idx)

    def set_video_fps(self, fps: float) -> None:
        if fps and fps > 1e-6:
            self.video_fps = float(fps)
            self.speed_estimator.set_fps(self.video_fps)

    def _get_detector(self):
        if self._detector is None:
            from ultralytics import YOLO

            self._detector = YOLO(self._det_path)
        return self._detector

    def _get_segmenter(self):
        if self._segmenter is None:
            from ultralytics import YOLO

            self._segmenter = YOLO(self._seg_path)
        return self._segmenter

    def _get_pose(self):
        if self._pose is None:
            from ultralytics import YOLO

            self._pose = YOLO(self._pose_path)
        return self._pose

    def _get_ocr(self):
        if self._ocr is None:
            import easyocr

            self._ocr = easyocr.Reader(["en"], gpu=False)
        return self._ocr

    def detect_vehicles(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        if self.line_orientation == "vertical":
            line_pos = int(w * self.line_x_ratio)
            cv2.line(out, (line_pos, 0), (line_pos, h), self._line_color, 2)
        else:
            line_pos = int(h * self.line_y_ratio)
            cv2.line(out, (0, line_pos), (w, line_pos), self._line_color, 2)

        vehicle_cls = {2, 3, 5, 7}
        results = self._get_detector().track(
            out, persist=True, tracker="bytetrack.yaml", conf=0.25, iou=0.7, device=self.device, verbose=False
        )

        self.latest_vehicle_tracks = []
        self.latest_speeds = {}
        for r in results:
            boxes = r.boxes
            if boxes is None or boxes.xyxy is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

            for i in range(len(xyxy)):
                if clss[i] not in vehicle_cls:
                    continue
                x1, y1, x2, y2 = map(int, xyxy[i])
                track_id = int(ids[i]) if ids is not None else -1
                conf = float(confs[i])
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                self.latest_vehicle_tracks.append(VehicleTrack(track_id, int(clss[i]), conf, (x1, y1, x2, y2)))
                speed = self.speed_estimator.update(track_id, self.frame_idx, cx, cy)
                self.latest_speeds[track_id] = speed

                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
                label = f"ID:{track_id}"
                if speed.speed_valid:
                    label += f" {speed.speed_kmh:.1f} km/h"
                cv2.putText(out, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

                if track_id >= 0:
                    coord = cx if self.line_orientation == "vertical" else cy
                    side = coord - line_pos
                    prev_side = self._line_side_prev.get(track_id)
                    self._line_side_prev[track_id] = side
                    if prev_side is not None:
                        crossed = (prev_side < 0 <= side) or (prev_side > 0 >= side)
                        if crossed and track_id not in self._counted_ids:
                            self._counted_ids.add(track_id)
                            self.vehicle_count += 1

        cv2.putText(
            out,
            f"Vehicles crossed: {self.vehicle_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )
        return out

    def segment_road(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        results = self._get_segmenter().predict(out, conf=0.25, device=self.device, verbose=False)
        if not results:
            return out
        r = results[0]
        if r.masks is None or r.masks.data is None:
            return out

        masks = (r.masks.data.detach().cpu().numpy() > 0.5).astype(np.uint8)
        if masks.ndim != 3:
            return out
        if masks.shape[1] != h or masks.shape[2] != w:
            resized = [cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) for m in masks]
            masks = np.stack(resized, axis=0)

        # Simple choice: largest mask
        areas = [int(m.sum()) for m in masks]
        if not areas:
            return out
        road_mask = masks[int(np.argmax(areas))]
        overlay = out.copy()
        overlay[road_mask > 0] = (0, 255, 0)
        return cv2.addWeighted(overlay, 0.5, out, 0.5, 0)

    def estimate_poses(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        results = self._get_pose().predict(out, conf=0.25, device=self.device, verbose=False)
        if not results:
            return out
        r = results[0]
        if r.keypoints is None or r.keypoints.data is None:
            return out

        edges = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        ]
        kpt_data = r.keypoints.data.detach().cpu().numpy()
        for kpts in kpt_data:
            for x, y, c in kpts:
                if c >= 0.3:
                    cv2.circle(out, (int(x), int(y)), 3, (0, 255, 255), -1)
            for a, b in edges:
                if kpts[a, 2] >= 0.3 and kpts[b, 2] >= 0.3:
                    cv2.line(out, (int(kpts[a, 0]), int(kpts[a, 1])), (int(kpts[b, 0]), int(kpts[b, 1])), (255, 255, 0), 2)
        return out

    def recognize_license_plates(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        ocr = self._get_ocr()
        for vt in self.latest_vehicle_tracks:
            x1, y1, x2, y2 = vt.xyxy
            car = out[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if car.size == 0:
                continue
            plate_roi = car[int(0.45 * car.shape[0]):, :]
            if plate_roi.size == 0:
                continue
            gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            texts = ocr.readtext(th, detail=1)
            best = ""
            best_conf = 0.0
            for _, txt, conf in texts:
                clean = "".join([c for c in txt.upper() if c.isalnum()])
                if len(clean) >= 4 and conf > best_conf:
                    best = clean
                    best_conf = float(conf)
            if best:
                cv2.putText(out, f"Plate:{best}", (x1, max(0, y1 - 26)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
        return out

    def draw_runtime_hud(self, frame: np.ndarray, fps: Optional[float]) -> np.ndarray:
        out = frame.copy()
        if fps is not None:
            cv2.putText(out, f"FPS: {fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return out

