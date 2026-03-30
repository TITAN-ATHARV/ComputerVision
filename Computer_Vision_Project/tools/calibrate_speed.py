from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import yaml


clicked: List[Tuple[int, int]] = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked.append((x, y))


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive speed calibration tool")
    parser.add_argument("--video", default="data/traffic.mp4", help="Path to input video")
    parser.add_argument("--config", default="config/config.yaml", help="Config yaml to update")
    parser.add_argument("--distance-m", type=float, default=10.0, help="Real distance between two clicked points")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Unable to read first frame from {args.video}")

    vis = frame.copy()
    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)

    while True:
        draw = vis.copy()
        for i, pt in enumerate(clicked):
            cv2.circle(draw, pt, 6, (0, 255, 255), -1)
            cv2.putText(draw, str(i + 1), (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(
            draw,
            "Click 2 points with known distance. Press s to save, q to quit.",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("calibrate", draw)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s") and len(clicked) >= 2:
            p1, p2 = clicked[0], clicked[1]
            px = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
            mpp = args.distance_m / max(1e-6, px)
            cfg_path = Path(args.config)
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg.setdefault("speed", {})
            cfg["speed"]["meters_per_pixel"] = float(mpp)
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            print(f"Saved speed.meters_per_pixel={mpp:.6f} to {cfg_path}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

