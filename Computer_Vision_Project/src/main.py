import os
import time

import cv2

from traffic_analyzer import TrafficAnalyzer


def main() -> None:
    video_path = "data/traffic.mp4"
    if not os.path.exists(video_path):
        raise FileNotFoundError("Place input video at data/traffic.mp4")

    analyzer = TrafficAnalyzer(
        yolo_det_path="yolo11n.pt",
        yolo_seg_path="yolo11n-seg.pt",
        yolo_pose_path="yolo11n-pose.pt",
        device="cpu",
        line_orientation=os.environ.get("COUNT_LINE_ORIENTATION", "horizontal"),
        line_y_ratio=float(os.environ.get("COUNT_LINE_Y_RATIO", "0.5")),
        line_x_ratio=float(os.environ.get("COUNT_LINE_X_RATIO", "0.5")),
        meters_per_pixel=float(os.environ.get("SPEED_METERS_PER_PIXEL", "0.05")),
    )

    max_frames_env = os.environ.get("MAX_FRAMES", "").strip()
    max_frames = int(max_frames_env) if max_frames_env else None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    analyzer.set_video_fps(cap.get(cv2.CAP_PROP_FPS))

    frame_idx = 0
    last = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        analyzer.set_frame_index(frame_idx)
        out = analyzer.detect_vehicles(frame)
        out = analyzer.segment_road(out)
        out = analyzer.estimate_poses(out)
        out = analyzer.recognize_license_plates(out)

        now = time.time()
        fps = 1.0 / max(1e-6, now - last)
        last = now
        out = analyzer.draw_runtime_hud(out, fps)

        cv2.imshow("Traffic Perception", out)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

        frame_idx += 1
        if max_frames is not None and frame_idx >= max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

