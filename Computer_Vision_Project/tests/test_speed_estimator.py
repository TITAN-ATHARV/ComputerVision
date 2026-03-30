from speed_estimator import SpeedEstimator


def test_stationary_track_speed_goes_zero():
    est = SpeedEstimator(fps=30.0, meters_per_pixel=0.1, stationary_px=10.0, min_track_age=3)
    for i in range(8):
        out = est.update(track_id=1, frame_idx=i, cx=100.0 + (i % 2), cy=100.0, det_conf=0.9)
    assert out.speed_kmh < 2.0
    assert out.state in {"STATIONARY", "WARMUP"}


def test_moving_track_produces_positive_speed():
    est = SpeedEstimator(fps=30.0, meters_per_pixel=0.05, min_track_age=3, stationary_px=2.0)
    out = None
    for i in range(10):
        out = est.update(track_id=2, frame_idx=i, cx=100.0 + (i * 5), cy=100.0, det_conf=0.9)
    assert out is not None
    assert out.speed_kmh > 2.0
    assert out.speed_valid

