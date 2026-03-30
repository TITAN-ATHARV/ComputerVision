# Smart City Traffic & Pedestrian Perception

Simple end-to-end video pipeline using OpenCV + YOLO + EasyOCR.

## Features

1. Vehicle detection + tracking (unique IDs)
2. Vehicle crossing count on a virtual line
3. Vehicle speed estimation (km/h, approximate)
4. Road segmentation overlay
5. Pedestrian pose keypoints/skeleton
6. Basic license plate OCR

## Project Structure

- `src/main.py` - main video loop
- `src/traffic_analyzer.py` - pipeline modules
- `src/speed_estimator.py` - speed logic
- `data/traffic.mp4` - input video

## Setup

```bash
cd /home/anshu/Computer_Vision_Project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Run

```bash
python3 src/main.py
```

Press `q` to quit.

## Useful Runtime Options

- `MAX_FRAMES` - stop after N frames
- `COUNT_LINE_ORIENTATION` - `horizontal` or `vertical`
- `COUNT_LINE_Y_RATIO` - horizontal line position (0.0 to 1.0)
- `COUNT_LINE_X_RATIO` - vertical line position (0.0 to 1.0)
- `SPEED_METERS_PER_PIXEL` - speed scale calibration

Example:

```bash
COUNT_LINE_ORIENTATION=vertical COUNT_LINE_X_RATIO=0.5 SPEED_METERS_PER_PIXEL=0.05 MAX_FRAMES=300 python3 src/main.py
```

## Notes

- Place your input video at `data/traffic.mp4`.
- YOLO model files are auto-downloaded on first run if missing.
- Speed is an estimate and depends on camera angle + `SPEED_METERS_PER_PIXEL`.
- Large files are not committed: `*.pt` weights, `data/*.mp4`, and `outputs/` (see `.gitignore`).

## Push to GitHub

I cannot log in to your GitHub account from here. After you create an empty repository on GitHub (no README/license if you already have a local commit), run:

```bash
cd /home/anshu/Computer_Vision_Project
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

If your default branch is `master` instead of `main`:

```bash
git push -u origin master
```

Use **SSH** if you prefer:

```bash
git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

For HTTPS, GitHub may ask for a **Personal Access Token** instead of your password.

