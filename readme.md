# 🎯 SmartStreetVision — AI-Powered Urban Scene Analysis

> Real-time street intelligence using YOLOv8 | Built by **May Jomni** — Engineering Student at ENSTAB

---

## 📌 Project Origin & My Contributions

This project started from a basic object detection script cloned from [@ihebaln](https://github.com/ihebaln).

### 🔵 What the original project had:
- Basic YOLOv8 object detection
- Colored bounding boxes around detected objects
- Simple video display

### 🟢 What I added and built:

| Feature | Description |
|---------|-------------|
| 🌤️ Weather Detection | Detects Sunny, Cloudy, Foggy, Rainy/Dark conditions from frame analysis |
| ⚠️ Danger Alerts | Alerts when crowd detected or person is too close to a vehicle |
| 🔊 Sound Alert | Beep alarm triggered automatically when danger is detected |
| 🚦 Traffic Density | Real-time traffic level: No Traffic → Light → Moderate → Heavy → Traffic Jam |
| 📊 Traffic History Graph | Live graph showing vehicle count over the last 30 frames |
| 🕐 Date, Time & FPS | Live date, time, time of day and FPS counter on screen |
| ➡️ Direction Tracking | Detects vehicle movement direction (>>>, <<<, ^^^, vvv) |
| 📈 Object Counter | Counts each detected object category in real-time |
| 💾 CSV Export | Saves all detections frame by frame to `detection_stats.csv` |
| 🌙 Night Vision | CLAHE enhancement to improve detection in dark/night videos |
| 🎯 Confidence Filter | Ignores detections below 45% confidence |
| 🚫 Class Filter | Excludes irrelevant classes (train, airplane, boat) for street scenes |

---

## 🚀 What This Project Does

SmartStreetVision analyzes street videos in real-time and provides:
- Full object detection with confidence scores
- Weather condition analysis from video frames
- Danger detection with sound alerts
- Traffic density monitoring with history graph
- Vehicle direction tracking
- Complete detection logging to CSV

---

## 🛠️ Technologies Used

- **Python 3.12**
- **YOLOv8** (Ultralytics) — State-of-the-art object detection
- **OpenCV** — Video processing and visualization
- **NumPy** — Numerical computation
- **CSV** — Detection data logging

---

## 📦 Installation

```bash
git clone https://github.com/MayJomni/SmartStreetVision.git
cd SmartStreetVision
pip install ultralytics opencv-python numpy
```

---

## ▶️ Usage

1. Place your video file (.mp4) in the project folder
2. Edit line 13 in `object_detection.py`:
```python
video_file = "your_video_name.mp4"
```
3. Run:
```bash
python object_detection.py
```

### Controls
| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot |

---

## 📁 Output Files

- `output_<video_name>.mp4` — Annotated video with all detections
- `detection_stats.csv` — Frame-by-frame detection log
- `screenshot_frame_X.png` — Screenshots captured during playback

---

## 🧠 How It Works