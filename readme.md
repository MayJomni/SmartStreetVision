# 🎯 SmartStreetVision by MAY JOMNI — AI-Powered Urban Scene Analysis

> Real-time street intelligence using YOLOv8 | Built by **May Jomni** — Engineering Student at ENSTAB

---

## 🚀 What This Project Does

SmartStreetVision is an AI computer vision system that analyzes street videos in real-time and detects:

| Feature | Description |
|--------|-------------|
| 🚗 Object Detection | Cars, trucks, buses, motorcycles, persons, bicycles |
| 🌤️ Weather Detection | Sunny, Cloudy, Foggy, Rainy/Dark |
| ⚠️ Danger Alerts | Crowd detected, Person near vehicle |
| 🚦 Traffic Density | No traffic → Light → Moderate → Heavy → Traffic Jam |
| 📊 Traffic History | Real-time graph of vehicle count over time |
| 🕐 Date & Time | Live date, time, and time of day |
| ➡️ Direction Tracking | Vehicle movement direction (>>>, <<<, ^^^, vvv) |
| 📈 FPS Counter | Real-time performance monitoring |
| 🔊 Sound Alert | Beep alarm when danger is detected |
| 💾 CSV Export | All detections saved to `detection_stats.csv` |
| 🌙 Night Vision | CLAHE enhancement for dark/night videos |

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