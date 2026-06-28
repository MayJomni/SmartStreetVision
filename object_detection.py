"""
AI Object Detection - Ultimate Version
Weather, Danger, Counter, Traffic, Time, FPS, CSV, Direction, Sound, Night Vision
"""

import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import csv
import winsound

# ==================== CONFIGURATION ====================
video_file = "15052366_720_1280_30fps.mp4"
save_output = True
csv_file = "detection_stats.csv"

# ==================== LOAD AI MODEL ====================
print("Loading AI Model...")
model = YOLO("yolov8n.pt")
print("Model loaded successfully!")

# ==================== OPEN VIDEO ====================
cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {width}x{height}, {fps}fps, {total_frames} frames")

# ==================== SETUP VIDEO WRITER ====================
output_file = "output_" + video_file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# ==================== CSV SETUP ====================
csv_writer_file = open(csv_file, 'w', newline='')
csv_writer = csv.writer(csv_writer_file)
csv_writer.writerow(['frame', 'time', 'object', 'confidence', 'weather', 'traffic'])

# ==================== COLORS ====================
COLORS = {
    'person': (0, 200, 255),
    'car': (0, 255, 150),
    'truck': (255, 180, 0),
    'bus': (255, 120, 50),
    'bicycle': (200, 100, 255),
    'motorcycle': (200, 100, 255),
    'traffic light': (255, 50, 100),
    'stop sign': (255, 50, 100),
    'default': (180, 180, 200)
}

# ==================== TRACKING ====================
prev_positions = {}
traffic_history = []

# ==================== FUNCTIONS ====================

def enhance_night_vision(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def detect_weather(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = hsv[:,:,2].mean()
    saturation = hsv[:,:,1].mean()

    if saturation < 30 and brightness > 150:
        weather = "FOGGY"
        color = (200, 200, 200)
    elif brightness < 80:
        weather = "RAINY/DARK"
        color = (100, 100, 180)
    elif saturation < 50:
        weather = "CLOUDY"
        color = (150, 150, 150)
    else:
        weather = "SUNNY"
        color = (0, 180, 255)

    cv2.rectangle(frame, (10, 10), (240, 50), color, -1)
    cv2.putText(frame, f"WEATHER: {weather}", (15, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return weather


def detect_danger(boxes_info, frame):
    persons = []
    vehicles = []

    for (x1, y1, x2, y2, class_name, conf) in boxes_info:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        if "person" in class_name.lower():
            persons.append((cx, cy))
        if any(v in class_name.lower() for v in ["car", "truck", "bus", "motorcycle"]):
            vehicles.append((cx, cy))

    dangers = []

    for i in range(len(persons)):
        for j in range(i+1, len(persons)):
            dist = ((persons[i][0]-persons[j][0])**2 +
                   (persons[i][1]-persons[j][1])**2) ** 0.5
            if dist < 80:
                dangers.append("CROWD DETECTED!")
                break

    for px, py in persons:
        for vx, vy in vehicles:
            dist = ((px-vx)**2 + (py-vy)**2) ** 0.5
            if dist < 120:
                dangers.append("PERSON NEAR VEHICLE!")
                break

    if dangers:
        try:
            winsound.Beep(1000, 200)
        except:
            pass

    for i, danger in enumerate(set(dangers)):
        cv2.rectangle(frame, (10, 60 + i*40), (420, 100 + i*40), (0, 0, 200), -1)
        cv2.putText(frame, f"DANGER: {danger}", (15, 88 + i*40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return len(dangers) > 0


def draw_counter(frame, boxes_info):
    counts = {}
    for (x1, y1, x2, y2, class_name, conf) in boxes_info:
        counts[class_name] = counts.get(class_name, 0) + 1

    if not counts:
        return

    total = sum(counts.values())
    y_start = height - 20

    cv2.rectangle(frame, (10, y_start - len(counts)*28 - 35),
                 (220, y_start + 5), (20, 20, 20), -1)

    cv2.putText(frame, "OBJECTS DETECTED:", (15, y_start - len(counts)*28 - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    for i, (name, count) in enumerate(sorted(counts.items())):
        y = y_start - (len(counts) - i - 1) * 28
        cv2.putText(frame, f"  {name}: {count}", (15, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)

    cv2.putText(frame, f"TOTAL: {total}", (15, y_start),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def detect_traffic_density(boxes_info, frame):
    vehicles = [b for b in boxes_info if any(
        v in b[4].lower() for v in ["car", "truck", "bus", "motorcycle"])]
    count = len(vehicles)

    traffic_history.append(count)
    if len(traffic_history) > 30:
        traffic_history.pop(0)

    if count == 0:
        density = "NO TRAFFIC"
        color = (0, 255, 0)
    elif count < 3:
        density = "LIGHT"
        color = (0, 255, 100)
    elif count < 6:
        density = "MODERATE"
        color = (0, 200, 255)
    elif count < 10:
        density = "HEAVY"
        color = (0, 100, 255)
    else:
        density = "TRAFFIC JAM!"
        color = (0, 0, 255)

    cv2.rectangle(frame, (width - 280, 10), (width - 10, 50), color, -1)
    cv2.putText(frame, density, (width - 275, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    graph_x = width - 280
    graph_y = 160
    graph_w = 270
    graph_h = 60
    cv2.rectangle(frame, (graph_x, graph_y),
                 (graph_x + graph_w, graph_y + graph_h), (20, 20, 20), -1)
    cv2.putText(frame, "TRAFFIC HISTORY", (graph_x + 5, graph_y + 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    if len(traffic_history) > 1:
        max_val = max(max(traffic_history), 1)
        for i in range(1, len(traffic_history)):
            x1 = graph_x + int((i-1) * graph_w / 30)
            x2 = graph_x + int(i * graph_w / 30)
            y1 = graph_y + graph_h - int(
                traffic_history[i-1] * (graph_h-20) / max_val)
            y2 = graph_y + graph_h - int(
                traffic_history[i] * (graph_h-20) / max_val)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 200), 2)

    return density


def draw_datetime_fps(frame, fps_real):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    hour = now.hour
    if 6 <= hour < 12:
        time_of_day = "MORNING"
        tod_color = (0, 200, 255)
    elif 12 <= hour < 17:
        time_of_day = "AFTERNOON"
        tod_color = (0, 150, 255)
    elif 17 <= hour < 20:
        time_of_day = "EVENING"
        tod_color = (0, 100, 200)
    else:
        time_of_day = "NIGHT"
        tod_color = (50, 50, 200)

    cv2.rectangle(frame, (width - 280, 55), (width - 10, 155), (20, 20, 20), -1)
    cv2.putText(frame, date_str, (width - 275, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, time_str, (width - 275, 105),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, time_of_day, (width - 275, 130),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, tod_color, 2)
    cv2.putText(frame, f"FPS: {fps_real:.1f}", (width - 275, 150),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)


def detect_direction(boxes_info, frame):
    global prev_positions

    for (x1, y1, x2, y2, class_name, conf) in boxes_info:
        if any(v in class_name.lower() for v in ["car", "truck", "bus"]):
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            obj_id = f"{class_name}_{x1//50}"

            if obj_id in prev_positions:
                prev_cx, prev_cy = prev_positions[obj_id]
                dx = cx - prev_cx
                dy = cy - prev_cy

                if abs(dx) > 3 or abs(dy) > 3:
                    if abs(dx) > abs(dy):
                        direction = ">>>" if dx > 0 else "<<<"
                    else:
                        direction = "vvv" if dy > 0 else "^^^"

                    cv2.putText(frame, direction, (cx - 15, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                               (255, 255, 0), 2)

            prev_positions[obj_id] = (cx, cy)


def get_color(class_name):
    for key, color in COLORS.items():
        if key in class_name.lower():
            return color
    return COLORS['default']


def draw_modern_box(frame, x1, y1, x2, y2, label, confidence):
    color = get_color(label)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    corner_length = min(15, (x2 - x1) // 4)
    cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, 2)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, 2)
    cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, 2)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, 2)
    cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, 2)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, 2)
    cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, 2)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, 2)

    full_text = f"{label} {int(confidence * 100)}%"
    (text_width, text_height), _ = cv2.getTextSize(
        full_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    if y1 - text_height - 10 > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1 - text_height - 10),
                     (x1 + text_width + 12, y1 + 4), (20, 22, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, full_text, (x1 + 6, y1 - 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y2 + 2),
                     (x1 + text_width + 12, y2 + text_height + 12),
                     (20, 22, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, full_text, (x1 + 6, y2 + text_height + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ==================== PROCESS VIDEO ====================
print("\nProcessing... Press 'q' to quit, 's' for screenshot\n")
frame_count = 0
prev_time = datetime.now()

cv2.namedWindow("AI Street Detection", cv2.WINDOW_NORMAL)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1

    curr_time = datetime.now()
    time_diff = (curr_time - prev_time).total_seconds()
    fps_real = 1.0 / time_diff if time_diff > 0 else 0
    prev_time = curr_time

    enhanced_frame = enhance_night_vision(frame)
    results = model(enhanced_frame)
    annotated_frame = frame.copy()

    weather = detect_weather(annotated_frame)
    draw_datetime_fps(annotated_frame, fps_real)

    boxes_info = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            if confidence < 0.45:
                continue
            if class_name not in ["train", "airplane", "boat"]:
                boxes_info.append((x1, y1, x2, y2, class_name, confidence))
                draw_modern_box(annotated_frame, x1, y1, x2, y2,
                              class_name, confidence)

    draw_counter(annotated_frame, boxes_info)
    traffic = detect_traffic_density(boxes_info, annotated_frame)
    detect_danger(boxes_info, annotated_frame)
    detect_direction(boxes_info, annotated_frame)

    now_str = datetime.now().strftime("%H:%M:%S")
    for (x1, y1, x2, y2, class_name, conf) in boxes_info:
        csv_writer.writerow([frame_count, now_str, class_name,
                            round(conf, 2), weather, traffic])

    out.write(annotated_frame)
    cv2.imshow("AI Street Detection", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f"screenshot_frame_{frame_count}.png", annotated_frame)
        print(f"Screenshot saved (frame {frame_count})")

# ==================== CLEANUP ====================
cap.release()
out.release()
cv2.destroyAllWindows()
csv_writer_file.close()
print(f"\nDone! Processed {frame_count} frames")
print(f"Video saved to: {output_file}")
print(f"Stats saved to: {csv_file}")