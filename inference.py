from ultralytics import YOLO
import cv2
import supervision as sv
import time
from pathlib import Path
import random
import subprocess
import threading
import queue
import numpy as np
import os

# Limit CPU thread explosion
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# ---------------- CONFIG ---------------- #
MODEL_PATH = Path("./runs/detect/yolov11s_trained/weights/best_ncnn_model")
# BEST_WEIGHTS_PATH = Path("./runs/detect/yolov11s_trained/weights/best_ncnn_model")
CONFIDENCE = 0.4
IOU_THRESHOLD = 0.4
IMG_SIZE = 512

FRAME_WIDTH = 128
FRAME_HEIGHT = 128
FRAME_SIZE = int(FRAME_WIDTH * FRAME_HEIGHT * 1.5)

# Queue buffer
frame_q = queue.Queue(maxsize=3)
stop_event = threading.Event()

# ---------------- MODEL LOAD ---------------- #
print("Loading YOLO NCNN model...")
model = YOLO(str(MODEL_PATH), task="detect")
print("Model loaded!")

# ---------------- ANNOTATORS ---------------- #
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5)

class_colors = {}
random.seed(42)

# ---------------- CAMERA PROCESS ---------------- #
print("Starting rpicam stream...")

cam_proc = subprocess.Popen(
    [
        "rpicam-vid",
        "--width", str(FRAME_WIDTH),
        "--height", str(FRAME_HEIGHT),
        "--framerate", "8",            # Slightly safer FPS
        "--codec", "yuv420",
        "--inline",
        "--nopreview",
        "-t", "0",
        "-o", "-"
    ],
    stdout=subprocess.PIPE,
    bufsize=0   # ⭐ CRITICAL FIX
)

# Camera warmup
time.sleep(1)


# ---------------- CAMERA READER THREAD ---------------- #
def cam_reader():
    while not stop_event.is_set():
        try:
            raw = b''

            # ⭐ SAFE PARTIAL READ FIX
            while len(raw) < FRAME_SIZE:
                chunk = cam_proc.stdout.read(FRAME_SIZE - len(raw))
                if not chunk:
                    return
                raw += chunk

            if len(raw) != FRAME_SIZE:
                continue

            yuv = np.frombuffer(raw, dtype=np.uint8).reshape((FRAME_HEIGHT * 3 // 2, FRAME_HEIGHT))
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

            # Drop old frames → real-time behaviour
            if frame_q.full():
                try:
                    frame_q.get_nowait()
                except queue.Empty:
                    pass

            frame_q.put(frame)

        except Exception as e:
            print("Camera read error:", e)
            break


threading.Thread(target=cam_reader, daemon=True).start()


# ---------------- PROCESS FRAME ---------------- #
def process_frame(frame):

    results = model.predict(
        frame,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE,
        iou=IOU_THRESHOLD,
        verbose=False
    )[0]

    detections = sv.Detections.from_ultralytics(results)

    labels = []
    for class_id, conf in zip(detections.class_id, detections.confidence):
        cname = model.names[class_id]

        if cname not in class_colors:
            class_colors[cname] = (
                random.randint(50,255),
                random.randint(50,255),
                random.randint(50,255)
            )

        labels.append(f"{cname} {conf:.2f}")

    annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels
    )

    return annotated


# ---------------- MAIN LOOP ---------------- #
fps = 0
frame_counter = 0
last_annotated = None

print("Starting detection... Press 'q' to quit")

try:
    while True:

        try:
            frame = frame_q.get(timeout=1)
        except queue.Empty:
            continue

        frame_counter += 1

        # ⭐ Run inference every 2nd frame
        if frame_counter % 2 == 0:

            t0 = time.time()
            last_annotated = process_frame(frame)
            infer_time = time.time() - t0

            # Correct FPS = inference FPS
            fps = 0.9 * fps + 0.1 * (1 / max(infer_time, 1e-6))

        if last_annotated is None:
            continue

        display = last_annotated.copy()

        cv2.putText(
            display,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        cv2.imshow("YOLOv11 NCNN Road Detection", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    stop_event.set()
    cam_proc.kill()
    cv2.destroyAllWindows()
    print("Stopped cleanly.")
