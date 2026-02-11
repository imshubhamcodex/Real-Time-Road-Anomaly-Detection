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
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# ---------------- CONFIG ---------------- #
# yolo export model="./runs/detect/yolov11s_trained/weights/best.pt" format=ncnn imgsz=320 half=False
BEST_WEIGHTS_PATH = Path("./runs/detect/yolov11s_trained/weights/best_ncnn_model")
# BEST_WEIGHTS_PATH = Path("./runs/detect/yolov11s_trained/weights/best.pt")

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4

MODE = "live"   # image | video | live

INPUT_IMAGE_PATH_DIR = Path("./RAD_DATASET/images/test")
INPUT_VIDEO_PATH = Path(r"path\to\your\test\video.mp4")

OUTPUT_DIR = Path("test_output")
OUTPUT_DIR.mkdir(exist_ok=True)

CAMERA_INDEX = 0

frame_counter = 0
last_annotated = None


# ---------------- LOAD MODEL ---------------- #
print(f"Loading model from: {BEST_WEIGHTS_PATH}")

try:
    model = YOLO(str(BEST_WEIGHTS_PATH))
    class_names = model.names
    print(f"Model loaded. Classes: {class_names}")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise SystemExit


# ---------------- SUPERVISION ---------------- #
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(
    text_thickness=1,
    text_scale=0.6,
    text_color=sv.Color.BLACK
)


# ---------------- FRAME PROCESSING ---------------- #
def process_frame(frame, frame_index=0):
    # Get original dimensions
    h_orig, w_orig = frame.shape[:2]

    # Run inference on the expected 320x320 size
    results = model(frame, imgsz=320, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]
    
    # Convert to Supervision detections
    detections = sv.Detections.from_ultralytics(results)
    
    # 1. SCALE detections back to original resolution before NMS
    # Ultralytics results often scale automatically, but if they are stuck at 320:
    # detections.xyxy = detections.xyxy * [w_orig/320, h_orig/320, w_orig/320, h_orig/320]

    # 2. Aggressive NMS (Lower threshold = fewer boxes)
    # class_agnostic=True is critical if you see different labels on one object
    detections = detections.with_nms(threshold=0.3, class_agnostic=True)
    
    # 3. Filter by area (Optional: removes tiny 'noise' boxes common in NCNN)
    # detections = detections[detections.area > 500] 

    labels = [
        f"{class_names[cid]} {conf:.2f}"
        for cid, conf in zip(detections.class_id, detections.confidence)
    ]

    # Annotate the ORIGINAL frame
    annotated = box_annotator.annotate(frame.copy(), detections)
    annotated = label_annotator.annotate(annotated, detections, labels)

    return annotated



# ---------------- IMAGE INFERENCE ---------------- #
def infer_on_image(image_path):

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    frame = cv2.imread(str(image_path))
    if frame is None:
        print("Failed to load image")
        return

    annotated = process_frame(frame)

    output_path = OUTPUT_DIR / f"{image_path.stem}_annotated.jpg"
    cv2.imwrite(str(output_path), annotated)

    print(f"Saved: {output_path}")


# ---------------- VIDEO INFERENCE ---------------- #
def infer_on_video(video_path):

    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    output_path = OUTPUT_DIR / f"{video_path.stem}_annotated.mp4"

    print(f"Processing video: {video_path.name}")

    try:
        sv.process_video(
            source_path=str(video_path),
            target_path=str(output_path),
            callback=lambda frame, i: process_frame(frame, i)
        )
        print(f"Video saved: {output_path}")

    except Exception as e:
        print(f"Video processing error: {e}")


# ---------------- LIVE CAMERA ---------------- #
# def infer_on_live_camera(camera_index=0):

#     cap = cv2.VideoCapture(camera_index)

#     if not cap.isOpened():
#         print("Cannot open camera")
#         return

#     prev_time = time.time()

#     print("Press 'q' to quit")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         annotated = process_frame(frame)

#         # ---- FPS ----
#         curr_time = time.time()
#         fps = 1 / max(curr_time - prev_time, 1e-6)
#         prev_time = curr_time

#         cv2.putText(
#             annotated,
#             f"FPS: {fps:.1f}",
#             (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (0, 255, 0),
#             2
#         )

#         cv2.imshow("Live Detection", annotated)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()




def infer_random_images(image_dir, num_samples=10):

    if not image_dir.exists():
        print(f"Folder not found: {image_dir}")
        return

    images = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    if not images:
        print("No images found in folder.")
        return

    sample_images = random.sample(images, min(num_samples, len(images)))

    print(f"Running inference on {len(sample_images)} random images...\n")

    for img_path in sample_images:
        print(f"Processing: {img_path.name}")
        infer_on_image(img_path)
        
        
# ---------------- LIVE CAMERA ---------------- #
def infer_on_live_camera(camera_index=0):

    WIDTH = 320
    HEIGHT = 320
    FRAME_SIZE = int(WIDTH * HEIGHT * 1.5)

    frame_q = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    print("[CAM]: Using Pi Camera (rpicam-vid)")

    cam_proc = subprocess.Popen(
        [
            "rpicam-vid",
            "--width", str(WIDTH),
            "--height", str(HEIGHT),
            "--framerate", "10",
            # "--codec", "yuv420",
            "--nopreview",
            "-t", "0",
            "-o", "-"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    # -------- Camera Reader Thread --------
    def cam_reader():
        while not stop_event.is_set():

            raw = cam_proc.stdout.read(FRAME_SIZE)

            if len(raw) != FRAME_SIZE:
                continue

            try:
                yuv = np.frombuffer(raw, np.uint8).reshape(
                    (int(HEIGHT * 1.5), WIDTH)
                )
                frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

            except:
                continue

            if frame_q.full():
                frame_q.get_nowait()

            frame_q.put(frame)

    threading.Thread(target=cam_reader, daemon=True).start()

    # -------- FPS --------
    prev_time = time.time()
    fps = 0.0

    print("Press 'q' to quit")

    try:
        while True:
            global frame_counter, last_annotated

            try:
                frame = frame_q.get(timeout=0.2)
            except queue.Empty:
                continue

            frame_counter += 1

            # # ---- Frame Skipping ----
            # if frame_counter % 2 == 0:
            #     last_annotated = process_frame(frame)

            # # Use last annotated frame OR raw frame
            # base_frame = last_annotated if last_annotated is not None else frame

            # # VERY IMPORTANT → copy before drawing FPS
            
            
            base_frame = process_frame(frame)
            display_frame = base_frame.copy()

            # ----- FPS smoothing -----
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1 / max(now - prev_time, 1e-6))
            prev_time = now

            cv2.putText(
                display_frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("Live Detection", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop_event.set()
        cam_proc.terminate()
        cv2.destroyAllWindows()




# ---------------- MAIN ---------------- #
if __name__ == "__main__":

    print("\n=== Road Defect Detection ===\n")

    if MODE == "image":
        infer_random_images(INPUT_IMAGE_PATH_DIR)

    elif MODE == "video":
        infer_on_video(INPUT_VIDEO_PATH)

    elif MODE == "live":
        infer_on_live_camera(CAMERA_INDEX)

    else:
        print("MODE must be: image | video | live")

    print("\nDone!")
