from ultralytics import YOLO
import cv2
from pathlib import Path
import supervision as sv
import os

# ---------------- CONFIG ---------------- #
MODEL_PATH = "./runs/detect/yolov11s_trained/weights/best.pt"
TEST_FOLDER = Path("./images/test")
OUTPUT_FOLDER = Path("./test_output")
VIDEO_OUTPUT = "output_video.mp4"

CONFIDENCE = 0.4
IOU = 0.4
IMG_SIZE = 640
VIDEO_FPS = 10

OUTPUT_FOLDER.mkdir(exist_ok=True)

# ---------------- LOAD MODEL ---------------- #
print("Loading model...")
model = YOLO(MODEL_PATH)
print("Model loaded!")

# ---------------- ANNOTATORS ---------------- #
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5)

# ---------------- GET IMAGE LIST ---------------- #
image_paths = sorted([
    p for p in TEST_FOLDER.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
])

if len(image_paths) == 0:
    raise ValueError("No images found in test folder!")

print(f"Found {len(image_paths)} images")

# ---------------- PROCESS IMAGES ---------------- #
output_images = []

for img_path in image_paths:
    print(f"Processing: {img_path.name}")

    frame = cv2.imread(str(img_path))

    results = model.predict(
        frame,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE,
        iou=IOU,
        verbose=False
    )[0]

    detections = sv.Detections.from_ultralytics(results)

    labels = [
        f"{model.names[c]} {conf:.2f}"
        for c, conf in zip(detections.class_id, detections.confidence)
    ]

    annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels
    )

    output_path = OUTPUT_FOLDER / img_path.name
    cv2.imwrite(str(output_path), annotated)

    output_images.append(output_path)

print("All images processed!")

# ---------------- CREATE VIDEO ---------------- #

print("Creating video...")

# Read first frame to get size
first_frame = cv2.imread(str(output_images[0]))
height, width, _ = first_frame.shape

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(
    VIDEO_OUTPUT,
    fourcc,
    VIDEO_FPS,
    (width, height)
)

for img_path in output_images:
    frame = cv2.imread(str(img_path))
    video_writer.write(frame)

video_writer.release()

print(f"Video saved :{VIDEO_OUTPUT}")
