import cv2
import yaml
from pathlib import Path

# -------- CONFIG -------- #
FINAL_DATASET = Path("RAD_DATASET")
SPLIT = "train"

IMG_DIR = FINAL_DATASET / "images" / SPLIT
LBL_DIR = FINAL_DATASET / "labels" / SPLIT
DATA_YAML = FINAL_DATASET / "data.yaml"

MAX_W, MAX_H = 1080, 720   # display size


# -------- Load Classes -------- #
with open(DATA_YAML, "r") as f:
    names = yaml.safe_load(f)["names"]

if isinstance(names, dict):
    names = [names[k] for k in sorted(names.keys())]


# -------- Resize Helper -------- #
def resize_keep_aspect(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1)   # never upscale
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h))


# -------- Viewer -------- #
def view_annotations():

    images = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in [".jpg",".png",".jpeg"]])
    idx = 0

    while True:

        img_path = images[idx]
        lbl_path = LBL_DIR / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():

                cls, xc, yc, bw, bh = map(float, line.split())
                cls = int(cls)

                x1 = int((xc - bw/2) * w)
                y1 = int((yc - bh/2) * h)
                x2 = int((xc + bw/2) * w)
                y2 = int((yc + bh/2) * h)

                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(img, names[cls], (x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Resize for display
        display = resize_keep_aspect(img, MAX_W, MAX_H)

        cv2.imshow("YOLO Viewer (n/p/q)", display)

        key = cv2.waitKey(0)
        if key == ord("q"): # Quit
            break
        elif key == ord("n"): # Next
            idx = min(idx+1, len(images)-1)
        elif key == ord("p"): # Previous
            idx = max(idx-1, 0)

    cv2.destroyAllWindows()


# -------- Run -------- #
view_annotations()
