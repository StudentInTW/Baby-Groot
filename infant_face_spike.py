"""
infant_face_spike.py
Stage I 風險前置實驗 v2 —— 用「人臉偵測」判斷每個嬰兒的臉看不看得到。

為什麼有 v2：v1（infant_prone_spike.py）用姿態模型的「臉部關鍵點信心度」判斷臉朝下，
在趴睡圖上漏報。YOLOv8-pose 的關鍵點信心度訓練目標是「這個點有沒有被標註」，
不是「看不看得到」，所以被遮住的鼻子、眼睛照樣過門檻（見 docs/context.md §3）。

v2 的分工：
  - 姿態模型（yolov8n-pose）：畫面裡有幾個人、每個人的頭在哪（只用座標，不用信心度）
  - 人臉偵測（OpenCV YuNet）：畫面裡哪些臉「看得到」
  - 配對：每張臉歸給頭部最近的人；配不到臉的人 = 臉看不見 → ALERT

這是單張圖版本。影片版還要加 ByteTrack（每個嬰兒固定 ID）和每個 ID 的狀態機
（連續 N 秒看不到臉才告警；「由看得到變看不到」和「一直看不到」分兩級）。

用法：
    uv run python infant_face_spike.py                        # 跑 dataset/ 裡所有圖片
    uv run python infant_face_spike.py dataset/infant_1.jpg   # 指定圖片
輸出：outputs/<檔名>_face.jpg
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 要在 import cv2 前設；關掉 OpenCV 5 dnn 的 target 警告

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent
POSE_MODEL = ROOT / "model" / "yolov8n-pose.pt"
FACE_MODEL = ROOT / "model" / "face_detection_yunet_2023mar.onnx"
OUT_DIR = ROOT / "outputs"

FACE_KPTS = [0, 1, 2, 3, 4]     # nose, L/R eye, L/R ear —— 只拿座標來定位頭
BODY_KPTS = list(range(5, 17))  # 肩、肘、腕、髖、膝、踝
KPT_TH = 0.3                    # 身體關鍵點可見門檻
MIN_BODY_KPTS = 3               # 可見身體點少於這個數的框當雜訊略過（例如只拍到大人的手）
HEAD_R_RATIO = 0.3              # 臉中心離頭中心不超過「框長邊 × 這個比例」，才算這個人的臉
FACE_SCORE_TH = 0.7             # YuNet 分數門檻；測試圖上的假臉（拳頭、大人的手）最高 0.57。
                                # 還沒用「臉看得到」的圖校正過，之後用仰睡圖調
FACE_SCALES = (640, 1280)       # 長邊縮到這些尺寸各跑一次：YuNet 只擅長 10–300 px 的臉

RED, GREEN, YELLOW, GRAY = (0, 0, 255), (0, 170, 0), (0, 200, 255), (150, 150, 150)


def detect_faces(detector, img):
    """多尺度跑 YuNet，回傳 [(x1, y1, x2, y2, score), ...]（原圖座標）。"""
    h, w = img.shape[:2]
    boxes, scores = [], []
    for side in FACE_SCALES:
        s = min(1.0, side / max(h, w))  # 只縮不放大
        small = cv2.resize(img, (round(w * s), round(h * s))) if s < 1 else img
        detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = detector.detect(small)
        for f in [] if faces is None else faces:
            boxes.append([float(v) / s for v in f[:4]])  # x, y, w, h
            scores.append(float(f[-1]))
    if not boxes:
        return []
    result = []
    for i in np.array(cv2.dnn.NMSBoxes(boxes, scores, FACE_SCORE_TH, 0.3)).flatten():  # 合併不同尺度的同一張臉
        x, y, bw, bh = boxes[i]
        result.append((x, y, x + bw, y + bh, scores[i]))
    return result


def make_person(box, kpts):
    """可見身體關鍵點太少就回傳 None（當雜訊）；否則回傳 person：框、頭中心、配對半徑、配到的臉。"""
    if (kpts[BODY_KPTS, 2] > KPT_TH).sum() < MIN_BODY_KPTS:
        return None
    return {
        "box": box,
        "head": kpts[FACE_KPTS, :2].mean(axis=0),  # 座標可用，只是信心度不能當「看得到」
        "head_r": HEAD_R_RATIO * max(box[2] - box[0], box[3] - box[1]),
        "faces": [],
    }


def find_persons(pose, img):
    """回傳 (persons, ignored)。"""
    result = pose(img, verbose=False)[0]
    persons, ignored = [], []
    if result.keypoints is None:
        return persons, ignored
    for box, kpts in zip(result.boxes.xyxy.cpu().numpy(), result.keypoints.data.cpu().numpy()):
        person = make_person(box, kpts)
        if person is None:
            ignored.append(box)
        else:
            persons.append(person)
    return persons, ignored


def assign_faces(persons, faces):
    """每張臉歸給頭中心最近、且在配對半徑內的人；回傳沒配到人的臉數。"""
    unassigned = 0
    for face in faces:
        cx, cy = (face[0] + face[2]) / 2, (face[1] + face[3]) / 2
        dists = [np.hypot(cx - p["head"][0], cy - p["head"][1]) for p in persons]
        near = [i for i, d in enumerate(dists) if d <= persons[i]["head_r"]]
        if near:
            persons[min(near, key=lambda i: dists[i])]["faces"].append(face)
        else:
            unassigned += 1
    return unassigned


def put_label(img, text, x, y, color, scale, thick):
    """以 (x, y) 為左下角畫白底文字。"""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(x), max(int(y), th + base)
    cv2.rectangle(img, (x, y - th - base), (x + tw, y + base), (255, 255, 255), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def analyze(pose, detector, image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[略過] 讀不到圖片：{image_path}")
        return
    scale = max(0.6, img.shape[1] / 1600)  # 字和線條跟著圖片大小放大
    thick = max(2, round(scale * 2))

    persons, ignored = find_persons(pose, img)
    faces = detect_faces(detector, img)
    unassigned = assign_faces(persons, faces)

    for x1, y1, x2, y2 in ignored:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), GRAY, 1)
        put_label(img, "ignored: few keypoints", x1, y1 + 25 * scale, GRAY, scale * 0.6, 1)
    for x1, y1, x2, y2, score in faces:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), YELLOW, thick)
        put_label(img, f"face {score:.2f}", x1, y2 + 25 * scale, YELLOW, scale * 0.7, max(1, thick - 1))
    for i, p in enumerate(persons, 1):
        ok = bool(p["faces"])
        color = GREEN if ok else RED
        x1, y1, x2, y2 = p["box"]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)
        cv2.circle(img, tuple(int(v) for v in p["head"]), int(p["head_r"]), color, max(1, thick // 2))
        label = f"#{i} face visible - OK" if ok else f"#{i} FACE NOT VISIBLE - ALERT"
        put_label(img, label, x1, y1 - 8, color, scale, thick)

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{Path(image_path).stem}_face.jpg"
    cv2.imwrite(str(out_path), img)

    print(f"[{Path(image_path).name}] {len(persons)} 人、{len(faces)} 張臉（{unassigned} 張沒配到人）、"
          f"略過 {len(ignored)} 個雜訊框 → outputs/{out_path.name}")
    for i, p in enumerate(persons, 1):
        print(f"   #{i} " + ("臉看得到 → OK" if p["faces"] else "臉看不見 → ALERT"))
    if not persons:
        print("   沒偵測到人：通用姿態模型可能抓不到這個嬰兒 → 考慮嬰兒專屬模型")


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(
        str(p) for p in (ROOT / "dataset").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    pose = YOLO(str(POSE_MODEL))
    detector = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), FACE_SCORE_TH)
    for path in paths:
        analyze(pose, detector, path)
