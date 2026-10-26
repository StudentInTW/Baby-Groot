"""
face_common.py
v2、v3 共用的部分：門檻、YuNet 找臉、把臉配給頭部最近的人、畫標籤。

這個檔案不依賴 ultralytics（YOLOv8，AGPL），所以只跑 v3 的話不用裝 PyTorch。
"""

import os
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 要在 import cv2 前設；關掉 OpenCV 5 dnn 的 target 警告

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FACE_MODEL = ROOT / "model" / "face_detection_yunet_2023mar.onnx"
OUT_DIR = ROOT / "outputs"

FACE_KPTS = [0, 1, 2, 3, 4]     # nose, L/R eye, L/R ear —— 只拿座標來定位頭
BODY_KPTS = list(range(5, 17))  # 肩、肘、腕、髖、膝、踝
KPT_TH = 0.3                    # 身體關鍵點可見門檻
MIN_BODY_KPTS = 3               # 可見身體點少於這個數的框當雜訊略過（例如只拍到大人的手）
HEAD_R_RATIO = 0.3              # 臉中心離頭中心不超過「框長邊 × 這個比例」，才算這個人的臉
FACE_SCORE_TH = 0.7             # YuNet 分數門檻；測試圖上的假臉（拳頭、大人的手）最高 0.57
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
