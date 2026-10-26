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

需要 YOLOv8（AGPL，選配）：先執行 `uv sync --group yolo`。v3（infant_face_v3.py）不需要。

用法：
    uv run python -m experiments.infant_face_spike                        # 跑 dataset/ 裡所有圖片
    uv run python -m experiments.infant_face_spike dataset/infant_1.jpg   # 指定圖片
輸出：outputs/<檔名>_face.jpg
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 要在 import cv2 前設；關掉 OpenCV 5 dnn 的 target 警告

import cv2
from ultralytics import YOLO

from pipeline.face_common import (  # 門檻、找臉、配對與畫圖都和 v3 共用
    FACE_MODEL, FACE_SCORE_TH, GRAY, GREEN, OUT_DIR, RED, ROOT, YELLOW,
    assign_faces, detect_faces, make_person, put_label,
)

POSE_MODEL = ROOT / "model" / "yolov8n-pose.pt"


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
