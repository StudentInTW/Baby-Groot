"""
infant_prone_spike.py
Stage I 風險前置實驗 —— 在「嬰兒趴睡/臉朝下」影像上跑姿態偵測，
並以簡單規則標記「臉朝下 = 危險」，產出一張可給隊友看的截圖。

目的：驗證「模型抓得到嬰兒 + 能標出危險姿勢」這個假設。
注意：這是最快的探路版(Tier-1)，用的是成人 COCO 姿態模型，
      對嬰兒準確率有限 —— 目的是先看它 work 到什麼程度。
      正式版應改用嬰兒專屬模型(ostadabbas Infant-Pose-Estimation / FiDIP)。

用法：
    uv sync --group yolo
    uv run python -m experiments.infant_prone_spike 你的圖片.jpg
"""

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# COCO 17 關鍵點：臉部相關索引
FACE_KPTS = [0, 1, 2, 3, 4]      # nose, L/R eye, L/R ear
BODY_KPTS = list(range(5, 17))   # 肩、肘、腕、髖、膝、踝
CONF_TH = 0.3                    # 關鍵點可見度門檻
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "yolov8n-pose.pt"


def analyze(image_path, out_path="spike_out.jpg"):
    model = YOLO(str(MODEL_PATH))
    results = model(image_path)[0]
    img = results.orig_img.copy()

    if results.keypoints is None or len(results.keypoints) == 0:
        cv2.putText(img, "NO PERSON DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imwrite(out_path, img)
        print("[結果] 模型沒偵測到人。")
        print("       這是重要訊號：通用模型可能對嬰兒失效 -> 正式版需嬰兒專屬模型。")
        return

    n_danger = 0
    for kpts, box in zip(results.keypoints.data, results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)

        face_visible = sum(float(kpts[i][2]) > CONF_TH for i in FACE_KPTS)
        body_visible = sum(float(kpts[i][2]) > CONF_TH for i in BODY_KPTS)

        # 規則：臉部關鍵點幾乎看不到、但身體看得到 -> 判定臉朝下（危險）
        danger = (face_visible <= 1) and (body_visible >= 3)
        n_danger += int(danger)

        color = (0, 0, 255) if danger else (0, 180, 0)
        label = "FACE-DOWN / PRONE - ALERT" if danger else "face visible - OK"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, label, (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        for i in range(len(kpts)):
            x, y, c = kpts[i]
            if float(c) > CONF_TH:
                cv2.circle(img, (int(x), int(y)), 4, color, -1)

    cv2.imwrite(out_path, img)
    print(f"[結果] 已輸出 {out_path}；偵測到 {len(results.boxes)} 人，"
          f"其中判定危險(臉朝下) {n_danger} 人。")
    print("       若框/關鍵點明顯不準，這代表通用模型不適用嬰兒 -> 換嬰兒專屬模型。")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "infant.jpg"
    analyze(path)