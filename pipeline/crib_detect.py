"""
pipeline/crib_detect.py
每床判讀的一幀：每個床位裡的嬰兒口鼻看不看得到。eval_video.py、live_view.py、demo_render.py、
bench_multistream.py 都用這支，評估和示範跑的是同一段程式。

床位：安裝時框好的範圍（畫面比例座標）和方向 rot（把床位轉正要轉幾度）。頭在框裡的人就算這個床位。
四種判定一起算（狀態：visible 看得到口鼻／hidden 看不到／none 床位裡沒偵測到人）：
  v3        床位裡偵測到人、那個人找不到臉 → 看不到口鼻；床位裡沒偵測到人 → 不警示（v3 本來的行為）
  v3＋床位  床位模式（docs/娃娃測試結果.md §4）：除了透過人找臉，也直接在床位範圍依床位方向轉正後找臉
            ——身體被被子蓋住、姿態模型找不到人時，臉露出來還是找得到。床位裡找不到臉就算看不到口鼻。（基準）
  v4＋床位  再加口鼻檢查（infant_face_v4.py，實驗版）
  v5＋床位  臉分三級 ok／warn／alert（infant_face_v5.py，實驗版）
警示狀態機（連續 3 秒看不到才警示）在 crib_state.py。
"""

import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 要在 import cv2 前設

import cv2
import numpy as np

from pipeline import infant_face_v4, infant_face_v5
from pipeline.face_common import assign_faces
from pipeline.infant_face_v3 import make_persons

MODES = ("v3", "v3_slot", "v4_slot", "v5_slot")
SLOT_INPUT = 640    # 床位範圍縮到長邊這麼多像素再找臉
SLOT_FACE_TH = 0.8  # 床位範圍找到的臉要多高分才算（跟 v3 頭部裁切一樣：裁切後比較容易把抱枕圖案看成臉）
ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def slot_faces(detector, frame, rect, rot, min_score=SLOT_FACE_TH):
    """在床位範圍裡找臉：依床位方向轉正後找，框和特徵點換回原圖座標。min_score 可降低以抓側臉（v5）。"""
    h, w = frame.shape[:2]
    x0, y0 = int(rect[0] * w), int(rect[1] * h)
    crop = frame[y0:int(rect[3] * h), x0:int(rect[2] * w)]
    s = min(1.0, SLOT_INPUT / max(crop.shape[:2]))
    if s < 1:
        crop = cv2.resize(crop, (round(crop.shape[1] * s), round(crop.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ch, cw = crop.shape[:2]
    img = cv2.rotate(crop, ROTATE[rot]) if rot else crop
    detector.setInputSize((img.shape[1], img.shape[0]))
    _, found = detector.detect(img)
    faces = []
    for f in [] if found is None else found:
        if f[-1] < min_score:
            continue
        pts = np.vstack([f[:2], f[:2] + f[2:4], f[4:14].reshape(5, 2)]).astype(np.float64)  # 框的兩角＋5 個特徵點
        u, v = pts[:, 0].copy(), pts[:, 1].copy()
        if rot == 90:        # 轉正後的座標 (u, v) → 轉正前的裁切座標 (x, y)
            pts[:, 0], pts[:, 1] = v, ch - u
        elif rot == 180:
            pts[:, 0], pts[:, 1] = cw - u, ch - v
        elif rot == 270:
            pts[:, 0], pts[:, 1] = cw - v, u
        pts = pts / s + (x0, y0)
        (bx0, by0), (bx1, by1) = pts[:2].min(axis=0), pts[:2].max(axis=0)
        faces.append(infant_face_v4.Face((bx0, by0, bx1, by1, float(f[-1])), pts[2:]))
    return faces


def detect(run_pose, detector, det_v5, frame, slots, rots):
    """一幀：回傳每個床位在四種判定下的狀態，和畫圖用的偵測結果。
    狀態：visible 看得到口鼻／hidden 看不到／none 床位裡沒偵測到人（只有 v3 會出現）。
    v5 額外輸出三級 tier（ok 正臉／warn 側臉／alert）；binary 狀態把 ok、warn 都當 visible（不硬警示）。"""
    h, w = frame.shape[:2]
    persons = make_persons(run_pose(frame))
    faces = infant_face_v4.detect_faces_v4(detector, frame, persons)  # 跟 v3 找到的臉逐一相同，多帶特徵點
    assign_faces(persons, faces)
    v3_faces = [list(p["faces"]) for p in persons]
    infant_face_v4.drop_covered_faces(frame, persons)
    v4_faces = [list(p["faces"]) for p in persons]

    result, extra = {}, []
    for doll, (x0, y0, x1, y1) in slots.items():
        def in_slot(x, y):
            return x0 * w <= x <= x1 * w and y0 * h <= y <= y1 * h

        def face_in_slot(f):
            return in_slot((f[0] + f[2]) / 2, (f[1] + f[3]) / 2)

        inside = [i for i, p in enumerate(persons) if in_slot(*p["head"])]
        found = slot_faces(detector, frame, (x0, y0, x1, y1), rots[doll])
        for f in found:
            f.skin = infant_face_v4.mouth_skin_ratio(frame, f.lm)
        extra += found
        # v3：照原本的做法，床位裡的人有臉就算看得到（臉在哪都算）
        any_v3 = any(v3_faces[i] for i in inside)
        # 床位模式：臉要落在這個床位裡才算（依位置，不透過人框——兩隻被框成一個人時才不會互相拿走對方的臉）
        slot_v3 = any(face_in_slot(f) for fs in v3_faces for f in fs) or bool(found)
        slot_v4 = any(face_in_slot(f) for fs in v4_faces for f in fs) \
            or any(f.skin >= infant_face_v4.MOUTH_SKIN_MIN for f in found)
        # v5：混合三級——backbone 用 v4_slot（保住召回），visible 再分正臉 ok／側臉 warn；
        #     backbone 判 alert 時，只有低門檻找到「乾淨側臉」才把 alert 降成 warn（found5）
        backbone5 = [f for fs in v4_faces for f in fs if face_in_slot(f)] + list(found)
        found5 = slot_faces(det_v5, frame, (x0, y0, x1, y1), rots[doll], min_score=infant_face_v5.FACE_TH_V5)
        v5_tier = infant_face_v5.tier_hybrid(slot_v4, backbone5, found5, frame)
        result[doll] = {
            "persons": len(inside),
            "v3": "visible" if any_v3 else ("hidden" if inside else "none"),
            "v3_slot": "visible" if slot_v3 else "hidden",
            "v4_slot": "visible" if slot_v4 else "hidden",
            "v5_slot": "visible" if v5_tier in ("ok", "warn") else "hidden",
            "v5_tier": v5_tier,
            "face_scores": " ".join(f"{f[4]:.2f}" for i in inside for f in persons[i]["faces"] + persons[i]["covered"]),
            "slot_face_scores": " ".join(f"{f[4]:.2f}" for f in found),
            "mouth_skin": " ".join(f"{f.skin:.2f}" for i in inside for f in persons[i]["faces"] + persons[i]["covered"]
                                   ) + ("｜" + " ".join(f"{f.skin:.2f}" for f in found) if found else ""),
        }
    return result, persons, faces + extra
