"""
infant_face_v4.py
v3 再加「口鼻檢查」—— 娃娃測試（docs/娃娃測試結果.md）發現：口鼻被衣領、玩偶、被子蓋住、只露眼睛時，
YuNet 還是會找到臉（0.77–0.87 分），v3 就判成「臉看得到 → OK」。但產品要確認的是「口鼻看得到」。

做法（不用訓練）：YuNet 除了臉框還輸出 5 個特徵點（雙眼、鼻尖、兩個嘴角）。口鼻被蓋住時它仍會猜出位置，
只是那裡其實是衣服或被子。所以把臉依雙眼轉正，拿眉毛上方的皮膚當這張臉的膚色基準，
看「鼻尖到嘴下方」那一塊有多少比例像這個膚色——太少就當成口鼻被蓋住，這張臉不算數。

  - 基準取眉毛上方、不取鼻樑：衣領拉到眼睛下緣時，鼻樑本身就被蓋住了。
  - 色差要算亮度（Lab 的 L）：娃娃的粉色衣領和膚色色相很接近，差在亮度。
  - 擋不住的：後腦勺、枕頭上的卡通臉（整塊顏色一致）、鼻子露出只蓋住嘴。這些要靠影片版的連續幀。

v3（infant_face_v3.py）保留作對照；找人、頭部轉正、合併重複框都沿用 v3。

用法：
    uv run python infant_face_v4.py 圖片1.jpg [圖片2.jpg ...]
輸出：outputs/<檔名>_v4.jpg（橘框＝找到臉但口鼻被蓋住）
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 要在 import cv2 前設

import cv2
import numpy as np

# 門檻用「模組.常數」在呼叫時讀取（不用 from import），eval_doll.py 的 --face-th 等參數才改得到
import face_common as fc
import infant_face_v3 as v3
from face_common import FACE_MODEL, GREEN, OUT_DIR, RED, YELLOW, assign_faces, put_label
from infant_face_v3 import load_rtmo, make_persons, up_angle

ORANGE = (0, 140, 255)
EYE_PX = 64           # 口鼻檢查時把臉轉正、縮放成雙眼距離這麼多像素
CROP_W, CROP_H, EYE_Y = 192, 224, 80  # 轉正後的小圖大小；雙眼中點放在 (CROP_W / 2, EYE_Y)
SKIN_DE = 15          # 和膚色基準的色差（Lab ΔE）小於這個，算像皮膚
MOUTH_SKIN_MIN = 0.2  # 下半臉像皮膚的比例低於這個 → 口鼻被蓋住。用娃娃照第 1、2 批定的：
                      # 被衣領、被子、抱枕蓋住的 ≤ 0.04，看得到的最低 0.30（第 3 批當驗證）


class Face(tuple):
    """(x1, y1, x2, y2, score)，跟 v3 的臉一樣可以直接拆開；另外帶 5 個特徵點 lm 和口鼻檢查的結果 skin。"""

    def __new__(cls, box_score, lm):
        face = super().__new__(cls, box_score)
        face.lm = np.asarray(lm, dtype=np.float64).reshape(5, 2)
        face.skin = None
        return face


def _nms(faces):
    if not faces:
        return []
    boxes = [[f[0], f[1], f[2] - f[0], f[3] - f[1]] for f in faces]
    keep = np.array(cv2.dnn.NMSBoxes(boxes, [f[4] for f in faces], fc.FACE_SCORE_TH, 0.3)).flatten()
    return [faces[i] for i in keep]


def detect_faces_lm(detector, img):
    """face_common.detect_faces 同一套多尺度偵測，但留下特徵點。"""
    h, w = img.shape[:2]
    faces = []
    for side in fc.FACE_SCALES:
        s = min(1.0, side / max(h, w))
        small = cv2.resize(img, (round(w * s), round(h * s))) if s < 1 else img
        detector.setInputSize((small.shape[1], small.shape[0]))
        _, found = detector.detect(small)
        for f in [] if found is None else found:
            x, y, bw, bh = (float(v) / s for v in f[:4])
            faces.append(Face((x, y, x + bw, y + bh, float(f[-1])), f[4:14].reshape(5, 2) / s))
    return _nms(faces)


def head_faces_lm(detector, img, person):
    """infant_face_v3.head_faces 同一套頭部轉正，但留下特徵點（換回原圖座標）。"""
    angle = up_angle(person)
    cx, cy = (float(v) for v in person["head"])
    s = v3.HEAD_INPUT / max(2 * person["head_r"], 1.0)
    M = cv2.getRotationMatrix2D((cx, cy), 0.0 if angle is None else angle, s)
    M[:, 2] += (v3.HEAD_INPUT / 2 - cx, v3.HEAD_INPUT / 2 - cy)
    crop = cv2.warpAffine(img, M, (v3.HEAD_INPUT, v3.HEAD_INPUT), borderValue=(114, 114, 114))
    detector.setInputSize((v3.HEAD_INPUT, v3.HEAD_INPUT))
    _, found = detector.detect(crop)
    inv = cv2.invertAffineTransform(M)
    faces = []
    for f in [] if found is None else found:
        if f[-1] < v3.HEAD_FACE_TH:
            continue
        fx, fy = inv @ np.array([f[0] + f[2] / 2, f[1] + f[3] / 2, 1.0])
        half = max(f[2], f[3]) / s / 2
        lm = np.hstack([f[4:14].reshape(5, 2), np.ones((5, 1))]) @ inv.T
        faces.append(Face((fx - half, fy - half, fx + half, fy + half, float(f[-1])), lm))
    return faces


def detect_faces_v4(detector, img, persons):
    """跟 v3 的 detect_faces_v3 一樣（整張圖＋每個人頭部轉正，合併重複的臉），但每張臉帶特徵點。"""
    return _nms(detect_faces_lm(detector, img) + [f for p in persons for f in head_faces_lm(detector, img, p)])


def _disk(lab, center, r):
    yy, xx = np.ogrid[:lab.shape[0], :lab.shape[1]]
    return lab[(xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= r * r]


def mouth_skin_ratio(img, lm):
    """把臉依雙眼轉正，回傳下半臉（鼻尖上方一點到嘴下方）有多少比例像眉毛上方的膚色。"""
    r_eye, l_eye = lm[0], lm[1]
    d = float(np.linalg.norm(l_eye - r_eye))
    if d < 1:
        return 0.0
    mid = (r_eye + l_eye) / 2
    angle = float(np.degrees(np.arctan2(l_eye[1] - r_eye[1], l_eye[0] - r_eye[0])))
    M = cv2.getRotationMatrix2D((float(mid[0]), float(mid[1])), angle, EYE_PX / d)
    M[:, 2] += (CROP_W / 2 - mid[0], EYE_Y - mid[1])
    crop = cv2.warpAffine(img, M, (CROP_W, CROP_H), flags=cv2.INTER_AREA, borderValue=(114, 114, 114))
    nose, r_mouth, l_mouth = np.hstack([lm[2:], np.ones((3, 1))]) @ M.T
    mouth = (r_mouth + l_mouth) / 2

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255  # OpenCV 的 L 是 0–255，換成標準的 0–100
    lab[..., 1:] -= 128
    cx, e = CROP_W / 2, EYE_PX
    ref = np.median(np.vstack([  # 眉心上方＋兩邊眉毛上方，取中位數當膚色基準
        _disk(lab, (cx, EYE_Y - 0.3 * e), 0.15 * e),
        _disk(lab, (cx - 0.45 * e, EYE_Y - 0.35 * e), 0.12 * e),
        _disk(lab, (cx + 0.45 * e, EYE_Y - 0.35 * e), 0.12 * e),
    ]), axis=0)
    y0, y1 = int(max(0, nose[1] - 0.15 * e)), int(min(CROP_H, mouth[1] + 0.25 * e))
    x0, x1 = int(cx - 0.55 * e), int(cx + 0.55 * e)
    lower = lab[y0:y1, x0:x1].reshape(-1, 3)
    if len(lower) == 0:
        return 0.0
    return float((np.linalg.norm(lower - ref, axis=1) < SKIN_DE).mean())


def drop_covered_faces(img, persons):
    """每個人配到的臉都做口鼻檢查，口鼻被蓋住的臉從 p["faces"] 移到 p["covered"]
    （這個人若因此沒有臉 → ALERT）。回傳所有被拿掉的臉，畫圖用。"""
    for p in persons:
        for f in p["faces"]:
            f.skin = mouth_skin_ratio(img, f.lm)
        p["covered"] = [f for f in p["faces"] if f.skin < MOUTH_SKIN_MIN]
        p["faces"] = [f for f in p["faces"] if f.skin >= MOUTH_SKIN_MIN]
    return [f for p in persons for f in p["covered"]]


def analyze(run_pose, detector, image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[略過] 讀不到圖片：{image_path}")
        return
    scale = max(0.6, img.shape[1] / 1600)
    thick = max(2, round(scale * 2))

    persons = make_persons(run_pose(img))
    faces = detect_faces_v4(detector, img, persons)
    unassigned = assign_faces(persons, faces)
    covered = drop_covered_faces(img, persons)

    for f in faces:
        is_covered = any(f is c for c in covered)
        color = ORANGE if is_covered else YELLOW
        text = f"mouth covered {f[4]:.2f}" if is_covered else f"face {f[4]:.2f}"
        cv2.rectangle(img, (int(f[0]), int(f[1])), (int(f[2]), int(f[3])), color, thick)
        put_label(img, text, f[0], f[3] + 25 * scale, color, scale * 0.7, max(1, thick - 1))
    for i, p in enumerate(persons, 1):
        ok = bool(p["faces"])
        color = GREEN if ok else RED
        x1, y1, x2, y2 = p["box"]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)
        cv2.circle(img, tuple(int(v) for v in p["head"]), int(p["head_r"]), color, max(1, thick // 2))
        label = f"#{i} mouth/nose visible - OK" if ok else f"#{i} MOUTH/NOSE NOT VISIBLE - ALERT"
        put_label(img, label, x1, y1 - 8, color, scale, thick)

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{Path(image_path).stem}_v4.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"[{Path(image_path).name}] {len(persons)} 人、{len(faces)} 張臉（{len(covered)} 張口鼻被蓋住、"
          f"{unassigned} 張沒配到人）→ outputs/{out_path.name}")
    for i, p in enumerate(persons, 1):
        print(f"   #{i} " + ("口鼻看得到 → OK" if p["faces"] else "口鼻看不到 → ALERT"))
    if not persons:
        print("   沒偵測到人")


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        sys.exit("用法：uv run python infant_face_v4.py 圖片1.jpg [圖片2.jpg ...]")
    run_pose = load_rtmo()
    detector = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), fc.FACE_SCORE_TH)
    for path in paths:
        analyze(run_pose, detector, path)
