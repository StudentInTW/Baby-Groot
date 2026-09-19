"""
infant_face_v3.py
Stage I 風險前置實驗 v3 —— 修正 SyRIP 評估找到的兩個問題（見 docs/context.md §7）：

1. 躺著的嬰兒臉常是橫的或倒的，人臉偵測器只認得正的臉
   → 除了 v2 的整張圖偵測，再把每個嬰兒的頭部區域依身體方向（髖部中點 → 肩膀中點）轉正、放大後找一次臉。
     轉正後比較容易把耳朵、臉頰看成臉，所以這裡的門檻比整張圖嚴（HEAD_FACE_TH）。
2. 姿態模型（尤其 RTMO）會對同一個嬰兒輸出重複的框，臉被配給重複的那個
   → 對應關鍵點幾乎重疊的兩個人視為同一人，只留框最貼合關鍵點的那個。

姿態模型改用 RTMO（Apache-2.0，一次處理畫面裡所有嬰兒）；v2 用的 YOLOv8 是 AGPL，提交前要換掉。
v2（infant_face_spike.py）保留作對照。

用法：
    uv run python infant_face_v3.py                        # 跑 dataset/ 裡所有圖片
    uv run python infant_face_v3.py dataset/infant_1.jpg   # 指定圖片
輸出：outputs/<檔名>_v3.jpg
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 要在 import cv2 前設

import cv2
import numpy as np

from face_common import (  # 和 v2 共用的門檻、整張圖人臉偵測、配對與畫圖；不依賴 YOLOv8
    FACE_MODEL, FACE_SCORE_TH, GREEN, KPT_TH, OUT_DIR, RED, ROOT, YELLOW,
    assign_faces, detect_faces, make_person, put_label,
)

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12
DET_TH = 0.3        # RTMO 找人門檻（rtmlib 預設 0.7 太高，躺著的嬰兒容易漏）
HEAD_INPUT = 320    # 頭部區域轉正後縮放成這個大小再找臉（順便把小臉放大）
HEAD_FACE_TH = 0.8  # 頭部裁切找到的臉要更高分才算（整張圖是 0.7）：SyRIP 上臉朝下的圖，
                    # 轉正後出現過 0.72–0.80 的假臉（耳朵、臉頰）；漏判比誤報危險
DUP_DIST = 0.1      # 兩人對應關鍵點的平均距離 < 框對角線 × 這個值，就當成同一人


def load_rtmo():
    """回傳找人函式：輸入 BGR 圖，輸出 [(box_xyxy, kpts(17,3)), ...]。"""
    from rtmlib import Body
    from rtmlib.tools.object_detection.post_processings import multiclass_nms
    rtmo = Body(pose="rtmo", mode="balanced", backend="onnxruntime", device="cpu").pose_model

    def run(img):
        # rtmlib 的 RTMO 只回傳關鍵點；照它的後處理重做一次，順便把人框留下來
        x, ratio = rtmo.preprocess(img)
        det_out, pose_out = rtmo.inference(x)
        boxes = det_out[0, :, :4] / ratio
        kpts = pose_out[0].copy()
        kpts[:, :, :2] /= ratio
        _, keep = multiclass_nms(boxes, det_out[0, :, 4:5], nms_thr=rtmo.nms_thr, score_thr=DET_TH)
        return [] if keep is None else [(boxes[i], kpts[i]) for i in keep]
    return run


def same_person(p, q):
    """p、q 的可見關鍵點幾乎重疊 → 同一人（姿態模型的重複框）。"""
    both = (p["kpts"][:, 2] > KPT_TH) & (q["kpts"][:, 2] > KPT_TH)
    if both.sum() < 5:
        return False
    dist = np.linalg.norm(p["kpts"][both, :2] - q["kpts"][both, :2], axis=1).mean()
    diag = max(np.hypot(b[2] - b[0], b[3] - b[1]) for b in (p["box"], q["box"]))
    return dist < DUP_DIST * diag


def box_fit(p):
    """框和可見關鍵點範圍的 IoU。重複框裡最貼合關鍵點的那個最可能是對的
    （RTMO 常多出一個包進周圍環境的大框）。"""
    vis = p["kpts"][p["kpts"][:, 2] > KPT_TH, :2]
    if len(vis) == 0:
        return 0.0
    (kx1, ky1), (kx2, ky2) = vis.min(axis=0), vis.max(axis=0)
    x1, y1, x2, y2 = p["box"]
    inter = max(0.0, min(x2, kx2) - max(x1, kx1)) * max(0.0, min(y2, ky2) - max(y1, ky1))
    union = (x2 - x1) * (y2 - y1) + (kx2 - kx1) * (ky2 - ky1) - inter
    return inter / union if union > 0 else 0.0


def make_persons(dets):
    """去掉雜訊框、合併重複框；每個 person 比 v2 多存一份 kpts。"""
    kept = []
    for box, kpts in dets:
        p = make_person(box, kpts)
        if p is None:
            continue
        p["kpts"] = kpts
        dup = next((i for i, q in enumerate(kept) if same_person(p, q)), None)
        if dup is None:
            kept.append(p)
        elif box_fit(p) > box_fit(kept[dup]):
            kept[dup] = p
    return kept


def up_angle(person):
    """把這個人的頭轉到正上方要轉幾度（OpenCV 逆時針為正）；肩膀看不到就回傳 None。"""
    k = person["kpts"]
    sho, hip = k[[L_SHOULDER, R_SHOULDER]], k[[L_HIP, R_HIP]]
    sho_ok, hip_ok = sho[:, 2] > KPT_TH, hip[:, 2] > KPT_TH
    if not sho_ok.any():
        return None
    if hip_ok.any():  # 身體「往頭」的方向：髖部中點 → 肩膀中點
        vx, vy = sho[sho_ok, :2].mean(axis=0) - hip[hip_ok, :2].mean(axis=0)
    else:             # 看不到髖部就用肩膀中點 → 頭
        vx, vy = person["head"] - sho[sho_ok, :2].mean(axis=0)
    return float(np.degrees(np.arctan2(vx, -vy)))


def head_faces(detector, img, person):
    """把頭部區域轉正、縮放成 HEAD_INPUT 後找臉；回傳原圖座標的臉 [(x1, y1, x2, y2, score), ...]。"""
    angle = up_angle(person)
    cx, cy = (float(v) for v in person["head"])
    s = HEAD_INPUT / max(2 * person["head_r"], 1.0)  # 裁切邊長＝配對直徑
    M = cv2.getRotationMatrix2D((cx, cy), 0.0 if angle is None else angle, s)
    M[:, 2] += (HEAD_INPUT / 2 - cx, HEAD_INPUT / 2 - cy)  # 頭中心移到裁切圖正中央
    crop = cv2.warpAffine(img, M, (HEAD_INPUT, HEAD_INPUT), borderValue=(114, 114, 114))
    detector.setInputSize((HEAD_INPUT, HEAD_INPUT))
    _, found = detector.detect(crop)
    inv = cv2.invertAffineTransform(M)
    faces = []
    for f in [] if found is None else found:
        if f[-1] < HEAD_FACE_TH:
            continue
        fx, fy = inv @ np.array([f[0] + f[2] / 2, f[1] + f[3] / 2, 1.0])
        half = max(f[2], f[3]) / s / 2
        faces.append((fx - half, fy - half, fx + half, fy + half, float(f[-1])))
    return faces


def detect_faces_v3(detector, img, persons):
    """v2 的整張圖偵測＋每個人頭部轉正後的偵測，合併重複的臉。"""
    faces = detect_faces(detector, img) + [f for p in persons for f in head_faces(detector, img, p)]
    if not faces:
        return []
    boxes = [[f[0], f[1], f[2] - f[0], f[3] - f[1]] for f in faces]
    keep = np.array(cv2.dnn.NMSBoxes(boxes, [f[4] for f in faces], FACE_SCORE_TH, 0.3)).flatten()
    return [faces[i] for i in keep]


def analyze(run_pose, detector, image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[略過] 讀不到圖片：{image_path}")
        return
    scale = max(0.6, img.shape[1] / 1600)  # 字和線條跟著圖片大小放大
    thick = max(2, round(scale * 2))

    persons = make_persons(run_pose(img))
    faces = detect_faces_v3(detector, img, persons)
    unassigned = assign_faces(persons, faces)

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
    out_path = OUT_DIR / f"{Path(image_path).stem}_v3.jpg"
    cv2.imwrite(str(out_path), img)

    print(f"[{Path(image_path).name}] {len(persons)} 人、{len(faces)} 張臉（{unassigned} 張沒配到人）"
          f" → outputs/{out_path.name}")
    for i, p in enumerate(persons, 1):
        print(f"   #{i} " + ("臉看得到 → OK" if p["faces"] else "臉看不見 → ALERT"))
    if not persons:
        print("   沒偵測到人")


if __name__ == "__main__":
    dataset = ROOT / "dataset"
    paths = sys.argv[1:] or (sorted(
        str(p) for p in dataset.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ) if dataset.is_dir() else [])
    if not paths:
        sys.exit("用法：uv run python infant_face_v3.py 圖片1.jpg [圖片2.jpg ...]"
                 "（dataset/ 不在 GitHub 上，請自己指定圖片）")
    run_pose = load_rtmo()
    detector = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), FACE_SCORE_TH)
    for path in paths:
        analyze(run_pose, detector, path)
