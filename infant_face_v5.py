"""
infant_face_v5.py
把「臉」這一關從二級（看得到/看不到）改成三級 —— ok / warn / alert。

為什麼（2026/9/22）：6 分鐘正常睡姿影片 IMG_8184 顯示，側睡（側臉、口鼻其實看得到）被 v3/v4 誤判成危險，
因為 YuNet 是正面臉偵測器、側臉分數約 0.7，被床位找臉的 0.8 門檻擋掉。側臉 spike（sideface_spike.py）證明：
把門檻降到 0.3 側臉找回率 99%，且「雙眼到鼻的水平不對稱」eye_asym（正臉≈0.13、側臉≈1.0）能 99% 分正臉/側臉。

**混合架構（v5，2026/9/22 定案）**——因為「降門檻直接當主力」會犧牲真實嬰兒召回
（SyRIP「臉看不到→ALERT」98.6%→46%，見 docs/影片測試結果.md §3d），所以 v5 不動 alert 的召回骨幹：
  1. **backbone**（v4：正常門檻找臉＋口鼻膚色）決定 visible / alert —— 召回維持 v4 水準（SyRIP 98.6%、娃娃 97%）。
  2. backbone 判 **visible** 的臉：用幾何分 **正臉→ok**、**側臉→warn**（只是把可見再分兩級，不改 visible/alert）。
  3. backbone 判 **alert** 時：另外用低門檻找臉，**只有找到「乾淨、有把握的側臉」（clean_profile）才把 alert 降成 warn**，
     否則維持 alert。寧可保守維持 alert，也不為了少一個誤報去冒漏掉真實危險的險。

warn＝軟提醒（「側睡/需關注，看一眼」），不算硬警示、不算硬誤報。
eye_asym 等幾何都相對「雙眼軸」算，跟臉在畫面裡轉幾度無關（躺著的臉是橫的、倒的都成立）。

用法（單張圖，畫三級）：
    uv run python infant_face_v5.py 圖片1.jpg [圖片2.jpg ...]
輸出：outputs/<檔名>_v5.jpg（人框：綠 ok／橘 warn／紅 alert）
床位/影片評估見 eval_video.py（v5_slot 模式）、SyRIP 見 eval_syrip.py（--face v5）。
門檻用「模組.常數」在呼叫時讀取，方便實驗調參。
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np

import face_common as fc
import infant_face_v3 as v3
import infant_face_v4 as v4
from face_common import FACE_MODEL, FACE_SCORE_TH, GREEN, OUT_DIR, RED, assign_faces, put_label
from infant_face_v4 import ORANGE, drop_covered_faces, mouth_skin_ratio
from infant_face_v3 import load_rtmo, make_persons

FACE_TH_V5 = 0.3        # 低門檻找臉（找回側臉；side face spike：側臉中位分數 0.66），只用來把 alert 降成 warn
FRONTAL_MAX = 0.5       # backbone 可見的臉：eye_asym ≤ 此值算正臉→ok，否則側臉→warn（spike：正臉≈0.13、側臉≈1.0）
# 「把 alert 降成 warn」的側臉要夠乾淨、夠有把握（保守，寧可維持 alert）
DOWNGRADE_ASYM = (0.8, 1.3)   # eye_asym 要落在這個窄帶（真側臉≈1.0）
DOWNGRADE_SKIN = 0.20         # 口鼻區膚色比例（同 v4 正臉門檻，擋卡通臉/口鼻被蓋）
DOWNGRADE_SCORE = 0.5         # 臉分數下限（太低分不可靠，維持 alert）
STRUCT_T = (0.0, 1.25)        # 鼻沿「雙眼中點→嘴中點」軸的位置要落在這區間，否則結構說不通＝假臉


def eye_asym(lm):
    """雙眼到鼻的水平距離不對稱（沿雙眼軸）：正臉≈0、側臉≈1。跟臉在畫面裡的旋轉無關。"""
    r_eye, l_eye, nose = lm[0], lm[1], lm[2]
    eye = float(np.linalg.norm(l_eye - r_eye))
    if eye < 1:
        return 9.9
    ax = (l_eye - r_eye) / eye
    d_r = abs(float(np.dot(nose - r_eye, ax))) / eye
    d_l = abs(float(np.dot(nose - l_eye, ax))) / eye
    return abs(d_r - d_l)


def structure_ok(lm):
    """雙眼、鼻、嘴的相對排列合不合理（擋後腦勺、YuNet 亂猜的假臉）。旋轉無關。"""
    r_eye, l_eye, nose, r_mouth, l_mouth = lm
    eye = float(np.linalg.norm(l_eye - r_eye)) + 1e-6
    eye_mid = (r_eye + l_eye) / 2
    mouth_mid = (r_mouth + l_mouth) / 2
    vax = mouth_mid - eye_mid
    vlen = float(np.linalg.norm(vax))
    if vlen < 0.3 * eye:              # 雙眼中點和嘴中點太近＝排列崩掉
        return False
    t = float(np.dot(nose - eye_mid, vax)) / (vlen * vlen)  # 鼻沿臉縱軸的位置（眼=0、嘴=1）
    return STRUCT_T[0] <= t <= STRUCT_T[1]


def is_frontal(face):
    lm = getattr(face, "lm", None)
    return lm is not None and eye_asym(lm) <= FRONTAL_MAX


def visible_tier(faces):
    """backbone 已判 visible 的臉：有正臉→ok，否則（都是側臉）→warn。"""
    return "ok" if (not any(getattr(f, "lm", None) is not None for f in faces)
                    or any(is_frontal(f) for f in faces)) else "warn"


def clean_profile(img, face):
    """夠乾淨、夠有把握的側臉，才拿來把 alert 降成 warn（保守）。"""
    lm = getattr(face, "lm", None)
    if lm is None or face[4] < DOWNGRADE_SCORE or not structure_ok(lm):
        return False
    if not (DOWNGRADE_ASYM[0] <= eye_asym(lm) <= DOWNGRADE_ASYM[1]):
        return False
    return mouth_skin_ratio(img, lm) >= DOWNGRADE_SKIN


def tier_hybrid(backbone_visible, backbone_faces, low_faces, img):
    """混合三級：backbone 決定 visible/alert（保召回）；visible→正臉 ok／側臉 warn；
    alert→只有找到乾淨側臉才降成 warn，否則維持 alert。"""
    if backbone_visible:
        return visible_tier(backbone_faces)
    return "warn" if any(clean_profile(img, f) for f in low_faces) else "alert"


def detect_faces_v5(detector, img, persons):
    """低門檻版的找臉（整張圖＋每個人頭部轉正、帶特徵點），只當「降級用的側臉候選」。
    以「模組.常數」暫時降低 v3/face_common 的門檻，找完還原。detector 的建立門檻要 ≤ FACE_TH_V5。"""
    old_face, old_head = fc.FACE_SCORE_TH, v3.HEAD_FACE_TH
    fc.FACE_SCORE_TH = v3.HEAD_FACE_TH = FACE_TH_V5
    try:
        return v4.detect_faces_v4(detector, img, persons)
    finally:
        fc.FACE_SCORE_TH, v3.HEAD_FACE_TH = old_face, old_head


def assign_low(persons, low_faces):
    """把低門檻候選臉依頭中心最近、在配對半徑內歸給人；回傳 {id(person): [faces]}（不動 person['faces']）。"""
    out = {id(p): [] for p in persons}
    for f in low_faces:
        cx, cy = (f[0] + f[2]) / 2, (f[1] + f[3]) / 2
        near = [(np.hypot(cx - p["head"][0], cy - p["head"][1]), p) for p in persons]
        near = [(d, p) for d, p in near if d <= p["head_r"]]
        if near:
            out[id(min(near, key=lambda dp: dp[0])[1])].append(f)
    return out


TIER_COLOR = {"ok": GREEN, "warn": ORANGE, "alert": RED}
TIER_TEXT = {"ok": "OK (frontal)", "warn": "WARN (side face)", "alert": "ALERT (no clear face)"}


def analyze(run_pose, det_normal, det_low, image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[略過] 讀不到圖片：{image_path}")
        return
    scale = max(0.6, img.shape[1] / 1600)
    thick = max(2, round(scale * 2))
    persons = make_persons(run_pose(img))
    backbone = v4.detect_faces_v4(det_normal, img, persons)   # backbone：正常門檻
    assign_faces(persons, backbone)
    drop_covered_faces(img, persons)                          # v4 口鼻膚色 → person['faces'] 是可見的臉
    low = detect_faces_v5(det_low, img, persons)              # 低門檻候選（降級用）
    low_by = assign_low(persons, low)
    for p in persons:
        p["tier"] = tier_hybrid(bool(p["faces"]), p["faces"], low_by[id(p)], img)
    for f in backbone:
        cv2.rectangle(img, (int(f[0]), int(f[1])), (int(f[2]), int(f[3])),
                      GREEN if is_frontal(f) else ORANGE, thick)
    for i, p in enumerate(persons, 1):
        color = TIER_COLOR[p["tier"]]
        x1, y1, x2, y2 = p["box"]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)
        put_label(img, f"#{i} {TIER_TEXT[p['tier']]}", x1, y1 - 8, color, scale, thick)
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"{Path(image_path).stem}_v5.jpg"
    cv2.imwrite(str(out), img)
    tiers = ", ".join(f"#{i}:{p['tier']}" for i, p in enumerate(persons, 1)) or "沒偵測到人"
    print(f"[{Path(image_path).name}] {len(persons)} 人 → outputs/{out.name}｜{tiers}")


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        sys.exit("用法：uv run python infant_face_v5.py 圖片1.jpg [圖片2.jpg ...]")
    run_pose = load_rtmo()
    det_normal = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), FACE_SCORE_TH)
    det_low = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), FACE_TH_V5)
    for path in paths:
        analyze(run_pose, det_normal, det_low, path)
