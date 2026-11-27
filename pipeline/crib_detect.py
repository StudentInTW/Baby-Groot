"""
pipeline/crib_detect.py
每床判讀的一幀：每個床位裡的嬰兒口鼻看不看得到。eval_video.py、live_view.py、demo_render.py、
bench_multistream.py、experiments/rtsp_check.py 都用這支，評估、示範和現場判讀跑的是同一段程式。

床位：安裝時框好的範圍（畫面比例座標）和方向 rot（把床位轉正要轉幾度）。頭在框裡的人就算這個床位。
幾種判定一起算（狀態：visible 看得到口鼻／hidden 看不到／none 床位裡沒偵測到人）：
  v3        床位裡偵測到人、那個人找不到臉 → 看不到口鼻；床位裡沒偵測到人 → 不警示（v3 本來的行為）
  v3＋床位  床位模式（docs/娃娃測試結果.md §4）：除了透過人找臉，也直接在床位範圍依床位方向轉正後找臉
            ——身體被被子蓋住、姿態模型找不到人時，臉露出來還是找得到。床位裡找不到臉就算看不到口鼻。
  v4＋床位  再加口鼻膚色檢查（infant_face_v4.py，對照用）
  v5＋床位  臉分三級 ok／warn／alert（infant_face_v5.py，對照用）
  v6＋床位  口鼻小塊模型取代 v4 的膚色門檻（infant_face_v6.py；2026/11/24–11/27 的主流程）——
            床位裡的臉（v3＋床位那些）任一張「口鼻露出」機率 ≥ 0.5 才算看得到。
            要傳 mouth（infant_face_v6.load_mouth_model()）才算，沒傳就不出現這一欄（模型檔不在 repo 裡，見 README）。
  v7＋床位  主流程（2026/11/27 起，MAIN_MODE）：v6 判看不到時，才在床位範圍裡用 RetinaFace 再找一次臉
            （infant_face_v7.py），有一張分數 ≥ 0.7、口鼻露出 ≥ 0.5 的臉就改判看得到。v6 判看得到的照舊。
            要再傳 retina（infant_face_v7.load_retina()）才算；沒有模型檔時主流程退回 v6（main_mode）。
警示狀態機（連續 3 秒看不到才警示）在 crib_state.py。
"""

import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 要在 import cv2 前設

import cv2
import numpy as np

from pipeline import infant_face_v4, infant_face_v5, infant_face_v6, infant_face_v7
from pipeline.face_common import assign_faces
from pipeline.infant_face_v3 import make_persons

MODES = ("v3", "v3_slot", "v4_slot", "v5_slot")
MODES_V6 = MODES + ("v6_slot",)   # 有口鼻小塊模型時（detect 的 mouth）
MODES_V7 = MODES_V6 + ("v7_slot",)   # 再有 RetinaFace 時（detect 的 retina）
MAIN_MODE = "v7_slot"   # 主流程的判定（燈號、警示都看這一欄）；2026/11/24 從 v3_slot 換成 v6、11/27 換成 v7（Harry 定）
FALLBACK_MODES = ("v6_slot", "v3_slot")   # 沒有 RetinaFace（或口鼻小塊模型）、或讀到舊的每幀判定時，依序退回


def main_mode(row):
    """這一列（detect 的結果或 eval_video 存的每幀判定）要看哪一欄：有 v7_slot 就用，沒有就退回 v6_slot，再沒有就 v3_slot。"""
    for m in (MAIN_MODE,) + FALLBACK_MODES[:-1]:
        if row.get(m):
            return m
    return FALLBACK_MODES[-1]
SLOT_INPUT = 640    # 床位範圍縮到長邊這麼多像素再找臉
SLOT_FACE_TH = 0.8  # 床位範圍找到的臉要多高分才算（跟 v3 頭部裁切一樣：裁切後比較容易把抱枕圖案看成臉）
ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def slot_crop(frame, rect, rot):
    """床位範圍裁下來、縮到長邊 SLOT_INPUT、依床位方向轉正。回傳 (轉正後的圖, to_frame)：
    to_frame 把轉正後的圖上的點 (N, 2) 換回原圖座標。YuNet（slot_faces）和 RetinaFace（slot_faces_retina）共用。"""
    h, w = frame.shape[:2]
    x0, y0 = int(rect[0] * w), int(rect[1] * h)
    crop = frame[y0:int(rect[3] * h), x0:int(rect[2] * w)]
    s = min(1.0, SLOT_INPUT / max(crop.shape[:2]))
    if s < 1:
        crop = cv2.resize(crop, (round(crop.shape[1] * s), round(crop.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ch, cw = crop.shape[:2]
    img = cv2.rotate(crop, ROTATE[rot]) if rot else crop

    def to_frame(pts):
        pts = np.array(pts, np.float64)
        u, v = pts[:, 0].copy(), pts[:, 1].copy()
        if rot == 90:        # 轉正後的座標 (u, v) → 轉正前的裁切座標 (x, y)
            pts[:, 0], pts[:, 1] = v, ch - u
        elif rot == 180:
            pts[:, 0], pts[:, 1] = cw - u, ch - v
        elif rot == 270:
            pts[:, 0], pts[:, 1] = cw - v, u
        return pts / s + (x0, y0)
    return img, to_frame


def _face(pts, score):
    """pts：框的兩角＋5 個特徵點（原圖座標）→ Face。"""
    (bx0, by0), (bx1, by1) = pts[:2].min(axis=0), pts[:2].max(axis=0)
    return infant_face_v4.Face((bx0, by0, bx1, by1, float(score)), pts[2:])


def slot_faces(detector, frame, rect, rot, min_score=SLOT_FACE_TH):
    """在床位範圍裡找臉（YuNet）：依床位方向轉正後找，框和特徵點換回原圖座標。min_score 可降低以抓側臉（v5）。"""
    img, to_frame = slot_crop(frame, rect, rot)
    detector.setInputSize((img.shape[1], img.shape[0]))
    _, found = detector.detect(img)
    return [_face(to_frame(np.vstack([f[:2], f[:2] + f[2:4], f[4:14].reshape(5, 2)])), f[-1])  # 框的兩角＋5 個特徵點
            for f in ([] if found is None else found) if f[-1] >= min_score]


def slot_faces_retina(retina, frame, rect, rot, min_score=infant_face_v7.RETINA_TH):
    """同一個床位範圍改用 RetinaFace 找臉（v7）。"""
    img, to_frame = slot_crop(frame, rect, rot)
    return [_face(to_frame(np.vstack([box.reshape(2, 2), lm])), sc) for box, sc, lm in retina.faces(img, min_score)]


def slot_pose(run_pose, slots):
    """實驗（2026/11/13，eval_video --slot-rtmo）：RTMO 不看整張畫面，改成每個床位各裁一次給它，
    框和關鍵點換回整張圖座標再合併（重複的人由 make_persons 合併）。
    廣角鏡頭裡床只佔一角時，娃娃在 RTMO 的輸入裡會大好幾倍：場域測試的托嬰中心鏡頭「被子蓋到胸口」找到人 0% → 94%。
    代價：每個床位多跑一次 RTMO；空床上會把床單當成人（判定仍是看不到，空床另外靠參考畫面判）。"""
    def run(img):
        h, w = img.shape[:2]
        out = []
        for x0, y0, x1, y1 in slots.values():
            X0, Y0 = int(x0 * w), int(y0 * h)
            for box, kpts in run_pose(img[Y0:int(y1 * h), X0:int(x1 * w)]):
                box, kpts = box.copy(), kpts.copy()
                box[[0, 2]] += X0
                box[[1, 3]] += Y0
                kpts[:, 0] += X0
                kpts[:, 1] += Y0
                out.append((box, kpts))
        return out
    return run


def detect(run_pose, detector, det_v5, frame, slots, rots, mouth=None, retina=None):
    """一幀：回傳每個床位在四種判定下的狀態（有 mouth 再加 v6，再有 retina 加 v7），和畫圖用的偵測結果。
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
        if mouth is not None:   # v6：v3＋床位的那些臉（人身上落在床位裡的＋直接在床位找到的），任一張口鼻露出
            cand = [f for fs in v3_faces for f in fs if face_in_slot(f)] + list(found)
            probs = mouth.probs(frame, cand)
            v6 = any(q >= infant_face_v6.MOUTH_P_MIN for q in probs)
            result[doll]["v6_slot"] = "visible" if v6 else "hidden"
            result[doll]["mouth_p"] = " ".join(f"{q:.2f}" for q in probs)
            if retina is not None:   # v7：v6 判看不到時，才在床位裡用 RetinaFace 再找一次臉
                rfaces = [] if v6 else slot_faces_retina(retina, frame, (x0, y0, x1, y1), rots[doll])
                rprobs = mouth.probs(frame, rfaces)
                extra += rfaces
                result[doll]["v7_slot"] = "visible" if v6 or any(q >= infant_face_v6.MOUTH_P_MIN for q in rprobs) else "hidden"
                result[doll]["retina"] = " ".join(f"{f[4]:.2f}:{q:.2f}" for f, q in zip(rfaces, rprobs))   # 分數:口鼻機率
    return result, persons, faces + extra
