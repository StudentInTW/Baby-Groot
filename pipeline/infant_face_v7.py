"""
infant_face_v7.py（2026/11/27 起的主流程；每床判讀在 crib_detect.py 的 v7_slot）
v6 再加「RetinaFace 補找臉」：v6 判口鼻看不到的床位，才在同一個床位範圍裡用第二個找臉模型 RetinaFace 再找一次，
有一張分數 ≥ RETINA_TH、口鼻小塊模型判露出（≥ 0.5）的臉就改判看得到。結果見 docs/architecture.md「RetinaFace 補找臉」。

為什麼：v6 在沒訓練過的影片上的誤報（口鼻看得到卻警示），多半是 YuNet 在側臉、半張臉時根本沒找到臉，
  口鼻小塊模型沒有機會判。RetinaFace 也是用 WIDER FACE 訓練的找臉模型，但側臉找得到的比較多。
  只在 v6 判看不到時才跑：v6 判看得到的幀不會變慢，也不會被改成看不到——v7 只會讓警示變少，
  所以危險有沒有漏掉要另外看（評估時危險、黃燈都要不比 v6 少抓）。
做法：
  1. 床位範圍跟 YuNet 同一套：縮到長邊 640、依床位方向轉正（crib_detect.slot_crop）。
  2. 放進 640×480（高×寬）輸入的左上角、其餘補 0（比這大的先等比例縮小）。
  3. RetinaFace（MobileNet v1 骨幹、FPN 三層、每格 2 個 anchor）解碼出框、分數和 5 個特徵點（兩眼、鼻尖、兩嘴角），NMS 0.4。
  4. 分數 ≥ RETINA_TH 的臉交給 v6 的口鼻小塊模型（同一個 mouth_patch、同一個 0.5 門檻）。
門檻 RETINA_TH＝0.7：只在 calib 那一半影片上從 0.5／0.7／0.9 選（紅、黃段不比 v6 少抓，綠段誤報最少，一樣多選大的；
  選法跑之前寫好），hold、308 床、另一間月子中心只評估。
模型：model/retinaface_v1_640x480.onnx（14 MB，不進 repo；環境變數 FG_RETINA_ONNX 可換路徑）。
  來源是 Hailo Model Zoo 的 retinaface_mobilenet_v1（MIT；biubug6/Pytorch_Retinaface 的架構，WIDER FACE 訓練），
  Hailo 只給 736×1280 的 TFLite；hailo/retina_port.py 把權重原封搬成 640×480 的 ONNX（核對過跟 TFLite 的輸出一樣）。
  onnxruntime 在主機 CPU 跑、1 條執行緒（RETINA_ORT_THREADS，一次約 73 ms）；
  設 FG_RETINA_HEF＝編好的 HEF（hailo/compile_hef.py --model retina）就改在 UGen300 上跑（hailo/retina_hef.py，晶片上約 16 ms）。
限制：RetinaFace 沒有用嬰兒或娃娃微調；口鼻露不露出仍由 v6 的口鼻小塊模型決定。
照片（eval_doll、eval_syrip 不走床位，2026/11/28 加 --face v7）：rescue_persons_v7——v6 判看不到的人，在這個人的範圍
  （人框放大、依身體方向轉正，對應床位模式的床位裁切）用 RetinaFace 再找一次。11/27 以前簡報裡照片、SyRIP 的 v7 寫「跟 v6 一樣」
  是因為照片模式沒有這一步，不是量出來的。

用法（整張圖當一個床位，畫出 RetinaFace 找到的臉和口鼻機率）：
    uv run python -m pipeline.infant_face_v7 圖片1.jpg [圖片2.jpg ...]
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 要在 import cv2 前設

import cv2
import numpy as np

from pipeline.face_common import OUT_DIR, ROOT, YELLOW, put_label

RETINA_ONNX_ENV = "FG_RETINA_ONNX"
RETINA_HEF_ENV = "FG_RETINA_HEF"   # 設了就改在 UGen300 上跑這顆 HEF（hailo/retina_hef.py）；要裝 pyHailoRT
RETINA_MODEL = ROOT / "model" / "retinaface_v1_640x480.onnx"
RETINA_TH = 0.7         # 臉的分數門檻（calib 上選的，見檔頭）
RETINA_NMS = 0.4        # 重疊框的 IoU 門檻（biubug6 的預設）
RETINA_ORT_THREADS = 1  # 跟口鼻小塊模型一樣只用 1 條執行緒（多床時各床不互搶）
MEAN_BGR = np.array([104, 117, 123], np.float32)   # biubug6 的前處理：BGR 減平均，不除標準差
MIN_SIZES = ((16, 32), (64, 128), (256, 512))       # 每層 2 個 anchor 的邊長（像素）
STEPS = (8, 16, 32)
VARIANCE = (0.1, 0.2)


def letterbox(img, W, H):
    """等比例縮放到放得進 W×H，放左上角、其餘補 0。回傳 (畫布, 縮放倍數)。"""
    s = min(W / img.shape[1], H / img.shape[0])
    r = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)))
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[:r.shape[0], :r.shape[1]] = r
    return canvas, s


def nms(boxes, scores, th=RETINA_NMS):
    idx = cv2.dnn.NMSBoxes([[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes],
                           [float(s) for s in scores], 0.0, th)
    return [int(i) for i in np.array(idx).reshape(-1)]


class RetinaFace:
    """RetinaFace mobilenet v1（onnxruntime）。faces(img, th) → [(框 xyxy, 分數, 5 個特徵點 (5, 2)), ...]（img 的座標）。
    晶片版（hailo/retina_hef.py 的 RetinaHef）繼承這個，只換 raw。"""

    def __init__(self, path):
        import onnxruntime as ort
        self.path = Path(path)
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = RETINA_ORT_THREADS, 1
        self.sess = ort.InferenceSession(str(self.path), so, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.setup(*inp.shape[2:4])

    def setup(self, H, W):
        """輸入大小 H×W 的 anchor（中心、邊長都是 0–1 的比例）和各層的格數。"""
        self.H, self.W = int(H), int(W)
        pri = []
        for mins, st in zip(MIN_SIZES, STEPS):
            for i in range(int(np.ceil(self.H / st))):
                for j in range(int(np.ceil(self.W / st))):
                    for ms in mins:
                        pri.append([(j + 0.5) * st / self.W, (i + 0.5) * st / self.H, ms / self.W, ms / self.H])
        self.priors = np.array(pri, np.float32)
        self.levels = [(int(np.ceil(self.H / st)), int(np.ceil(self.W / st))) for st in STEPS]

    def raw(self, x):
        """x：H×W×3 BGR uint8（letterbox 過）→ {(格高, 格寬, 通道數): (格高, 格寬, 通道數) float32}；
        每層 3 個輸出：框 8（2 個 anchor × 4）、分類 4（2 × 2）、特徵點 20（2 × 10）。"""
        blob = (x.astype(np.float32) - MEAN_BGR).transpose(2, 0, 1)[None]
        return {(o.shape[2], o.shape[3], o.shape[1]): o[0].transpose(1, 2, 0)
                for o in self.sess.run(None, {self.input_name: blob})}

    def faces(self, img, th=RETINA_TH):
        x, s = letterbox(img, self.W, self.H)
        out = self.raw(x)
        loc = np.concatenate([out[(h, w, 8)].reshape(-1, 4) for h, w in self.levels])
        conf = np.concatenate([out[(h, w, 4)].reshape(-1, 2) for h, w in self.levels])
        lan = np.concatenate([out[(h, w, 20)].reshape(-1, 10) for h, w in self.levels])
        e = np.exp(conf - conf.max(1, keepdims=True))
        score = e[:, 1] / e.sum(1)                       # softmax 後「是臉」的機率
        k = score >= th
        p, l, d = self.priors[k], loc[k], lan[k]
        cxcy = p[:, :2] + l[:, :2] * VARIANCE[0] * p[:, 2:]
        wh = p[:, 2:] * np.exp(l[:, 2:] * VARIANCE[1])
        WH = np.array([self.W, self.H], np.float32)
        boxes = list(np.concatenate([cxcy - wh / 2, cxcy + wh / 2], 1) * np.tile(WH, 2))
        lms = list((p[:, None, :2] + d.reshape(-1, 5, 2) * VARIANCE[0] * p[:, None, 2:]) * WH)
        scores = list(score[k])
        keep = nms(boxes, scores) if boxes else []
        return [(boxes[i] / s, float(scores[i]), lms[i] / s) for i in keep]


PHOTO_REGION = 1.25     # 照片模式：人框長邊放大這麼多倍的正方形，當這個人的「床位」（見 rescue_persons_v7）


def person_faces_retina(retina, img, person, persons, min_score=RETINA_TH):
    """照片模式（只給 eval_doll、eval_syrip 用；主流程是 crib_detect 的床位模式）：在這個人的範圍裡用 RetinaFace 找臉。
    範圍＝以人框中心為中心、人框長邊 × PHOTO_REGION 的正方形，依身體方向轉正（跟 v3 頭部轉正同一個角度），
    縮到 crib_detect.SLOT_INPUT——對應床位模式「床位裁切、依床位方向轉正、縮到長邊 640」。
    回傳原圖座標的 Face（有特徵點），只留臉的中心離這個人的頭最近、而且在配對半徑內的（跟 assign_faces 同一個規則）。"""
    from pipeline.crib_detect import SLOT_INPUT, _face
    from pipeline.infant_face_v3 import up_angle
    angle = up_angle(person)
    x0, y0, x1, y1 = (float(v) for v in person["box"][:4])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = SLOT_INPUT / max(max(x1 - x0, y1 - y0) * PHOTO_REGION, 1.0)
    M = cv2.getRotationMatrix2D((cx, cy), 0.0 if angle is None else angle, s)
    M[:, 2] += (SLOT_INPUT / 2 - cx, SLOT_INPUT / 2 - cy)
    crop = cv2.warpAffine(img, M, (SLOT_INPUT, SLOT_INPUT), borderValue=(0, 0, 0))
    inv = cv2.invertAffineTransform(M)
    out = []
    for box, sc, lm in retina.faces(crop, min_score):
        pts = np.vstack([box.reshape(2, 2), lm])
        f = _face(np.hstack([pts, np.ones((len(pts), 1))]) @ inv.T, sc)
        fx, fy = (f[0] + f[2]) / 2, (f[1] + f[3]) / 2
        d = [np.hypot(fx - q["head"][0], fy - q["head"][1]) for q in persons]
        mine = d[next(i for i, q in enumerate(persons) if q is person)]
        if mine <= person["head_r"] and mine <= min(d):
            out.append(f)
    return out


def rescue_persons_v7(img, persons, retina, mouth):
    """照片模式的 v7（eval_doll、eval_syrip 的 --face v7）：先照 v6 跑完（drop_covered_faces_v6 之後），
    v6 判看不到的人（p["faces"] 是空的），才在這個人的範圍裡用 RetinaFace 再找一次臉（person_faces_retina），
    有一張口鼻露出 ≥ 0.5 的就補進 p["faces"]（改判看得到）；v6 判看得到的照舊。回傳補回來的人數。
    床位模式一樣只會把「看不到」改成「看得到」：照片上危險（手擋口鼻）有沒有因此漏掉，要看這個。"""
    from pipeline.infant_face_v6 import MOUTH_P_MIN
    n = 0
    for p in persons:
        if p["faces"]:
            continue
        rf = person_faces_retina(retina, img, p, persons)
        ok = [f for f, q in zip(rf, mouth.probs(img, rf)) if q >= MOUTH_P_MIN]
        if ok:
            p["faces"] = ok
            p["retina"] = True
            n += 1
    return n


def load_retina(path=None, required=False):
    """讀 RetinaFace；檔案不在時回傳 None（required=True 就直接結束），讓沒有這個檔的人照樣能跑 v6。
    有設 FG_RETINA_HEF（而且沒指定 path）就改用晶片版。"""
    hef = os.environ.get(RETINA_HEF_ENV)
    if hef and path is None:
        from hailo.retina_hef import RetinaHef
        return RetinaHef(hef)
    path = Path(path or os.environ.get(RETINA_ONNX_ENV) or RETINA_MODEL)
    if not path.exists():
        msg = f"找不到 RetinaFace 模型 {path}（不在 repo 裡；見 README），v7 不跑"
        if required:
            sys.exit(msg)
        print(msg)
        return None
    return RetinaFace(path)


def main(paths):
    from pipeline import crib_detect, infant_face_v6
    retina = load_retina(required=True)
    mouth = infant_face_v6.load_mouth_model(required=True)
    OUT_DIR.mkdir(exist_ok=True)
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[略過] 讀不到圖片：{path}")
            continue
        faces = crib_detect.slot_faces_retina(retina, img, (0, 0, 1, 1), 0)
        probs = mouth.probs(img, faces)
        scale = max(0.6, img.shape[1] / 1600)
        for f, q in zip(faces, probs):
            cv2.rectangle(img, (int(f[0]), int(f[1])), (int(f[2]), int(f[3])), YELLOW, max(2, round(scale * 2)))
            put_label(img, f"face {f[4]:.2f} mouth {q:.2f}", f[0], f[3] + 25 * scale, YELLOW, scale * 0.7, 2)
        out = OUT_DIR / f"{Path(path).stem}_v7.jpg"
        cv2.imwrite(str(out), img)
        ok = any(q >= infant_face_v6.MOUTH_P_MIN for q in probs)
        print(f"[{Path(path).name}] RetinaFace 找到 {len(faces)} 張臉（分數 ≥ {RETINA_TH}），"
              + ("有一張口鼻露出" if ok else "沒有口鼻露出的臉") + f" → outputs/{out.name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("用法：uv run python -m pipeline.infant_face_v7 圖片1.jpg [圖片2.jpg ...]")
    main(sys.argv[1:])
