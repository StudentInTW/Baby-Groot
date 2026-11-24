"""
infant_face_v6.py（2026/11/24 起的主流程；每床判讀在 crib_detect.py 的 v6_slot）
v3 再加「口鼻小塊模型」，取代 v4 的口鼻膚色門檻。結果見 docs/architecture.md「口鼻小塊模型」。

為什麼：v4 的膚色門檻擋得住正臉被衣領、被子蓋到只露眼睛，但在每床鏡頭的側躺寶寶上等於沒做——
雙眼距離 < 22 px 的臉不檢查（側臉兩眼會擠在一起，幾乎都小於 22 px），布從旁邊蓋過來、口鼻旁邊還有皮膚時比例也過得了門檻。
做法：
  1. 跟 v4 一樣用 YuNet 的 5 個特徵點（右眼、左眼、鼻尖、右嘴角、左嘴角）。
  2. 把臉轉正，只切「鼻子到下巴」一小塊：臉的縱軸＝兩眼中點 → 兩嘴角中點，長度 m；
     中心沿縱軸往下 0.8 m、左右對齊鼻尖和嘴巴，邊長 1.3 m（不靠雙眼距離，側臉也切得到）；縮成 96×96。
  3. 小型分類模型判「口鼻露出」的機率：三顆用不同亂數種子訓練的 ResNet18 平均（合判），≥ 0.5 算看得到。
     口鼻看不到的臉（< 0.5）跟 v4 一樣從這個人的臉裡拿掉。
模型：model/mouth_patch_ens3.onnx（三顆合成一個檔，134 MB，不進 repo；環境變數 FG_MOUTH_ONNX 可換路徑）。
  onnxruntime 在主機 CPU 上跑，不用 PyTorch；每張臉約 3–7 ms。
訓練：只用自己拍的娃娃影片和照片（hailo/prep_calib.py SPLITS 的 calib 那一半，2550 張臉），
  綠的幀裡的臉＝露出、黃紅的幀裡 YuNet 找到的（假）臉＝沒露出；另外把露出的小塊 25% 貼一塊布蓋住鼻嘴當成沒露出。
  預訓練權重 timm/resnet18.a1_in1k（Apache-2.0）。真實嬰兒、SyRIP、網路影像都沒進訓練。
  程式在 experiments/mouth_patch_crops.py → mouth_patch_build.py → mouth_patch_train.py → mouth_patch_export.py。
限制：只有 YuNet 找到臉時才判；YuNet 沒找到臉的幀照 v3 算看不到。

用法：
    uv run python -m pipeline.infant_face_v6 圖片1.jpg [圖片2.jpg ...]
輸出：outputs/<檔名>_v6.jpg（橘框＝找到臉但口鼻小塊判成看不到，旁邊是露出的機率）
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 要在 import cv2 前設

import cv2
import numpy as np

from pipeline import face_common as fc
from pipeline.face_common import GREEN, OUT_DIR, RED, ROOT, YELLOW, assign_faces, put_label
from pipeline.infant_face_v3 import load_rtmo, make_persons
from pipeline.infant_face_v4 import ORANGE, detect_faces_v4

MOUTH_ONNX_ENV = "FG_MOUTH_ONNX"
MOUTH_MODEL = ROOT / "model" / "mouth_patch_ens3.onnx"
PATCH = 96            # 口鼻小塊邊長（像素）
SIDE, DOWN = 1.3, 0.8  # 小塊邊長＝m × SIDE；中心在兩眼中點往嘴的方向 m × DOWN
MOUTH_P_MIN = 0.5     # 三顆平均的「口鼻露出」機率低於這個 → 口鼻看不到。沒有調過（訓練時的預設）


def mouth_patch(img, lm, size=PATCH, ctx=1.0):
    """依 5 個特徵點切口鼻小塊（BGR uint8，縱軸朝下）；特徵點疊在一起時回傳 None。
    ctx > 1 時多留範圍（訓練時隨機平移、縮放用，experiments/mouth_patch_build.py）：回傳 round(size×ctx) 見方，
    中間 size 見方跟 ctx=1 切的一樣。"""
    lm = np.asarray(lm, np.float64).reshape(5, 2)
    eye = (lm[0] + lm[1]) / 2
    mouth = (lm[3] + lm[4]) / 2
    v = mouth - eye
    m = float(np.linalg.norm(v))
    if m < 1e-3:
        return None
    u = v / m                      # 朝下
    x = np.array([u[1], -u[0]])    # 臉正放時朝畫面右邊
    lat = float(np.mean([(lm[2] - eye) @ x, (mouth - eye) @ x]))
    center = eye + DOWN * m * u + lat * x
    k = size / (SIDE * m)          # 原圖像素 → 小塊像素
    out = int(round(size * ctx))
    M = np.array([[k * x[0], k * x[1], 0.0], [k * u[0], k * u[1], 0.0]])
    M[:, 2] = out / 2 - M[:, :2] @ center
    return cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR,
                          borderValue=(114, 114, 114))


class MouthModel:
    """口鼻小塊模型（onnxruntime）。probs() 會把結果記在每張臉的 f.p_mouth，同一張臉不重算。"""

    def __init__(self, path):
        import onnxruntime as ort
        self.path = Path(path)
        self.sess = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])

    def probs(self, img, faces):
        todo = [f for f in faces if getattr(f, "p_mouth", None) is None]
        patches = [mouth_patch(img, f.lm) for f in todo]
        ok = [i for i, p in enumerate(patches) if p is not None]
        if ok:
            out = self.sess.run(None, {"bgr": np.stack([patches[i] for i in ok])})[0]
            for i, p in zip(ok, out):
                todo[i].p_mouth = float(p)
        for i, p in enumerate(patches):
            if p is None:
                todo[i].p_mouth = 0.0      # 切不出小塊（特徵點疊在一起）當看不到
        return [f.p_mouth for f in faces]


def load_mouth_model(path=None, required=False):
    """讀口鼻小塊模型；檔案不在時回傳 None（required=True 就直接結束），讓沒有這個檔的人照樣能跑 v3、v4。"""
    path = Path(path or os.environ.get(MOUTH_ONNX_ENV) or MOUTH_MODEL)
    if not path.exists():
        msg = f"找不到口鼻小塊模型 {path}（不在 repo 裡；見 README），v6 不跑"
        if required:
            sys.exit(msg)
        print(msg)
        return None
    return MouthModel(path)


def drop_covered_faces_v6(img, persons, model):
    """跟 v4 的 drop_covered_faces 一樣，但用口鼻小塊模型：p < MOUTH_P_MIN 的臉從 p["faces"] 移到 p["covered"]。"""
    model.probs(img, [f for p in persons for f in p["faces"]])
    for p in persons:
        p["covered"] = [f for f in p["faces"] if f.p_mouth < MOUTH_P_MIN]
        p["faces"] = [f for f in p["faces"] if f.p_mouth >= MOUTH_P_MIN]
    return [f for p in persons for f in p["covered"]]


def analyze(run_pose, detector, model, image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[略過] 讀不到圖片：{image_path}")
        return
    scale = max(0.6, img.shape[1] / 1600)
    thick = max(2, round(scale * 2))
    persons = make_persons(run_pose(img))
    faces = detect_faces_v4(detector, img, persons)
    unassigned = assign_faces(persons, faces)
    model.probs(img, faces)
    covered = drop_covered_faces_v6(img, persons, model)
    for f in faces:
        color = ORANGE if f.p_mouth < MOUTH_P_MIN else YELLOW
        cv2.rectangle(img, (int(f[0]), int(f[1])), (int(f[2]), int(f[3])), color, thick)
        put_label(img, f"face {f[4]:.2f} mouth {f.p_mouth:.2f}", f[0], f[3] + 25 * scale, color, scale * 0.7,
                  max(1, thick - 1))
    for i, p in enumerate(persons, 1):
        ok = bool(p["faces"])
        color = GREEN if ok else RED
        x1, y1, x2, y2 = p["box"]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)
        put_label(img, f"#{i} mouth/nose visible - OK" if ok else f"#{i} MOUTH/NOSE NOT VISIBLE - ALERT",
                  x1, y1 - 8, color, scale, thick)
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{Path(image_path).stem}_v6.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"[{Path(image_path).name}] {len(persons)} 人、{len(faces)} 張臉（{len(covered)} 張口鼻看不到、"
          f"{unassigned} 張沒配到人）→ outputs/{out_path.name}")
    for i, p in enumerate(persons, 1):
        print(f"   #{i} " + ("口鼻看得到 → OK" if p["faces"] else "口鼻看不到 → ALERT"))
    if not persons:
        print("   沒偵測到人")


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        sys.exit("用法：uv run python -m pipeline.infant_face_v6 圖片1.jpg [圖片2.jpg ...]")
    run_pose = load_rtmo()
    detector = fc.create_face_detector(fc.FACE_SCORE_TH)
    model = load_mouth_model(required=True)
    for path in paths:
        analyze(run_pose, detector, model, path)
