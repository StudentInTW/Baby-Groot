"""
cctv_sim.py
把我們自己拍的娃娃影片做成「像托嬰中心監視器拍出來的」影片，給 eval_video.py（--cctv）評估，也可以當訓練資料。

為什麼要做：只把畫面縮小太樂觀。2026/11/13 的縮小實驗（experiments/scale_sweep.py：縮小＋JPEG）裡，
每床鏡頭的錄影縮到臉只剩 22–38 px 還找得到臉（0.91 分）；場域測試的托嬰中心廣角鏡頭，臉 26–40 px，
「被子蓋到胸口、臉露出」卻一幀都判不到。

真實監視器跟「手機拍再縮小」差在哪，先量過（measure）：
  - 強邊緣不比較糊（2026/11/14，直邊量 10–90% 上升距離）：托嬰那支左右 σ≈0.5、上下 σ≈1.3 像素，
    台南那批手機影片縮到同樣臉寬後約 0.9。監視器多半有銳化，邊緣反而利。所以這裡不加模糊（blur 留著備用）。
  - 色調差很多：畫面灰白、飽和度低，同一隻娃娃的臉亮度、對比、飽和度都跟手機拍的不一樣。
  - 細節被抹平：降噪＋壓縮把低對比的細節（臉上的五官）抹掉，靜止的地方連雜訊都沒有。
calibrate 用「同一隻娃娃仰躺正臉」的臉部範圍對齊這兩項：色調（對比、亮度、飽和度）和壓縮強度（qp）。
只用仰躺正臉調，其他擺法（蓋到身體、側臉、蓋住）留著驗證像不像（docs/影片測試結果.md §3j）。

每一幀的處理（參數在 PRESETS）：
  1. 縮小：整幀縮到「口鼻看得到的片段裡，床位裁切中 YuNet 臉框寬的中位數」等於 face_px（真實鏡頭仰躺時的臉寬）。不放大。
  2. 擺位：貼到 1920×1080 的灰底上（anchor：br 右下角、c 中間；放不下就裁掉床位以外的部分）。
     畫面大小跟真實鏡頭一樣，路徑 A（整張圖找臉）和 RTMO 看整張圖時的縮放才會一樣。
  3. 色調：contrast（以 128 為中心縮放）、brightness（位移）、saturation（顏色往灰階拉）。
  4. 模糊、雜訊：blur=(σx, σy)、noise（目前都是 0，見上）。
  5. 壓縮：ffmpeg（libx264）編成 15 fps、固定量化參數 qp 的 H.264（沒有 B 幀、每 2 秒一個 I 幀、4:2:0）。
     用固定 QP 不用位元率：畫面大半是灰底，給位元率的話位元都花在娃娃身上，畫質反而比真實鏡頭好。
做不到的：斜拍的角度（前縮）、娃娃的長相、包巾樣式、床欄、紅外線夜視（eval_video 的 --gray 只是近似）。

資料規定：場域錄影只拿來量 PRESETS 裡那幾個數字，影像本身不進訓練；錄影的檔名用參數給，不寫進程式。
做出來的影片照 hailo/prep_calib.py 的 SPLITS 分：calib 那一半可以訓練，hold 那一半只能評估。

用法（要 ffmpeg：用 uv run --with imageio-ffmpeg 臨時掛上，不寫進 pyproject，免得 uv sync 動到 pyHailoRT）：
  F="uv run --with imageio-ffmpeg python -m evaluation.cctv_sim"
  $F measure --video <錄影.mp4> --slot x0,y0,x1,y1                  # 量真實鏡頭（錄影旁邊要有 rtsp_check 的 .csv）
  $F calibrate --video <錄影.mp4> --slot x0,y0,x1,y1 --preset daycare IMG_8735 ...   # 用同一隻娃娃的影片對齊
  $F render --preset daycare [影片 ...]
  uv run python -m evaluation.eval_video --cctv daycare
輸出：outputs/cctv_sim/<preset>/<影片>.mp4、<影片>.json（縮放倍率、新的床位座標）；臉寬快取 outputs/cctv_sim/face_px.json
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 要在 import cv2 前設

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from evaluation import video_labels  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "cctv_sim"
CANVAS_W, CANVAS_H = 1920, 1080
GRAY = 128
OUT_FPS = 15          # 監視器常見的幀率；eval_video 再從裡面每秒取 --fps 幀
FACE_TH = 0.6         # 量臉寬用的門檻（只量大小，不是判定；0.8 的話有一支一張臉都量不到）
PATCH_TH = 0.7        # 量臉部色調用的門檻
REPEAT = 5            # calibrate 時每張取樣重複編幾幀：靜止畫面在 P 幀穩定後的畫質，才像監視器拍靜止的寶寶
SHARPEN_SIGMA = 1.0   # 銳化用的模糊半徑（像素）
COVERED_BODY = ("蓋到身體", "蓋到脖子", "蓋住身體", "蓋棉被", "邊緣在嘴唇下", "壓在身上", "放身上")

IDENTITY_TONE = {"contrast": 1.0, "brightness": 0.0, "saturation": 1.0}
PRESETS = {
    # face_px：真實鏡頭仰躺、不蓋東西時，床位裁切裡 YuNet 臉框寬的中位數（跟 face_px() 量原影片的方法一樣）。
    #   現場 rtsp_check 記的 face_px 是各路徑臉框取最大（40 px），比較大，不能直接用。
    # 色調、sharpen：calibrate 用台南 11/08 那批（照護娃娃）仰躺正臉對齊場域錄影的結果（2026/11/14）：
    #   真實 亮度 165、對比（亮度標準差）27、飽和度 32、細節 3.26；對齊後 167、32、32、3.18，臉分數 0.80（真實 0.85）。
    #   sharpen 調完對比變大，contrast 再從 0.96 按比例降到 0.82。qp 30 是假設（DVR 的位元率讀不到），qp 24–40 細節差不多。
    "daycare": {"face_px": 32, "anchor": "br", "blur": (0.0, 0.0), "noise": 0.0, "qp": 30, "sharpen": 1.0,
                "contrast": 0.82, "brightness": 21.8, "saturation": 0.65},
    # 對照組：只縮小、擺位，不調色調、幾乎不壓縮（等於 11/13 縮小實驗的做法，拿來看色調和壓縮的影響）
    "daycare_shrink": {"face_px": 32, "anchor": "br", "blur": (0.0, 0.0), "noise": 0.0, "qp": 12, "sharpen": 0.0,
                       **IDENTITY_TONE},
}


# ---------- 量畫質 ----------

def edge_sigma(gray, min_len=20, min_contrast=25, half=7.0, step=0.25, max_angle=20, plateau=0.12):
    """直邊的模糊寬度（像素，相當於高斯 σ）：找長直邊（Canny＋HoughLinesP），沿邊的垂直方向取亮度剖面，
    量 10%→90% 的上升距離（高斯模糊的階梯＝2.563σ）。只用兩側平坦、單調上升的剖面；每條邊取中位數，
    再依邊長加權取中位數。x＝直的邊（左右方向的模糊）、y＝橫的邊。完全銳利的邊約 0.3（取樣本身）。
    回傳 {"x": (σ, 邊數), "y": (σ, 邊數)}。"""
    g = gray.astype(np.float32)
    lines = cv2.HoughLinesP(cv2.Canny(gray, 30, 90), 1, np.pi / 180, threshold=15, minLineLength=min_len, maxLineGap=2)
    res = {"x": [], "y": []}
    u = np.arange(-half, half + 1e-6, step)
    k = max(3, int(round(1.5 / step)))   # 兩端各 1.5 px 當平台
    H, W = g.shape
    for x1, y1, x2, y2 in ([] if lines is None else lines.reshape(-1, 4).astype(float)):
        L = math.hypot(x2 - x1, y2 - y1)
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        axis = "x" if abs(ang - 90) < max_angle else ("y" if min(ang, 180 - ang) < max_angle else None)
        s = np.arange(2, L - 2, 1.0)
        if axis is None or s.size < 5:
            continue
        tx, ty = (x2 - x1) / L, (y2 - y1) / L
        X = (x1 + tx * s[:, None] - ty * u[None, :]).astype(np.float32)
        Y = (y1 + ty * s[:, None] + tx * u[None, :]).astype(np.float32)
        if X.min() < 0 or Y.min() < 0 or X.max() > W - 1 or Y.max() > H - 1:
            continue
        prof = cv2.remap(g, X, Y, cv2.INTER_LINEAR)
        lo, hi = prof[:, :k].mean(1), prof[:, -k:].mean(1)
        C = hi - lo
        ok = (np.abs(C) >= min_contrast) & (prof[:, :k].std(1) < plateau * np.abs(C)) \
            & (prof[:, -k:].std(1) < plateau * np.abs(C))
        sig = []
        for row in (prof[ok] - lo[ok, None]) / C[ok, None]:     # 0 → 1
            i10, i90 = int(np.argmax(row >= 0.1)), int(np.argmax(row >= 0.9))
            if i10 == 0 or i90 <= i10 or row[:i10].max() > 0.2 or row[i90:].min() < 0.8 \
                    or row[i10:i90 + 1].max() > 1.1 or row[i10:i90 + 1].min() < -0.1:
                continue
            cross = [u[i - 1] + (lvl - row[i - 1]) / (row[i] - row[i - 1]) * step for i, lvl in ((i10, 0.1), (i90, 0.9))]
            sig.append((cross[1] - cross[0]) / 2.5631)
        if len(sig) >= 3:
            res[axis].append((float(np.median(sig)), L))
    out = {}
    for a, v in res.items():
        if not v:
            out[a] = (float("nan"), 0)
            continue
        s, w = np.array(v).T
        order = np.argsort(s)
        cw = np.cumsum(w[order])
        out[a] = (float(s[order][np.searchsorted(cw, cw[-1] / 2)]), len(v))
    return out


def face_patch(frame, rect, rot, det):
    """床位裡最高分的臉（路徑 C：床位裁切、≥PATCH_TH），框往外擴 1.3 倍的範圍：
    (亮度平均, 亮度標準差, 飽和度平均, 細節＝拉普拉斯標準差／亮度標準差, 臉分數)；找不到臉回傳 None。"""
    from pipeline.crib_detect import slot_faces
    fs = slot_faces(det, frame, rect, rot, min_score=PATCH_TH)
    if not fs:
        return None
    f = max(fs, key=lambda f: f[4])
    cx, cy, w, h = (f[0] + f[2]) / 2, (f[1] + f[3]) / 2, (f[2] - f[0]) * 1.3, (f[3] - f[1]) * 1.3
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2)), int(min(W, cx + w / 2)), int(min(H, cy + h / 2))
    patch = frame[y0:y1, x0:x1]
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sat = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    detail = cv2.Laplacian(g, cv2.CV_32F, ksize=3).std() / max(float(g.std()), 1e-3)
    return float(g.mean()), float(g.std()), float(sat.mean()), float(detail), float(f[4])


def patch_summary(rows):
    a = np.array(rows)
    return dict(zip(("mean", "std", "sat", "detail", "score"), np.median(a, axis=0).round(3).tolist()), n=len(rows))


def real_supine(video, slot, key="2", every=5):
    """真實錄影（rtsp_check --record 存的 .mp4，旁邊要有同名 .csv 的按鍵標註）裡，仰躺正臉（按鍵 key）的臉部範圍。"""
    from pipeline.face_common import create_face_detector
    det = create_face_detector(PATCH_TH)
    keys = [r["key"] for r in csv.DictReader(open(Path(video).with_suffix(".csv"), encoding="utf-8-sig"))]
    cap = cv2.VideoCapture(str(video))
    rows, i = [], 0
    while i < len(keys):
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0 and keys[i] == key:
            r = face_patch(frame, slot, 0, det)
            if r:
                rows.append(r)
        i += 1
    cap.release()
    return patch_summary(rows)


def is_supine(seg):
    """仰躺正臉、口鼻看得到、身上沒蓋東西的片段（calibrate 只用這種）。"""
    return seg["label"] == "ok" and seg.get("staged") != "prone" and "側" not in seg["note"] \
        and not any(k in seg["note"] for k in COVERED_BODY)


# ---------- 做影片 ----------

def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
    raise SystemExit("需要 ffmpeg：指令前面加 uv run --with imageio-ffmpeg（見檔頭用法）")


class H264Writer:
    """把 BGR 幀用管線送給 ffmpeg，編成固定 QP 的 H.264（監視器常見的設定：沒有 B 幀、每 2 秒一個 I 幀、4:2:0）。"""

    def __init__(self, path, size, qp, fps=OUT_FPS):
        w, h = size
        self.proc = subprocess.Popen(
            [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-qp", str(qp),
             "-g", str(2 * fps), "-bf", "0", "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait():
            raise RuntimeError(f"ffmpeg 失敗（{self.proc.returncode}）")


def layout(w, h, scale, slots, anchor):
    """縮小後的畫面放在畫布哪裡：回傳 (縮小後寬高, 裁切左上角, 貼上的左上角, 貼上的寬高, 新的床位座標)。
    縮小後比畫布大時，裁成畫布大小、盡量把所有床位留在裡面。"""
    sw, sh = round(w * scale), round(h * scale)
    pw, ph = min(sw, CANVAS_W), min(sh, CANVAS_H)
    ux = (min(r[0] for r in slots.values()) + max(r[2] for r in slots.values())) / 2 * sw
    uy = (min(r[1] for r in slots.values()) + max(r[3] for r in slots.values())) / 2 * sh
    cx = int(np.clip(round(ux - pw / 2), 0, sw - pw))
    cy = int(np.clip(round(uy - ph / 2), 0, sh - ph))
    ox, oy = (CANVAS_W - pw, CANVAS_H - ph) if anchor == "br" else ((CANVAS_W - pw) // 2, (CANVAS_H - ph) // 2)
    new = {}
    for doll, (x0, y0, x1, y1) in slots.items():
        X0, X1 = (np.clip([x0 * sw - cx, x1 * sw - cx], 0, pw) + ox) / CANVAS_W
        Y0, Y1 = (np.clip([y0 * sh - cy, y1 * sh - cy], 0, ph) + oy) / CANVAS_H
        new[doll] = tuple(round(float(v), 4) for v in (X0, Y0, X1, Y1))
    return (sw, sh), (cx, cy), (ox, oy), (pw, ph), new


def tone(img, preset):
    """監視器的色調：對比（以 128 為中心縮放）、亮度位移、飽和度（顏色往灰階拉）。預設值不處理。"""
    c, b, s = preset.get("contrast", 1.0), preset.get("brightness", 0.0), preset.get("saturation", 1.0)
    if (c, b, s) == (1.0, 0.0, 1.0):
        return img
    f = img.astype(np.float32)
    gray = f @ np.array([0.114, 0.587, 0.299], np.float32)  # BGR 的亮度
    f = gray[..., None] + s * (f - gray[..., None])
    f = (f - 128) * c + 128 + b
    return np.clip(f, 0, 255).astype(np.uint8)


def make_frame(frame, spec, preset, rng):
    """一幀原始畫面（或已經縮好的）→ 縮小、擺位、色調、模糊、雜訊之後的畫布（還沒壓縮）。"""
    (sw, sh), (cx, cy), (ox, oy), (pw, ph) = spec["size"], spec["crop"], spec["offset"], spec["paste"]
    small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA) if (sw, sh) != frame.shape[1::-1] else frame
    canvas = np.full((CANVAS_H, CANVAS_W, 3), GRAY, np.uint8)
    region = tone(small[cy:cy + ph, cx:cx + pw], preset)
    if preset.get("sharpen"):  # 監視器的銳化（unsharp mask）：原圖＋a×(原圖−模糊)
        f = region.astype(np.float32)
        region = np.clip(f + preset["sharpen"] * (f - cv2.GaussianBlur(f, (0, 0), SHARPEN_SIGMA)), 0, 255).astype(np.uint8)
    canvas[oy:oy + ph, ox:ox + pw] = region
    bx, by = preset["blur"]
    if bx or by:
        k = tuple(2 * math.ceil(3 * s) + 1 if s else 1 for s in (bx, by))
        canvas = cv2.GaussianBlur(canvas, k, sigmaX=bx or 0.1, sigmaY=by or 0.1)
    if preset["noise"]:
        region = canvas[oy:oy + ph, ox:ox + pw].astype(np.float32)
        region += rng.normal(0, preset["noise"], region.shape[:2]).astype(np.float32)[..., None]  # 亮度雜訊
        canvas[oy:oy + ph, ox:ox + pw] = np.clip(region, 0, 255).astype(np.uint8)
    return canvas


def face_px(name, ann, cache_path=OUT_DIR / "face_px.json"):
    """這支影片「口鼻看得到」的片段裡，床位中 YuNet 找到的臉寬中位數（原解析度像素；每秒一幀）。結果存快取。"""
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    if name in cache:
        return cache[name]
    from pipeline.crib_detect import slot_faces
    from pipeline.face_common import create_face_detector
    det = create_face_detector(FACE_TH)
    cap = cv2.VideoCapture(str(video_labels.find_video(name)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    oks = [s for s in ann["segments"] if s["label"] == "ok"]
    widths, i = [], 0
    while cap.grab():
        t = i / fps
        if i % round(fps) == 0 and any(s["start"] <= t <= s["end"] for s in oks):
            _, frame = cap.retrieve()
            for s in oks:
                if s["start"] <= t <= s["end"]:
                    slot = ann["slots"][s["doll"]]
                    widths += [f[2] - f[0] for f in slot_faces(det, frame, slot["rect"], slot["rot"], min_score=FACE_TH)]
        i += 1
    cap.release()
    if not widths:
        raise RuntimeError(f"{name}：口鼻看得到的片段裡量不到臉寬")
    value = round(float(np.median(widths)), 1)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}  # 重讀：可能有別的程序剛寫過
    cache[name] = value
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, cache_path)
    return value


def video_spec(name, ann, preset):
    """這支影片在這個條件下的縮放和擺位。"""
    cap = cv2.VideoCapture(str(video_labels.find_video(name)))
    ok, frame = cap.read()
    cap.release()
    h, w = frame.shape[:2]
    src_face = face_px(name, ann)
    scale = min(1.0, preset["face_px"] / src_face)
    slots = {d: s["rect"] for d, s in ann["slots"].items()}
    size, crop, offset, paste, new_slots = layout(w, h, scale, slots, preset["anchor"])
    return {"scale": round(scale, 4), "src_face_px": src_face, "size": size, "crop": crop, "offset": offset,
            "paste": paste, "slots": new_slots}


def out_paths(preset_name, name):
    d = OUT_DIR / preset_name
    return d / f"{name}.mp4", d / f"{name}.json"


def load_rendered(preset_name, name):
    """eval_video 用：做好的影片路徑和新的床位座標。還沒做就提示指令。"""
    video, meta = out_paths(preset_name, name)
    if not (video.exists() and meta.exists()):
        raise SystemExit(f"還沒做 {preset_name} 的 {name}：先跑 uv run --with imageio-ffmpeg python -m evaluation.cctv_sim "
                         f"render --preset {preset_name} {name}")
    spec = json.loads(meta.read_text(encoding="utf-8"))
    return video, {d: tuple(r) for d, r in spec["slots"].items()}


def source_frames(name, clips=None, fps=OUT_FPS):
    """每 1/fps 秒回傳一幀 (秒, 幀)；clips＝[(開始秒, 結束秒), ...] 時只回傳這幾段。
    時間跟 eval_video 一樣用「幀號／影片標的 fps」算。手機在暗處會降幀率（台南那批有 25、29 fps 的），
    所以照時間挑幀、不照固定間隔，做出來的影片片長和時間軸才跟標註對得上。"""
    cap = cv2.VideoCapture(str(video_labels.find_video(name)))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    end = max(b for _, b in clips) if clips else None
    i = k = 0
    while cap.grab():
        t = i / src_fps
        if end is not None and t > end:
            break
        frame = None
        while k / fps <= t:              # 到這一幀為止該輸出的時間點，都用這一幀
            T = k / fps
            if clips is None or any(a <= T <= b for a, b in clips):
                frame = cap.retrieve()[1] if frame is None else frame
                yield T, frame
            k += 1
        i += 1
    cap.release()


def render(name, ann, preset_name):
    """做一支影片，存 .mp4 和 .json。回傳 spec。"""
    preset = PRESETS[preset_name]
    spec = video_spec(name, ann, preset)
    video, meta = out_paths(preset_name, name)
    video.parent.mkdir(parents=True, exist_ok=True)
    writer, rng = H264Writer(video, (CANVAS_W, CANVAS_H), preset["qp"]), np.random.default_rng(0)
    for _, frame in source_frames(name):
        writer.write(make_frame(frame, spec, preset, rng))
    writer.close()
    meta.write_text(json.dumps({"preset": preset_name, **preset, **spec}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return spec


# ---------- 對齊 ----------

def supine_samples(names, anns, preset, fps=1.0):
    """這些影片裡仰躺正臉片段的取樣（每秒 fps 張）：[(spec, 已縮小的幀, 床位, rot), ...]。"""
    out = []
    for name in names:
        ann = anns[name]
        spec = video_spec(name, ann, preset)
        segs = [s for s in ann["segments"] if is_supine(s)]
        for t, frame in source_frames(name, [(s["start"], s["end"]) for s in segs], fps):
            small = cv2.resize(frame, tuple(spec["size"]), interpolation=cv2.INTER_AREA)
            for s in segs:
                if s["start"] <= t <= s["end"]:
                    out.append((spec, small, spec["slots"][s["doll"]], ann["slots"][s["doll"]]["rot"]))
    return out


def sim_supine(samples, preset):
    """用這組參數做出取樣（每張重複 REPEAT 幀、編碼、解碼，取每組最後一幀），量臉部範圍。"""
    from pipeline.face_common import create_face_detector
    det = create_face_detector(PATCH_TH)
    tmp = Path(tempfile.mkdtemp()) / "calib.mp4"
    writer, rng = H264Writer(tmp, (CANVAS_W, CANVAS_H), preset["qp"]), np.random.default_rng(0)
    for spec, small, _, _ in samples:
        canvas = make_frame(small, spec, preset, rng)
        for _ in range(REPEAT):
            writer.write(canvas)
    writer.close()
    cap = cv2.VideoCapture(str(tmp))
    rows = []
    for spec, _, rect, rot in samples:
        frame = None
        for _ in range(REPEAT):
            frame = cap.read()[1]
        r = face_patch(frame, rect, rot, det)
        if r:
            rows.append(r)
    cap.release()
    shutil.rmtree(tmp.parent, ignore_errors=True)
    return patch_summary(rows) if rows else None


def cmd_measure(args):
    """量真實錄影：整張和床位範圍的直邊模糊寬度、仰躺正臉的臉部範圍。"""
    slot = tuple(float(v) for v in args.slot.split(","))
    cap = cv2.VideoCapture(args.video)
    vals, i = {"整張": [], "床位": []}, 0
    while cap.grab():
        if i % args.every == 0:
            frame = cv2.cvtColor(cap.retrieve()[1], cv2.COLOR_BGR2GRAY)
            h, w = frame.shape
            vals["整張"].append(edge_sigma(frame))
            vals["床位"].append(edge_sigma(frame[int(slot[1] * h):int(slot[3] * h), int(slot[0] * w):int(slot[2] * w)]))
        i += 1
    cap.release()
    for k, v in vals.items():
        sx = np.nanmedian([e["x"][0] for e in v])
        sy = np.nanmedian([e["y"][0] for e in v])
        print(f"直邊模糊（{k}，{len(v)} 幀）：σx={sx:.2f}（{sum(e['x'][1] for e in v)} 條邊）"
              f"  σy={sy:.2f}（{sum(e['y'][1] for e in v)} 條邊）")
    print(f"仰躺正臉的臉部範圍：{real_supine(args.video, slot, args.key)}")


def cmd_calibrate(args):
    """用同一隻娃娃的影片，對齊仰躺正臉的臉部範圍：先色調（對比、亮度、飽和度），再壓縮強度（qp 決定細節剩多少）。"""
    slot = tuple(float(v) for v in args.slot.split(","))
    real = real_supine(args.video, slot, args.key)
    print(f"真實：{real}", flush=True)
    anns = video_labels.load()
    preset = {**PRESETS[args.preset], **IDENTITY_TONE}
    samples = supine_samples(args.videos, anns, preset)
    print(f"取樣 {len(samples)} 張（仰躺正臉）", flush=True)
    if args.tone:
        c, b, s = (float(v) for v in args.tone.split(","))
        preset.update(contrast=c, brightness=b, saturation=s)
    for it in range(0 if args.tone else args.iters):
        sim = sim_supine(samples, preset)
        print(f"  第 {it + 1} 輪 {({k: preset[k] for k in ('contrast', 'brightness', 'saturation', 'qp')})} → {sim}", flush=True)
        # 色調是線性的：亮度標準差跟著 contrast 走、平均跟著 brightness 走；飽和度用比例修正
        c = preset["contrast"] * real["std"] / sim["std"]
        pre_mean = (sim["mean"] - 128 - preset["brightness"]) / preset["contrast"] + 128   # 調色前的平均
        preset.update(contrast=round(c, 3), brightness=round(real["mean"] - 128 - c * (pre_mean - 128), 1),
                      saturation=round(preset["saturation"] * real["sat"] / max(sim["sat"], 1e-3), 3))
    print("色調：" + str({k: preset[k] for k in ("contrast", "brightness", "saturation")}), flush=True)
    # 細節：壓縮只會讓細節變少；真實鏡頭的臉細節反而比手機影片縮小後多（監視器有銳化），所以調銳化強度
    best = None
    for a in args.sharpen:
        sim = sim_supine(samples, {**preset, "sharpen": a})
        print(f"  sharpen={a} → {sim}", flush=True)
        if sim and (best is None or abs(sim["detail"] - real["detail"]) < best[0]):
            best = (abs(sim["detail"] - real["detail"]), a)
    preset["sharpen"] = best[1]
    sim = sim_supine(samples, preset)
    print(f"最後：{({k: preset[k] for k in ('contrast', 'brightness', 'saturation', 'sharpen', 'qp')})} → {sim}")
    print(f"建議：contrast={preset['contrast']} brightness={preset['brightness']} saturation={preset['saturation']} "
          f"sharpen={preset['sharpen']}（qp={preset['qp']}；細節最接近真實的 {real['detail']}）")


def cmd_render(args):
    anns = video_labels.load()
    names = args.videos or sorted(v for v, a in anns.items() if not a["local"] and v != "嬰兒翻身")
    for name in names:
        spec = render(name, anns[name], args.preset)
        print(f"{name}：臉寬 {spec['src_face_px']:.0f} px → ×{spec['scale']:.3f}，床位 {spec['slots']}", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="量真實錄影")
    c = sub.add_parser("calibrate", help="用同一隻娃娃的影片對齊色調和壓縮強度")
    for p in (m, c):
        p.add_argument("--video", required=True, help="真實錄影（rtsp_check --record 存的 .mp4）")
        p.add_argument("--slot", required=True, help="床位範圍 x0,y0,x1,y1（畫面比例）")
        p.add_argument("--key", default="2", help="仰躺正臉的按鍵（rtsp_check 的 LABELS，預設 2）")
    m.add_argument("--every", type=int, default=25)
    c.add_argument("--preset", default="daycare", choices=PRESETS)
    c.add_argument("--iters", type=int, default=3)
    c.add_argument("--sharpen", type=float, nargs="+", default=[0, 0.5, 1.0, 1.5, 2.0, 3.0])
    c.add_argument("--tone", help="跳過色調對齊，直接用這組 contrast,brightness,saturation")
    c.add_argument("videos", nargs="+", help="同一隻娃娃的影片（只用仰躺正臉的片段）")
    r = sub.add_parser("render", help="做模擬影片")
    r.add_argument("--preset", default="daycare", choices=PRESETS)
    r.add_argument("videos", nargs="*")
    args = ap.parse_args()
    {"measure": cmd_measure, "calibrate": cmd_calibrate, "render": cmd_render}[args.cmd](args)


if __name__ == "__main__":
    main()
