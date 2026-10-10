"""
demo_ui/export_assets.py
把系統對娃娃影片的真實判定匯出給 demo_ui/（可以點的護理站介面原型，給影片的操作示範錄螢幕用）。

每床（demo_render.GRID 的 8 床，07 號是主角）：
  - 燈號：demo_render.timeline_states()（eval_video.py 的每幀判定＋crib_context.py 的照護中／空床狀態機
    ＋vlm_spike.py --every 1 的 VLM 時間軸，有的話）每 STEP 秒取一次，連同四個訊號（偵測器、VLM、大人在床邊、像空床）
  - 影片：把那一段剪出來、縮到 CLIP_H 高，存成 VP8 .webm（OpenCV 內建就能寫、瀏覽器都能播，不用另裝 ffmpeg）
輸出成 data/cribs.js（網頁用：直接用 <script> 載入，雙擊 index.html 用 file:// 開也能讀）和 data/cribs.json（同內容，給人看）。

用法（在 repo 根目錄）：
    uv run python demo_ui/export_assets.py            # 真資料 → demo_ui/data/、demo_ui/media/（不進 repo）
    uv run python demo_ui/export_assets.py --sample   # 假資料（合成畫面＋寫死的燈號）→ demo_ui/sample/（進 repo，雲端開發用）
要先跑過 eval_video.py（每幀判定）；有 vlm_spike.py --every 1 的結果才會有黃燈「請看一眼」。
然後用瀏覽器開 demo_ui/index.html（沒有 data/ 時自動改用 sample/；網址加 ?sample 強制用假資料）。
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import demo_render  # noqa: E402
import video_labels  # noqa: E402

UI_DIR = Path(__file__).resolve().parent
STEP = 0.2       # 燈號每幾秒一筆（跟每秒 5 幀的判定一樣密）
CLIP_H = 720     # 剪出來的影片高度（單床詳細畫面會放大顯示）


def hex_color(rgb):
    return "#%02x%02x%02x" % rgb


def open_writer(path, fps, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"VP80"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV 無法寫 VP8 .webm：{path}")
    return writer


def cut_clip(video, start, end, out, height=CLIP_H):
    """把 video 的 start–end 秒剪出來、縮到 height 高，存成 .webm。回傳實際長度（秒）。"""
    cap = cv2.VideoCapture(str(video_labels.find_video(video)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    writer, n = None, 0
    while True:
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        ok, frame = cap.read()
        if not ok or t > end:
            break
        s = height / frame.shape[0]
        frame = cv2.resize(frame, (round(frame.shape[1] * s) // 2 * 2, height), interpolation=cv2.INTER_AREA)
        if writer is None:
            writer = open_writer(out, fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        n += 1
    cap.release()
    if writer is not None:
        writer.release()
    return n / fps


def resample(rows, start, end, step=STEP):
    """timeline_states 的結果 → 這一段每 step 秒一筆（取那個時刻之前最後一個取樣），欄位分開存成陣列。"""
    samples = [(r["t"], r) for r in rows]
    out = {"show": [], "hidden": [], "count": [], "adult": [], "matches": [], "vlm": []}
    for k in range(int(round((end - start) / step)) + 1):
        r = demo_render.latest(samples, start + k * step + 1e-6) or samples[0][1]
        out["show"].append(r["show"])
        out["hidden"].append(int(r["hidden"]))
        out["count"].append(round(r["count"], 2))
        out["adult"].append(int(r["adult"]))
        out["matches"].append(int(r["matches"]))
        out["vlm"].append(None if r["vlm"] is None else round(r["vlm"], 3))
    return out


def base_data(sample):
    hero = demo_render.GRID[6][0]
    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "sample": sample,
        "step": STEP,
        "clock0": demo_render.CLOCK0,
        "hero": hero,
        "vlm_th": demo_render.VLM_TH,
        "states": {k: {"color": hex_color(c), "title": title, "desc": desc}
                   for k, (c, title, desc) in demo_render.STATES.items()},
        "event_text": demo_render.EVENT_TEXT,
        "cribs": [],
    }


def write_data(folder, data):
    folder.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (folder / "cribs.json").write_text(text, encoding="utf-8")
    (folder / "cribs.js").write_text(f"window.FIRSTGLANCE_DATA = {text};\n", encoding="utf-8")
    print(f"→ {folder / 'cribs.js'}（{len(data['cribs'])} 床）")


def export_real():
    anns = video_labels.load()
    timelines = {}
    for video in sorted({g[1] for g in demo_render.GRID}):
        slot = next(iter(anns[video]["slots"].values()))
        timelines[video] = demo_render.timeline_states(video, slot, max(g[3] for g in demo_render.GRID if g[1] == video))
    data = base_data(sample=False)
    for label, video, start, end in demo_render.GRID:
        rel = f"media/crib{label}.webm"
        dur = cut_clip(video, start, end, UI_DIR / rel)
        data["cribs"].append({"id": label, "video": rel, "source": video, "start": start, "end": end,
                              "duration": round(dur, 2), **resample(timelines[video], start, end)})
        print(f"  Crib {label}：{video} {start:g}–{end:g} 秒 → {rel}")
    write_data(UI_DIR / "data", data)


# ── 假資料：合成畫面，燈號寫死成跟真資料差不多的走向（雲端沒有娃娃影片也能開發網頁） ──
SAMPLE_SCRIPT = {  # 床號 → [(到第幾秒, 顯示狀態), ...]
    "01": [(44, "green")], "02": [(19, "green")], "03": [(31, "green")], "04": [(12, "empty")],
    "05": [(12.5, "green")], "06": [(16.5, "green")],
    "07": [(3.4, "green"), (12.8, "care"), (15.8, "count"), (32, "red")],
    "08": [(17, "look")],
}
SAMPLE_FPS = 5


def sample_frame(label, show, t, w=270, h=480):
    """合成一幀：床、娃娃（看得到臉／被蓋住）、照護中有一隻手在動；空床就只有床。"""
    img = np.full((h, w, 3), (205, 214, 226), np.uint8)
    cv2.rectangle(img, (14, 14), (w - 14, h - 14), (150, 168, 188), 5)
    if show != "empty":
        hidden = show in ("count", "red")
        cv2.ellipse(img, (w // 2, 150), (46, 54), 0, 0, 360, (165, 195, 235), -1)
        if not hidden:
            for dx in (-16, 16):
                cv2.circle(img, (w // 2 + dx, 142), 5, (50, 50, 50), -1)
            cv2.ellipse(img, (w // 2, 172), (12, 6), 0, 0, 180, (60, 60, 150), 2)
        top = 120 if hidden else 205
        cv2.rectangle(img, (40, top), (w - 40, h - 40), (190, 140, 90), -1)
        if show == "care":
            x = int(w / 2 + 90 * np.sin(t * 2.5))
            cv2.circle(img, (x, 90), 32, (130, 160, 205), -1)
            cv2.rectangle(img, (x - 22, 0), (x + 22, 90), (130, 160, 205), -1)
    cv2.putText(img, f"SAMPLE  Crib {label}", (22, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    cv2.putText(img, f"{int(t):3d}s", (w - 70, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    return img


def export_sample():
    folder = UI_DIR / "sample"
    data = base_data(sample=True)
    for label, video, start, end in demo_render.GRID:
        script = SAMPLE_SCRIPT[label]
        dur = end - start

        def show_at(t):
            return next((s for until, s in script if t < until), script[-1][1])

        rel = f"sample/media/crib{label}.webm"
        writer = open_writer(UI_DIR / rel, SAMPLE_FPS, (270, 480))
        for n in range(int(dur * SAMPLE_FPS)):
            writer.write(sample_frame(label, show_at(n / SAMPLE_FPS), n / SAMPLE_FPS))
        writer.release()
        rows, since = [], None
        for k in range(int(round(dur / STEP)) + 1):
            t = k * STEP
            show = show_at(t)
            hidden = show in ("count", "red", "empty")
            since = (t if since is None else since) if show in ("count", "red") else None
            rows.append({"t": t, "show": show, "hidden": hidden, "count": 0.0 if since is None else t - since,
                         "adult": show == "care", "matches": show == "empty",
                         "vlm": 0.2 if show in ("look", "count", "red") else (None if show == "empty" else 0.9)})
        data["cribs"].append({"id": label, "video": rel, "source": f"sample (as {video})", "start": start, "end": end,
                              "duration": dur, **resample(rows, 0, dur)})
    write_data(folder, data)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--sample", action="store_true", help="產生假資料到 demo_ui/sample/（不需要娃娃影片）")
    args = ap.parse_args()
    export_sample() if args.sample else export_real()


if __name__ == "__main__":
    main()
