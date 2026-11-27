"""
wall_clips.py
決賽展演播放清單（demo/wall_playlist.json）用到的 4K 影片轉成長邊 1920 的 H.264，給 live_wall.py 播（2026/11/28 寫）。
11/23 那批是手機直拍 2160×3840：OpenCV 每幀讀取約 100 ms（308 床的 1080p 約 7 ms），8 床一起播畫面會跟不上。

- 轉好的放 dataset/demo_wall/<影片>_1080.mp4（dataset/ 不進 repo），播放清單的 "files" 指向它；
  床位標註照原本那支（畫面比例座標，換解析度不影響）。播放清單沒列在 "files" 的影片不轉。
- 只縮小、重新壓縮，不剪、不改速度（片段和速度都寫在播放清單裡）：libx264 CRF 18、每秒一個關鍵幀（換段時跳轉比較快）、
  沒有聲音。手機影片的旋轉資訊 ffmpeg 會自己轉正（跟 OpenCV 讀到的方向一樣，轉完會檢查）。
- ffmpeg 用 imageio-ffmpeg 臨時掛（跟 evaluation/cctv_sim.py 一樣，不進 pyproject）。
- --check：原檔和轉檔版各跑一次主流程判讀（每秒 5 幀、跟 eval_video 同一批幀；預設 CPU），比每幀「口鼻看不看得到」
  有幾幀不一樣、各自判看不到的比例。結果印出來，也存 outputs/demo_wall/clips_check.json。

重的 CPU 工作（轉檔約 1–2 分鐘、--check 約 5 分鐘），晶片測試、長時間測試在跑時不要跑。
用法：
    uv run --with imageio-ffmpeg python -m demo.wall_clips             # 轉檔（已經轉好的跳過，--force 重轉）
    uv run python -m demo.wall_clips --check                           # 比對判定（--chip 改在晶片上跑）
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 要在 import cv2 前設

import cv2
import numpy as np

from evaluation import video_labels

ROOT = Path(__file__).resolve().parent.parent
PLAYLIST = ROOT / "demo" / "wall_playlist.json"
OUT_DIR = ROOT / "outputs" / "demo_wall"
LONG_SIDE = 1920
CHECK_FPS = 5


def targets(playlist):
    """播放清單 "files" 裡的 (影片名稱, 原檔, 轉檔版路徑)。"""
    cfg = json.loads(Path(playlist).read_text(encoding="utf-8"))
    out = []
    for video, file in cfg.get("files", {}).items():
        dst = Path(file) if Path(file).is_absolute() else ROOT / file
        out.append((video, video_labels.find_video(video), dst))
    return out


def first_frame(path, scale_to=None):
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.resize(frame, scale_to, interpolation=cv2.INTER_AREA) if scale_to else frame


def convert(video, src, dst, force=False):
    if dst.exists() and not force:
        print(f"  {video}：已經有 {dst.name}，跳過")
        return
    import imageio_ffmpeg
    dst.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    s = LONG_SIDE / max(w, h)
    tw, th = round(w * s / 2) * 2, round(h * s / 2) * 2
    tmp = dst.with_suffix(".part.mp4")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
           "-vf", f"scale={tw}:{th}:flags=area", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-g", str(max(1, round(fps))), "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(tmp)]
    tic = time.perf_counter()
    subprocess.run(cmd, check=True)
    tmp.replace(dst)
    # 檢查：方向、幀數、第一幀跟原檔縮小後差多少
    cap = cv2.VideoCapture(str(dst))
    dw, dh, dn = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    a, b = first_frame(src, (dw, dh)), first_frame(dst)
    diff = float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()) if a is not None and b is not None else None
    ok = (dw, dh) == (tw, th) and abs(dn - n) <= 2 and diff is not None and diff < 8
    print(f"  {video}：{w}×{h} → {dw}×{dh}，{n} → {dn} 幀，第一幀平均色差 {diff:.1f}，"
          f"{time.perf_counter() - tic:.0f} 秒{'' if ok else '  ← 不對，檢查一下'}")


def check(items, chip=False):
    """原檔和轉檔版各跑一次主流程判讀，比每幀判定。"""
    if chip:
        from demo.live_view import CHIP_FINALS
        for k in ("FG_YUNET_HEF", "FG_CHIP_BATCH"):
            os.environ.pop(k, None)
        os.environ.update(CHIP_FINALS)
    from pipeline import crib_detect, infant_face_v3, infant_face_v5, infant_face_v6, infant_face_v7
    from pipeline.face_common import FACE_SCORE_TH, create_face_detector
    infant_face_v3.DET_TH = 0.15
    run_pose = infant_face_v3.load_rtmo()
    det, det_v5 = create_face_detector(FACE_SCORE_TH), create_face_detector(infant_face_v5.FACE_TH_V5)
    mouth, retina = infant_face_v6.load_mouth_model(required=True), infant_face_v7.load_retina(required=True)
    ann = video_labels.load()

    def judge(path, doll, rect, rot):
        cap = cv2.VideoCapture(str(path))
        step = max(1, round(cap.get(cv2.CAP_PROP_FPS) / CHECK_FPS))
        out, i = [], 0
        while cap.grab():
            if i % step == 0:
                _, frame = cap.retrieve()
                result, _, _ = crib_detect.detect(run_pose, det, det_v5, frame, {doll: rect}, {doll: rot},
                                                  mouth=mouth, retina=retina)
                out.append(result[doll][crib_detect.MAIN_MODE] == "hidden")
            i += 1
        cap.release()
        return out

    report = []
    for video, src, dst in items:
        if not dst.exists():
            print(f"  {video}：還沒轉檔，略過")
            continue
        doll, slot = next(iter(ann[video]["slots"].items()))
        a = judge(src, doll, slot["rect"], slot["rot"])
        b = judge(dst, doll, slot["rect"], slot["rot"])
        n = min(len(a), len(b))
        flips = sum(x != y for x, y in zip(a[:n], b[:n]))
        row = {"video": video, "frames": n, "flips": flips, "hidden_original": round(sum(a) / max(1, len(a)), 3),
               "hidden_converted": round(sum(b) / max(1, len(b)), 3)}
        report.append(row)
        print(f"  {video}：{n} 幀，判定不同 {flips} 幀；判看不到 原檔 {row['hidden_original']:.0%}、轉檔版 {row['hidden_converted']:.0%}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"clips_check{'_chip' if chip else ''}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {path}")


def main():
    ap = argparse.ArgumentParser(description="播放清單用到的 4K 影片轉成長邊 1920（給 live_wall.py）")
    ap.add_argument("--playlist", default=str(PLAYLIST))
    ap.add_argument("--force", action="store_true", help="已經轉好的也重轉")
    ap.add_argument("--check", action="store_true", help="不轉檔，改成比對原檔和轉檔版的判定")
    ap.add_argument("--chip", action="store_true", help="--check 在晶片上跑（live_view.CHIP_FINALS）")
    args = ap.parse_args()
    items = targets(args.playlist)
    print(f"播放清單 files 列了 {len(items)} 支")
    if args.check:
        return check(items, args.chip)
    for video, src, dst in items:
        convert(video, src, dst, args.force)


if __name__ == "__main__":
    main()
