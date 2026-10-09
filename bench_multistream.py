"""
bench_multistream.py
CPU 多路負載測試：一台主機最多能同時判讀幾支每床鏡頭？（晶片 10/26 才到，先量 CPU 當下限）

月子中心嬰兒室是每床一支鏡頭，整個畫面就是一個床位，所以每一幀要做的是：
  full：v3＋床位（eval_video.py 的 v3_slot，娃娃影片 85% 那組數字用的就是這個）
        RTMO 找人 → 整張圖多尺度 YuNet → 每人頭部轉正 YuNet → 整個畫面當床位再找一次臉
  slot：只做最後一步（不跑 RTMO）——每床鏡頭只拍一個寶寶，找人那步可能省得掉，量看看省多少
        （注意：這個組合還沒跑過準確率評估，數字只代表運算量）

量三件事：
  1. 單一程序每幀延遲（各步驟拆開，ONNX Runtime／OpenCV 用預設執行緒）
  2. 多程序吞吐量：W 個工作程序各自載入模型、每個限 1 條執行緒，同時跑 --sec 秒，
     加總每秒處理幾幀 → 每路 1 或 2 FPS 時撐得住幾路
  3. 解碼：RTSP 串流不能跳著解（H.264/H.265 要從關鍵幀往後解），所以每路都要解完整幀率，
     量一支影片在這台機器上每秒解得出幾幀

測試幀取自自己拍的娃娃影片 IMG_8115（橫式 1080p、單一娃娃俯拍，最接近每床鏡頭），縮到 --res 指定的高度。

用法：
    uv run python bench_multistream.py                         # 720p、1080p 都跑，W = 1 2 4 8 12 16
    uv run python bench_multistream.py --res 720 --workers 1 8 --sec 10   # 試跑
輸出：outputs/bench/cpu_multistream.csv（每種設定一列）、終端機摘要
"""

import argparse
import csv
import multiprocessing as mp
import os
import platform
import statistics
import time
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 要在 import cv2 前設

import cv2
import numpy as np

ROOT = Path(__file__).parent
VIDEO = ROOT / "dataset" / "doll_video" / "IMG_8115.MOV"
OUT_DIR = ROOT / "outputs" / "bench"
N_FRAMES = 40          # 從影片平均取這麼多幀，輪流餵給模型
WHOLE = (0.0, 0.0, 1.0, 1.0)


def limit_threads(n):
    """讓之後建立的 ONNX Runtime session 和 OpenCV 只用 n 條執行緒（rtmlib 建 session 時不給設定，只好包一層）。"""
    import onnxruntime as ort
    orig = ort.InferenceSession

    def session(path_or_bytes, sess_options=None, providers=None, **kw):
        so = sess_options or ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = n, 1
        return orig(path_or_bytes, sess_options=so, providers=providers, **kw)

    ort.InferenceSession = session
    cv2.setNumThreads(n)


def load_frames(height):
    cap = cv2.VideoCapture(str(VIDEO))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in np.linspace(0, total - 1, N_FRAMES).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            s = height / frame.shape[0]
            frames.append(cv2.resize(frame, (round(frame.shape[1] * s), height), interpolation=cv2.INTER_AREA))
    cap.release()
    return frames


def load_models():
    from face_common import FACE_MODEL, FACE_SCORE_TH
    from infant_face_v3 import load_rtmo
    run_pose = load_rtmo()
    detector = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), FACE_SCORE_TH)
    return run_pose, detector


def process(run_pose, detector, frame, mode, timing=None):
    """處理一幀，回傳這個床位「看不到口鼻」與否。timing 給 dict 就記錄各步驟秒數。"""
    from eval_video import slot_faces
    from face_common import assign_faces, detect_faces
    from infant_face_v3 import head_faces, make_persons

    def tick(name, t0):
        if timing is not None:
            timing.setdefault(name, []).append(time.perf_counter() - t0)
        return time.perf_counter()

    t = time.perf_counter()
    visible = False
    if mode == "full":
        persons = make_persons(run_pose(frame))
        t = tick("rtmo", t)
        faces = detect_faces(detector, frame)
        t = tick("yunet_whole", t)
        faces += [f for p in persons for f in head_faces(detector, frame, p)]
        t = tick("yunet_head", t)
        assign_faces(persons, faces)
        visible = any(p["faces"] for p in persons)
    visible = bool(slot_faces(detector, frame, WHOLE, 0)) or visible
    tick("yunet_slot", t)
    return not visible


def latency(frames, mode, repeat):
    run_pose, detector = load_models()
    for f in frames[:3]:  # 暖機
        process(run_pose, detector, f, mode)
    timing, totals = {}, []
    for _ in range(repeat):
        for f in frames:
            t0 = time.perf_counter()
            process(run_pose, detector, f, mode, timing)
            totals.append(time.perf_counter() - t0)
    ms = {k: 1000 * statistics.median(v) for k, v in timing.items()}
    ms["total"] = 1000 * statistics.median(totals)
    ms["total_p95"] = 1000 * float(np.percentile(totals, 95))
    return ms


def worker(height, mode, start, sec, counter):
    limit_threads(1)
    frames = load_frames(height)  # 各自解碼，不用把整批幀傳過去
    run_pose, detector = load_models()
    for f in frames[:2]:
        process(run_pose, detector, f, mode)
    start.wait()  # 全部載好模型才一起開始計時
    end, i, n = time.perf_counter() + sec, 0, 0
    while time.perf_counter() < end:
        process(run_pose, detector, frames[i % len(frames)], mode)
        i, n = i + 1, n + 1
    with counter.get_lock():
        counter.value += n


def throughput(height, mode, workers, sec):
    import psutil
    start, counter = mp.Barrier(workers + 1), mp.Value("i", 0)
    procs = [mp.Process(target=worker, args=(height, mode, start, sec, counter)) for _ in range(workers)]
    for p in procs:
        p.start()
    start.wait()
    psutil.cpu_percent(None)
    t0 = time.perf_counter()
    for p in procs:
        p.join()
    elapsed = time.perf_counter() - t0
    cpu = psutil.cpu_percent(None)
    return counter.value / min(elapsed, sec) if elapsed else 0.0, cpu


def decode_fps(height, sec=10):
    """預設執行緒下，解碼 IMG_8115 並縮到 height 的速度（每秒幾幀）。"""
    import psutil
    cap = cv2.VideoCapture(str(VIDEO))
    psutil.cpu_percent(None)
    t0, n = time.perf_counter(), 0
    while time.perf_counter() - t0 < sec:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        if frame.shape[0] != height:
            s = height / frame.shape[0]
            cv2.resize(frame, (round(frame.shape[1] * s), height), interpolation=cv2.INTER_AREA)
        n += 1
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
    return n / (time.perf_counter() - t0), psutil.cpu_percent(None), fourcc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, nargs="+", default=[720, 1080], help="畫面高度（預設 720 1080）")
    ap.add_argument("--modes", nargs="+", choices=["full", "slot"], default=["full", "slot"])
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16], help="同時跑幾個工作程序")
    ap.add_argument("--sec", type=float, default=20, help="每種設定跑幾秒（預設 20）")
    ap.add_argument("--repeat", type=int, default=2, help="延遲測試把測試幀跑幾輪（預設 2）")
    args = ap.parse_args()

    import psutil
    print(f"CPU：{platform.processor()}｜實體核心 {psutil.cpu_count(False)}、邏輯核心 {psutil.cpu_count()}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for height in args.res:
        frames = load_frames(height)
        dec, dec_cpu, codec = decode_fps(height)
        print(f"\n== {height}p（{frames[0].shape[1]}×{height}，{len(frames)} 幀輪流）"
              f"｜解碼 {codec}：{dec:.0f} 幀/秒（CPU {dec_cpu:.0f}%）")
        for mode in args.modes:
            ms = latency(frames, mode, args.repeat)
            steps = "、".join(f"{k} {v:.0f}" for k, v in ms.items() if k not in ("total", "total_p95"))
            print(f"  [{mode}] 單程序每幀 {ms['total']:.0f} ms（p95 {ms['total_p95']:.0f}）：{steps} ms")
            for w in args.workers:
                fps, cpu = throughput(height, mode, w, args.sec)
                print(f"  [{mode}] {w:>2} 個程序 → 共 {fps:5.1f} 幀/秒（CPU {cpu:3.0f}%）"
                      f"＝每路 1 FPS 撐 {int(fps)} 路、2 FPS 撐 {int(fps / 2)} 路")
                rows.append({"height": height, "mode": mode, "workers": w, "fps_total": round(fps, 2),
                             "cpu_percent": round(cpu), "latency_ms": round(ms["total"]),
                             "latency_p95_ms": round(ms["total_p95"]), "decode_fps": round(dec),
                             "decode_cpu_percent": round(dec_cpu), "codec": codec,
                             **{f"{k}_ms": round(v, 1) for k, v in ms.items() if k not in ("total", "total_p95")}})
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(OUT_DIR / "cpu_multistream.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n→ {OUT_DIR / 'cpu_multistream.csv'}")


if __name__ == "__main__":
    main()
