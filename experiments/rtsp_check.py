"""
rtsp_check.py
場域測試：讀一路 RTSP 串流（例如海康 DVR 某一路），量連線；加 --infer 跑 v3＋床位判讀，邊測邊用數字鍵標「現在擺的是什麼」，
每次判讀寫一列 CSV，結束時印出每種擺法的結果。預設不存任何畫面；--record 才存判讀用的原始畫面（只在畫面裡只有娃娃時用）。
--infer、--record 只對空床或娃娃用：真實嬰兒要等 IRB 和家長同意。

串流網址含帳密，放環境變數，不要寫進檔案：
    PowerShell:  $env:FG_RTSP = "rtsp://帳號:密碼@IP:埠/h264/ch4/main/av_stream"
    （海康舊款：/h264/chN/main|sub/av_stream；新款：/Streaming/Channels/N01（主）、N02（子）；
      密碼裡有 @ # : % 這類符號要改成 %40 %23 %3A %25）

    uv run python -m experiments.rtsp_check                         # 只量連線 10 秒，不開視窗
    uv run python -m experiments.rtsp_check --show                  # 開視窗看畫面
    uv run python -m experiments.rtsp_check --pick-slot --name cam04_main      # 拖框選嬰兒床，印出 --slot 數字後開始判讀
    uv run python -m experiments.rtsp_check --infer --name cam04_main --slot 0.31,0.42,0.55,0.80 --rot 0
    uv run python -m experiments.rtsp_check --infer --name cam04_main --slot ... --record     # 同時存原始畫面（只有娃娃時）
    uv run python -m experiments.rtsp_check --channels 1-16 --seconds 30 --name dvr_sub       # 同時讀 16 路（只解碼，不判讀、不顯示、不存）

視窗裡的按鍵（標「現在擺的是什麼」，寫進 CSV；換擺法時手還在畫面裡就先按 0）：
    0 排除（有人在擺、畫面有人）  1 空床  2 仰躺、不蓋東西  3 側躺、不蓋東西（口鼻看不看得到之後對照畫面標）  4 趴睡、臉朝下
    5 仰躺、被子蓋過口鼻（眼睛露出）  6 仰躺、被子蓋到胸口（臉露出）  7 仰躺、被子蓋住整張臉
    8 仰躺、被子蓋到胸口、頭轉側邊（側臉）  9 仰躺、小方巾在臉旁邊（口鼻露出）  c 仰躺、小方巾蓋住口鼻（眼睛露出）
    q／Esc 結束
輸出：outputs/field/<日期>/<name>_<時間>.csv（每次判讀一列）、.txt（結果摘要）、--record 時另存 .mp4（每秒 5 幀的原始畫面）。
--channels：網址裡的第幾路（/h264/chN/ 或 /Streaming/Channels/N02）會自動換成每一路；量每路實際 fps、斷線、總頻寬、CPU，
寫到 outputs/field/<日期>/<name>_multi<路數>_<時間>.txt。回答「一台主機同時從 DVR 讀幾路撐得住」（判讀的運算量另外看 bench_multistream）。
RTMO 要在 UGen300 上跑就先設 $env:FG_RTMO_HEF = "model/rtmo_D1.hef"（沒設就在 CPU 跑）。
"""

import argparse
import csv
import os
import re
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")  # 要在 import cv2 前設；UDP 在 Wi-Fi 上常掉封包

import cv2  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "outputs" / "field"
VIEW_W, VIEW_H = 1280, 760  # 視窗最大寬高：直式影片（1080×1920）也要整張放得進螢幕
INFER_EVERY = 0.2  # 每 0.2 秒判讀一次（每秒 5 幀，跟評估一樣）
LABELS = {  # 按鍵 → (英文，畫面用；中文，CSV 用；應該的判定)
    "0": ("exclude", "排除（有人在擺）", None),
    "1": ("empty crib", "空床", None),
    "2": ("supine, nothing on", "仰躺、不蓋東西", "visible"),
    # 側躺時口鼻看不看得到要看臉朝哪邊，之後對照畫面標（看不到但沒被蓋＝口鼻沒蓋、鏡頭看不到，不算對錯）
    "3": ("side-lying, nothing on", "側躺、不蓋東西", None),
    "4": ("prone, face down", "趴睡、臉朝下", "hidden"),
    "5": ("supine, mouth+nose covered", "仰躺、被子蓋過口鼻（眼睛露出）", "hidden"),
    "6": ("supine, blanket to chest", "仰躺、被子蓋到胸口（臉露出）", "visible"),
    "7": ("supine, whole face covered", "仰躺、被子蓋住整張臉", "hidden"),
    "8": ("supine, blanket to chest, head turned", "仰躺、被子蓋到胸口、頭轉側邊（側臉）", "visible"),
    # 2026/11/15 加：嬰兒室常見臉旁邊放一條小方巾（第二間月子中心的畫面）
    "9": ("supine, cloth beside face", "仰躺、小方巾在臉旁邊（口鼻露出）", "visible"),
    "c": ("supine, cloth over mouth+nose", "仰躺、小方巾蓋住口鼻（眼睛露出）", "hidden"),
}


class Reader(threading.Thread):
    """背景一直讀、只留最新一幀：判讀比串流慢時，畫面不會越積越延遲。讀不到就重連。"""

    def __init__(self, url, lazy=False):
        super().__init__(daemon=True)
        self.url, self.frame, self.n, self.fails, self.reconnects = url, None, 0, 0, 0
        self.lock, self.stopped = threading.Lock(), False
        self.cap = None if lazy else cv2.VideoCapture(url, cv2.CAP_FFMPEG)  # lazy：在執行緒裡才連（多路同時連）

    def run(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        while not self.stopped:
            ok, f = self.cap.read()
            if not ok:
                self.fails += 1
                time.sleep(0.5)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                self.reconnects += 1
                continue
            with self.lock:
                self.frame = f
                self.n += 1

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


def channel_url(url, ch):
    """把網址裡的「第幾路」換成 ch：{ch}、海康舊款 /chN/、新款 /Channels/N01|N02。"""
    if "{ch}" in url:
        return url.replace("{ch}", str(ch))
    new, n = re.subn(r"/ch\d+/", f"/ch{ch}/", url)  # 用 subn 看有沒有換到：網址本來就是第 ch 路時換完一樣
    if n:
        return new
    new, n = re.subn(r"/Channels/\d+?(0[12])$", lambda m: f"/Channels/{ch}{m.group(1)}", url)
    if n:
        return new
    raise SystemExit("網址裡找不到第幾路（/chN/ 或 /Channels/N02），請在那個位置寫 {ch}")


def parse_channels(spec):
    """'1-16' 或 '1,3,5' 或 '1-4,9' → [1, 2, ...]"""
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def run_multi(url, chans, seconds, name):
    """同時讀很多路，只解碼：每路實際 fps、失敗／重連次數、總頻寬、CPU。不判讀、不顯示、不存畫面。"""
    try:
        import psutil
    except ImportError:
        psutil = None
    readers = {ch: Reader(channel_url(url, ch), lazy=True) for ch in chans}
    t0 = time.time()
    for r in readers.values():
        r.start()
    while time.time() - t0 < 30 and any(r.frame is None for r in readers.values()):
        time.sleep(0.1)
    up = {ch: r for ch, r in readers.items() if r.frame is not None}
    print(f"{len(up)}/{len(chans)} 路在 {time.time() - t0:.1f} 秒內收到第一幀" +
          ("" if len(up) == len(chans) else f"；沒收到：{[ch for ch in chans if ch not in up]}"))
    n0 = {ch: r.n for ch, r in readers.items()}
    f0 = {ch: r.fails for ch, r in readers.items()}
    if psutil:
        psutil.cpu_percent(None)
        net0 = psutil.net_io_counters().bytes_recv
    t1 = time.time()
    time.sleep(seconds)
    el = time.time() - t1
    cpu = psutil.cpu_percent(None) if psutil else None
    mbps = (psutil.net_io_counters().bytes_recv - net0) * 8 / el / 1e6 if psutil else None
    for r in readers.values():
        r.stopped = True
    lines = [f"同時讀 {len(chans)} 路，量 {el:.0f} 秒（只解碼，不判讀）",
             "路 | 解析度 | 收到幀數 | 實際 fps | 讀取失敗 | 重連"]
    rates = []
    for ch, r in readers.items():
        n = r.n - n0[ch]
        rates.append(n / el)
        shape = "—" if r.frame is None else f"{r.frame.shape[1]}×{r.frame.shape[0]}"
        lines.append(f"{ch} | {shape} | {n} | {n / el:.1f} | {r.fails - f0[ch]} | {r.reconnects}")
    lines.append(f"平均每路 {statistics.mean(rates):.1f} fps、最低 {min(rates):.1f} fps"
                 + (f"；筆電 CPU {cpu:.0f}%；收到的總頻寬約 {mbps:.1f} Mbps" if psutil else ""))
    text = "\n".join(lines)
    print(text)
    stamp = datetime.now()
    day = OUT_ROOT / stamp.strftime("%Y%m%d")
    day.mkdir(parents=True, exist_ok=True)
    out = day / f"{name}_multi{len(chans)}_{stamp:%H%M%S}.txt"
    out.write_text(f"{channel_url(url, chans[0]).split('@')[-1]}（第 {chans[0]} 路的網址，其他路同格式）\n{text}\n",
                   encoding="utf-8")
    print(f"→ {out}")


def make_infer(slot, rot):
    """載入 RTMO 和 YuNet，回傳 infer(frame) → 這一幀的判讀結果（dict）。"""
    from pipeline import crib_detect, infant_face_v3, infant_face_v5
    from pipeline.face_common import FACE_SCORE_TH, create_face_detector
    infant_face_v3.DET_TH = 0.15  # 跟 eval_video.py 一樣
    run_pose = infant_face_v3.load_rtmo()
    det = create_face_detector(FACE_SCORE_TH)
    det_v5 = create_face_detector(infant_face_v5.FACE_TH_V5)
    x0, y0, x1, y1 = slot

    def infer(frame):
        t = time.time()
        res, persons, faces = crib_detect.detect(run_pose, det, det_v5, frame, {"S": slot}, {"S": rot})
        ms = (time.time() - t) * 1000
        h, w = frame.shape[:2]
        boxes = [tuple(int(v) for v in f[:4]) for f in faces]
        in_slot = [b for b in boxes if x0 * w <= (b[0] + b[2]) / 2 <= x1 * w and y0 * h <= (b[1] + b[3]) / 2 <= y1 * h]
        r = res["S"]
        return {"verdict": r["v3_slot"], "v3_person": r["v3"], "persons": len(persons), "persons_in_slot": r["persons"],
                "faces_in_slot": len(in_slot), "face_px": max((b[2] - b[0] for b in in_slot), default=0),
                "face_scores": r["slot_face_scores"], "ms": ms,
                "person_boxes": [tuple(int(v) for v in p["box"][:4]) for p in persons], "face_boxes": boxes}

    return infer


def pick_slot(frame):
    """拖框選嬰兒床範圍，回傳 0–1 的 (x0, y0, x1, y1)。"""
    s = min(1.0, VIEW_W / frame.shape[1], VIEW_H / frame.shape[0])
    view = cv2.resize(frame, (round(frame.shape[1] * s), round(frame.shape[0] * s)))
    x, y, w, h = cv2.selectROI("drag the crib, then Enter (c = cancel)", view, showCrosshair=False)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        raise SystemExit("沒有選到範圍")
    W, H = view.shape[1], view.shape[0]
    return round(x / W, 3), round(y / H, 3), round((x + w) / W, 3), round((y + h) / H, 3)


def draw(frame, slot, out, key):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = slot
    th = max(2, w // 640)
    cv2.rectangle(frame, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (255, 200, 0), th)  # 床位：淺藍
    if out:
        for b in out["person_boxes"]:
            cv2.rectangle(frame, b[:2], b[2:], (0, 215, 255), th)  # 找到的人：黃
        for b in out["face_boxes"]:
            cv2.rectangle(frame, b[:2], b[2:], (255, 255, 255), th)  # 找到的臉：白
    s = min(VIEW_W / w, VIEW_H / h)
    view = cv2.resize(frame, (round(w * s), round(h * s)))
    lines = [f"[{key}] {LABELS[key][0]}" if key else "[no label] press 0-9 or c"]
    color = (200, 200, 200)
    if out:
        ok = out["verdict"] == "visible"
        color = (50, 160, 46) if ok else (32, 45, 217)
        cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), color, 8)
        lines.append(f"crib: mouth/nose {'visible' if ok else 'NOT visible'}")
        lines.append(f"persons {out['persons']} (in crib {out['persons_in_slot']})  face {out['face_px']}px  "
                     f"{out['ms']:.0f} ms")
    fs = 0.8 if view.shape[1] >= 800 else 0.5  # 直式影片的視窗比較窄，字要小一點才放得下
    for i, text in enumerate(lines):
        y = int((34 + 30 * i) * fs / 0.8)
        cv2.putText(view, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 4)
        cv2.putText(view, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, fs, color if i else (255, 255, 255), 2)
    return view


def summarize(rows):
    """每種擺法：幾次判讀、判看得到幾成、找到人幾成、臉多大、多慢。"""
    lines = ["擺法 | 判讀次數 | 判「看得到」 | 判「看不到」 | 該有的判定對了 | 找到人（床位裡） | 臉寬中位數 px | 平均毫秒"]
    for key, (_, zh, want) in LABELS.items():
        rs = [r for r in rows if r["key"] == key]
        if not rs:
            continue
        n = len(rs)
        vis = sum(r["verdict"] == "visible" for r in rs)
        right = "—" if want is None else f"{sum(r['verdict'] == want for r in rs) / n:.0%}"
        found = sum(r["persons_in_slot"] > 0 for r in rs)
        px = [r["face_px"] for r in rs if r["face_px"]]
        lines.append(f"{key} {zh} | {n} | {vis / n:.0%} | {(n - vis) / n:.0%} | {right} | {found / n:.0%} | "
                     f"{statistics.median(px) if px else '—'} | {statistics.mean(r['ms'] for r in rs):.0f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10, help="不開視窗時跑幾秒（預設 10）")
    ap.add_argument("--show", action="store_true", help="開視窗看畫面")
    ap.add_argument("--infer", action="store_true", help="跑判讀（只對空床或娃娃）")
    ap.add_argument("--pick-slot", action="store_true", help="先拖框選嬰兒床，再開始判讀")
    ap.add_argument("--slot", default="0,0,1,1", help="嬰兒床範圍 x0,y0,x1,y1（0–1，預設整個畫面）")
    ap.add_argument("--rot", type=int, default=0, help="娃娃頭朝畫面哪邊：0 上、90 左、180 下、270 右（同床位檔的 rot）")
    ap.add_argument("--name", default="cam", help="檔名開頭，例如 cam04_main")
    ap.add_argument("--label", default=None, choices=list(LABELS), help="一開始的擺法（不開視窗時用）")
    ap.add_argument("--record", action="store_true", help="另存判讀用的原始畫面（只在畫面裡只有娃娃時用）")
    ap.add_argument("--no-window", action="store_true", help="判讀但不開視窗，跑 --seconds 秒")
    ap.add_argument("--channels", help="同時讀很多路（只解碼），例如 1-16、1-4、1,3,5")
    args = ap.parse_args()
    url = os.environ.get("FG_RTSP")
    if not url:
        raise SystemExit("先設環境變數 FG_RTSP（見檔頭說明）")
    if args.channels:
        return run_multi(url, parse_channels(args.channels), args.seconds, args.name)
    host = url.split("@")[-1]  # 印出來、寫進檔案時都不帶帳密
    infer_on = args.infer or args.pick_slot

    t0 = time.time()
    reader = Reader(url)
    if not reader.cap.isOpened():
        raise SystemExit(f"連不上 {host}。檢查：筆電跟 DVR 在同一個網路、IP 和 RTSP 埠、帳密、DVR 有沒有開 RTSP。")
    fps_tag = reader.cap.get(cv2.CAP_PROP_FPS)
    reader.start()
    while reader.latest() is None and time.time() - t0 < 15:
        time.sleep(0.05)
    first = reader.latest()
    if first is None:
        raise SystemExit(f"連上 {host} 了，但 15 秒內沒有收到畫面")
    print(f"連上 {host}：第一幀 {time.time() - t0:.1f} 秒，解析度 {first.shape[1]}×{first.shape[0]}，串流標示 {fps_tag:.0f} fps")

    slot = tuple(float(v) for v in args.slot.split(","))
    if args.pick_slot:
        slot = pick_slot(first)
        print(f"床位範圍：--slot {','.join(f'{v:g}' for v in slot)}（下次可以直接用這組）")

    infer, log, writer, rec, stem = None, None, None, None, None
    if infer_on:
        stamp = datetime.now()
        day = OUT_ROOT / stamp.strftime("%Y%m%d")
        day.mkdir(parents=True, exist_ok=True)
        stem = day / f"{args.name}_{stamp.strftime('%H%M%S')}"
        infer = make_infer(slot, args.rot)
        log = open(f"{stem}.csv", "w", newline="", encoding="utf-8-sig")
        writer = csv.writer(log)
        writer.writerow(["clock", "t", "key", "label", "verdict", "v3_person", "persons", "persons_in_slot",
                         "faces_in_slot", "face_px", "face_scores", "ms", "stream", "width", "height", "slot", "rot",
                         "rtmo_hef"])
        if args.record:
            rec = cv2.VideoWriter(f"{stem}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 1 / INFER_EVERY,
                                  (first.shape[1], first.shape[0]))
        print(f"紀錄寫到 {stem}.csv" + ("（另存原始畫面 .mp4）" if rec else ""))

    window = (args.show or infer_on) and not args.no_window
    key = args.label
    rows, out, last_inf = [], None, 0.0
    t_start, n_start = time.time(), reader.n
    while True:
        if not window and time.time() - t_start >= args.seconds:
            break
        frame = reader.latest()
        if infer and time.time() - last_inf >= INFER_EVERY:
            last_inf = time.time()
            if rec is not None:
                rec.write(frame)
            out = infer(frame)
            row = {"key": key, **{k: out[k] for k in ("verdict", "persons_in_slot", "face_px", "ms")}}
            rows.append(row)
            writer.writerow([datetime.now().strftime("%H:%M:%S.%f")[:-3], f"{last_inf - t_start:.2f}", key or "",
                             LABELS[key][1] if key else "", out["verdict"], out["v3_person"], out["persons"],
                             out["persons_in_slot"], out["faces_in_slot"], out["face_px"], out["face_scores"],
                             f"{out['ms']:.0f}", host, frame.shape[1], frame.shape[0],
                             ",".join(f"{v:g}" for v in slot), args.rot, os.environ.get("FG_RTMO_HEF", "")])
        if not window:
            time.sleep(0.02)
            continue
        cv2.imshow("FirstGlance field test (0-8 label, q quit)", draw(frame, slot, out, key))
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if chr(k) in LABELS:
            key = chr(k)
            print(f"[{datetime.now():%H:%M:%S}] 擺法 → {key} {LABELS[key][1]}")
    el = time.time() - t_start
    n = reader.n - n_start
    print(f"{el:.0f} 秒收到 {n} 幀（實際 {n / el:.1f} fps），讀取失敗 {reader.fails} 次、重連 {reader.reconnects} 次")
    reader.stopped = True
    cv2.destroyAllWindows()
    if rec is not None:
        rec.release()
    if log:
        log.close()
        text = summarize(rows)
        Path(f"{stem}.txt").write_text(f"{host}  {first.shape[1]}×{first.shape[0]}  slot {slot}  rot {args.rot}  "
                                       f"RTMO {'UGen300' if os.environ.get('FG_RTMO_HEF') else 'CPU'}\n{text}\n",
                                       encoding="utf-8")
        print(text)
        print(f"→ {stem}.csv、.txt")


if __name__ == "__main__":
    main()
