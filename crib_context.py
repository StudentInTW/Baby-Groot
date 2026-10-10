"""
crib_context.py
每床鏡頭的床位情境（2026/10/10，原型）：照護中暫停計時、空床不響。

問題：每床鏡頭只拍一張床，床位模式「找不到臉就算看不到」→ 寶寶被抱去餵奶、拍嗝後，空床一定響（補拍影片 2/2）。
不讓護理師每次抱走都手動標記（太增加負擔），改成：
  1. 大人在床邊 → 照護中：不累積「看不到」的時間。判法：床位邊緣一圈有明顯移動（大人一定從畫面邊緣伸手進來），
     連續 3 幀有 2 幀以上就算，動作停了 CARE_HOLD 秒才算離開。先扣掉整個畫面的平移（鏡頭晃動）再比：
     手機拍的 IMG_8429 會晃，不扣的話沒有手的片段有 17% 幀被當成有人在動，扣掉後 5%（剩下的是有人在調整手機）。
     已經在響的警示不會因為有人在動而解除（移動也可能是寶寶在掙扎）；要嘛臉重新看得到，要嘛判成空床。
  2. 大人離開後 ARM 秒內，臉不見、而且畫面跟這張床的空床參考畫面相符（色差 < EMPTY_TH）連續 CONFIRM 秒 → 空床（灰燈）。
     不能只看「抓不到嬰兒」：空床上 RTMO 會在皺床單上找到人，被蓋住的寶寶又常常抓不到人，兩邊都會錯。
  3. 沒有大人來過就不可能變空床（寶寶不會自己離開）：看到臉之後就關掉空床判定，直到下一次有大人來。
     開機時還沒看過臉，允許直接判空床。
  不確定就走一般 3 秒計時——寧可多叫一次。

輸入：eval_video.py 存的每幀判定（outputs/video_eval/<影片>_frames.csv，用 v3＋床位；要先跑過、每秒 5 幀），
再讀一次影片算兩個便宜的特徵（都在縮成長邊 160 像素的小圖上算，比找臉便宜得多）：
  床位邊緣一圈的移動量（跟前一幀比）、跟空床參考畫面的色差。
空床參考畫面：床位檔的 empty_ref 欄（影片第幾秒是空床）；實機是安裝時拍一張。

用法：
    uv run python crib_context.py                    # 有空床參考的影片（補拍的 IMG_8429、IMG_8432）
    uv run python crib_context.py IMG_8115           # 其他單床影片（沒有空床參考的話，只有「照護中暫停」）
輸出：outputs/crib_context/<影片>_context.csv（每幀特徵與狀態）、<影片>_context.png（時間軸）、終端機摘要
"""

import argparse
import csv
import math
import os
import statistics
from collections import deque
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 要在 import cv2 前設

import cv2
import numpy as np

import video_labels
from eval_video import AlertState
from video_labels import read_csv

ROOT = Path(__file__).parent
EVAL_DIR = ROOT / "outputs" / "video_eval"
OUT_DIR = ROOT / "outputs" / "crib_context"
FPS = 5              # 要跟 eval_video.py 存每幀判定時一樣
SMALL = 160          # 特徵小圖的長邊
BAND = 0.12          # 床位邊緣一圈的寬度（床位寬、高的比例）
PIXEL_TH = 25        # 灰階差超過這麼多算「這個像素在動」
MOTION_TH = 0.02     # 邊緣一圈有這麼多比例的像素在動，這一幀算「邊緣有動作」
CARE_HOLD = 1.5      # 動作停了幾秒才算大人離開
ARM = 10             # 大人離開後幾秒內允許判空床
EMPTY_TH = 14        # 跟空床參考的平均色差低於這個算「相符」（IMG_8432：空床 3.9–9.4，有娃娃的片段最低 19.4）
CONFIRM = 2          # 相符要持續幾秒才判空床（比 3 秒警示短，才不會先閃紅燈）
COLORS = {"ok": (80, 180, 80), "alert": (60, 60, 220), "care": (200, 140, 40), "empty": (170, 170, 170)}


class CribState:
    """一張床的狀態：ok（綠）／alert（紅）／care（照護中，暫停計時）／empty（空床，灰）。"""

    def __init__(self, on=3, off=2, blip=1):
        self.alarm = AlertState(on, off, blip)
        self.recent = deque(maxlen=3)
        self.last_motion = None
        self.arm_deadline = math.inf   # 開機時還沒看過臉：允許判空床
        self.match_since = None
        self.empty = False

    def update(self, t, hidden, motion, ref_diff):
        self.recent.append(motion >= MOTION_TH)
        if sum(self.recent) >= 2:
            self.last_motion = t
        if self.last_motion is not None and t - self.last_motion < CARE_HOLD:
            self.empty, self.match_since = False, None
            self.arm_deadline = t + ARM
            if hidden:  # 暫停：照護中看不到臉不算時間；已經在響的不解除
                self.alarm.since_hidden = self.alarm.since_visible = None
            else:
                self.alarm.update(t, False)
            return "alert" if self.alarm.alert else "care"

        matches = ref_diff is not None and ref_diff < EMPTY_TH
        if self.empty:
            if hidden and matches:
                return "empty"
            self.empty = False  # 看到臉、或床上多了東西 → 回到一般監看
        if hidden and matches and (t <= self.arm_deadline or self.match_since is not None):
            self.match_since = t if self.match_since is None else self.match_since
            if t - self.match_since >= CONFIRM - 1e-6:
                self.empty, self.match_since = True, None
                self.alarm = AlertState(self.alarm.on, self.alarm.off, self.alarm.blip)
                return "empty"
        else:
            self.match_since = None
        alert = self.alarm.update(t, hidden)
        if not hidden:
            self.arm_deadline = -math.inf  # 寶寶在床上：要等下一次有大人來，才可能變空床
        return "alert" if alert else "ok"


def small(frame, rect):
    h, w = frame.shape[:2]
    crop = frame[int(rect[1] * h):int(rect[3] * h), int(rect[0] * w):int(rect[2] * w)]
    s = SMALL / max(crop.shape[:2])
    crop = cv2.resize(crop, (round(crop.shape[1] * s), round(crop.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(crop, (5, 5), 0)


def edge_motion(prev, cur, band):
    """床位邊緣一圈「在動」的像素比例。先用 phase correlation 估整張小圖的平移（鏡頭晃動）、把前一幀對齊再相減；
    對齊後邊界外推出來的那幾排像素不算。"""
    (dx, dy), _ = cv2.phaseCorrelate(prev.astype(np.float32), cur.astype(np.float32))
    h, w = cur.shape
    aligned = cv2.warpAffine(prev, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), borderMode=cv2.BORDER_REPLICATE)
    mx, my = int(math.ceil(abs(dx))) + 1, int(math.ceil(abs(dy))) + 1
    valid = np.zeros_like(band)
    valid[my:h - my, mx:w - mx] = True
    return float((cv2.absdiff(cur, aligned) > PIXEL_TH)[band & valid].mean())


def border_mask(shape):
    h, w = shape
    bh, bw = max(1, int(h * BAND)), max(1, int(w * BAND))
    band = np.ones((h, w), bool)
    band[bh:h - bh, bw:w - bw] = False
    return band


class EdgeFeatures:
    """逐幀算床位特徵（live_view.py 即時用；features() 批次也用這個，兩邊算法一樣）。
    每次 update 給一幀（要是每秒 FPS 幀取樣的那幾幀），回傳 (邊緣移動比例, 跟空床參考的色差或 None)。"""

    def __init__(self, rect, ref_frame=None):
        self.rect = rect
        self.ref = small(ref_frame, rect).astype(np.float32) if ref_frame is not None else None
        self.prev, self.band = None, None

    def update(self, frame):
        img = small(frame, self.rect)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.band = border_mask(gray.shape) if self.band is None else self.band
        motion = edge_motion(self.prev, gray, self.band) if self.prev is not None else 0.0
        diff = float(np.abs(img.astype(np.float32) - self.ref).mean()) if self.ref is not None else None
        self.prev = gray
        return motion, diff


def read_empty_ref(cap, slot):
    """讀床位的空床參考畫面（安裝時拍的空床照；沒有就回傳 None），讀完把影片倒回開頭。"""
    if slot["empty_ref"] is None:
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, slot["empty_ref"] * 1000)
    ok, frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return frame if ok else None


def features(name, slot):
    """每秒 FPS 幀（跟 eval_video.py 同一批幀）：(秒, 邊緣移動比例, 跟空床參考的色差或 None)。"""
    cap = cv2.VideoCapture(str(video_labels.find_video(name)))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    feat = EdgeFeatures(slot["rect"], read_empty_ref(cap, slot))
    step = max(1, round(src_fps / FPS))
    rows, i = [], 0
    while cap.grab():
        if i % step == 0:
            _, frame = cap.retrieve()
            rows.append((round(i / src_fps, 3), *feat.update(frame)))
        i += 1
    cap.release()
    return rows


def run(name, ann):
    doll, slot = next(iter(ann["slots"].items()))
    frames = {float(r["t"]): r for r in read_csv(EVAL_DIR / f"{name}_frames.csv") if r["doll"] == doll}
    base, ctx, rows = AlertState(3, 2, 1), CribState(), []
    for t, motion, diff in features(name, slot):
        r = frames.get(t)
        if r is None:
            continue
        hidden = r["v3_slot"] == "hidden"
        seg = next((s for s in ann["segments"] if s["start"] <= t <= s["end"]), None)
        rows.append({"t": t, "gt": "" if seg is None else seg["label"], "staged": "" if seg is None else seg["staged"],
                     "hidden": int(hidden), "motion": round(motion, 4), "ref_diff": "" if diff is None else round(diff, 1),
                     "base": "alert" if base.update(t, hidden) else "ok", "context": ctx.update(t, hidden, motion, diff)})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{name}_context.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def score(name, ann, rows):
    """逐段比較：原本的 3 秒規則（base）vs 加上床位情境（context）。"""
    def at(seg):
        return [r for r in rows if seg["start"] <= r["t"] <= seg["end"]]

    out = {"danger": [], "safe": [], "prone": [], "empty": [], "gap_care": None}
    for seg in ann["segments"]:
        rs = at(seg)
        if not rs:
            continue
        late = [r for r in rs if r["t"] >= seg["start"] + 2]  # 扣掉開頭 2 秒（上一段的狀態還在轉換）
        res = {"seg": seg}
        for m in ("base", "context"):
            if seg["label"] == "alert":
                first = next((r["t"] for r in rs if r[m] == "alert"), None)
                res[m] = {"caught": rs[-1][m] == "alert", "latency": None if first is None else first - seg["start"],
                          "empty_frames": sum(r[m] == "empty" for r in rs)}
            else:  # 空床另外記：結束時是不是灰燈、進空床前是不是已經在響（例如真危險時大人來抱走）
                res[m] = {"red": any(r[m] == "alert" for r in late),
                          "gray": sum(r[m] == "empty" for r in late) / max(len(late), 1),
                          "end_gray": rs[-1][m] == "empty", "red_before": rs[0][m] == "alert",
                          "red_secs": sum(r[m] == "alert" for r in rs) / FPS}
        kind = {"alert": "danger", "empty": "empty"}.get(seg["label"], "prone" if seg["staged"] == "prone" else "safe")
        out[kind].append(res)
    gap = [r for r in rows if r["gt"] == ""]
    out["gap_care"] = sum(r["context"] == "care" for r in gap) / max(len(gap), 1)
    clean = [r for r in rows if r["gt"] != ""]
    out["clean_care"] = sum(r["context"] == "care" for r in clean) / max(len(clean), 1)
    return out


def report(name, out):
    print(f"\n== {name}")
    for kind, title in (("danger", "危險"), ("safe", "安全"), ("prone", "趴睡但口鼻看得到"), ("empty", "空床")):
        for res in out[kind]:
            s = res["seg"]
            cells = []
            for m in ("base", "context"):
                x = res[m]
                if kind == "danger":
                    lat = "—" if x["latency"] is None else f"{x['latency']:.1f}s 響"
                    cells.append(f"{'抓到' if x['caught'] else '漏掉'}（{lat}"
                                 + (f"，{x['empty_frames']} 幀判成空床！" if x["empty_frames"] else "") + "）")
                elif kind == "empty":
                    cells.append(f"紅燈 {x['red_secs']:.0f} 秒" + ("（之前就在響）" if x["red_before"] else "")
                                 + f"、結束時{'灰燈' if x['end_gray'] else '仍是紅燈'}")
                else:
                    cells.append("紅燈" if x["red"] else "沒響")
            print(f"  {s['start']:5.0f}-{s['end']:3.0f}s {title:<8} 原本：{cells[0]:<16} 加情境：{cells[1]:<22} {s['note'][:20]}")
    print(f"  手入鏡的過渡期有 {out['gap_care']:.0%} 幀判成照護中；乾淨片段誤判成照護中 {out['clean_care']:.0%}")


def summarize(outs):
    print(f"\n{'=' * 78}\n== 合計（{len(outs)} 支）")
    for m, title in (("base", "原本的 3 秒規則"), ("context", "加上照護中＋空床")):
        danger = [r for o in outs for r in o["danger"]]
        caught = sum(r[m]["caught"] for r in danger)
        lats = [r[m]["latency"] for r in danger if r[m]["latency"] is not None]
        bad = sum(r[m]["empty_frames"] > 0 for r in danger)
        safe = [r for o in outs for r in o["safe"]]
        prone = [r for o in outs for r in o["prone"]]
        empty = [r for o in outs for r in o["empty"]]
        reds = "、".join(f"{r[m]['red_secs']:.0f}" for r in empty) or "—"
        print(f"  {title:<12} 危險抓到 {caught}/{len(danger)}（開始響的中位數 {statistics.median(lats):.1f} 秒；"
              f"危險片段被判成空床 {bad} 段）｜安全誤報 {sum(r[m]['red'] for r in safe)}/{len(safe)}"
              f"｜趴睡但口鼻看得到 {sum(r[m]['red'] for r in prone)}/{len(prone)} 響"
              f"｜空床結束時灰燈 {sum(r[m]['end_gray'] for r in empty)}/{len(empty)}（紅燈秒數 {reds}）")


def draw(name, rows, ann, out_path):
    """時間軸：標註、原本的狀態、加情境的狀態，下面畫邊緣移動量和空床色差。"""
    W, left, row_h = 1800, 120, 26
    dur = rows[-1]["t"] + 1
    H = 60 + 3 * row_h + 2 * 70 + 50
    img = np.full((H, W, 3), 255, np.uint8)
    xs = lambda t: int(left + (W - left - 20) * t / dur)
    for t in range(0, int(dur) + 1, 10):
        cv2.line(img, (xs(t), 40), (xs(t), H - 30), (230, 230, 230), 1)
        cv2.putText(img, f"{t // 60}:{t % 60:02d}", (xs(t) - 16, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    cv2.putText(img, f"{name}  green ok / red alert / blue caregiving (timer paused) / gray empty crib",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    y = 45
    gt_colors = {"ok": COLORS["ok"], "alert": COLORS["alert"], "empty": COLORS["empty"]}
    cv2.putText(img, "GT", (10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    for s in ann["segments"]:
        cv2.rectangle(img, (xs(s["start"]), y + 3), (xs(s["end"]), y + row_h - 3), gt_colors[s["label"]], -1)
    for key, label in (("base", "3-s rule"), ("context", "+ context")):
        y += row_h
        cv2.putText(img, label, (10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        for r in rows:
            cv2.rectangle(img, (xs(r["t"]), y + 3), (max(xs(r["t"]) + 1, xs(r["t"] + 0.2)), y + row_h - 3),
                          COLORS[r[key]], -1)
    for key, label, scale, th in (("motion", "edge motion", 0.2, MOTION_TH), ("ref_diff", "diff to empty", 60, EMPTY_TH)):
        y += row_h + 10 if key == "motion" else 70
        cv2.putText(img, label, (10, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.line(img, (left, y + 60 - int(60 * th / scale)), (W - 20, y + 60 - int(60 * th / scale)), (0, 0, 255), 1)
        for r in rows:
            if r[key] == "":
                continue
            v = min(float(r[key]) / scale, 1.0)
            cv2.line(img, (xs(r["t"]), y + 60), (xs(r["t"]), y + 60 - int(60 * v)), (120, 120, 120), 1)
    cv2.imencode(".png", img)[1].tofile(str(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", help="影片名稱；不給就跑床位檔裡有空床參考（empty_ref）的影片")
    args = ap.parse_args()
    anns = video_labels.load()
    names = args.videos or sorted(v for v, a in anns.items()
                                  if any(s["empty_ref"] is not None for s in a["slots"].values()))
    outs = []
    for name in names:
        ann = anns[name]
        if len(ann["slots"]) != 1:
            print(f"[略過] {name} 一個畫面有 {len(ann['slots'])} 個床位；這支只處理每床鏡頭（一個畫面一張床）")
            continue
        rows = run(name, ann)
        out = score(name, ann, rows)
        report(name, out)
        draw(name, rows, ann, OUT_DIR / f"{name}_context.png")
        outs.append(out)
    if outs:
        summarize(outs)
    print(f"\n參數：邊緣移動門檻 {MOTION_TH}、動作停 {CARE_HOLD} 秒算離開、離開後 {ARM} 秒內可判空床、"
          f"空床色差門檻 {EMPTY_TH}、相符 {CONFIRM} 秒才判空床\n輸出：{OUT_DIR}")


if __name__ == "__main__":
    main()
