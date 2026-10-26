"""
crib_context.py
每床鏡頭的床位情境評估（2026/10/10，原型）：照護中暫停計時、空床不響。
規則與參數在 pipeline/crib_state.py（CribState）；這支用娃娃影片比較「原本的 3 秒規則」和「加上床位情境」。

輸入：eval_video.py 存的每幀判定（outputs/video_eval/<影片>_frames.csv，用 v3＋床位；要先跑過、每秒 5 幀），
再讀一次影片算兩個便宜的特徵（都在縮成長邊 160 像素的小圖上算，比找臉便宜得多）：
  床位邊緣一圈的移動量（跟前一幀比）、跟空床參考畫面的色差。
空床參考畫面：床位檔的 empty_ref 欄（影片第幾秒是空床）；實機是安裝時拍一張。

用法：
    uv run python -m evaluation.crib_context                    # 有空床參考的影片（補拍的 IMG_8429、IMG_8432）
    uv run python -m evaluation.crib_context IMG_8115           # 其他單床影片（沒有空床參考的話，只有「照護中暫停」）
輸出：outputs/crib_context/<影片>_context.csv（每幀特徵與狀態）、<影片>_context.png（時間軸）、終端機摘要
"""

import argparse
import csv
import os
import statistics
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 要在 import cv2 前設

import cv2
import numpy as np

from evaluation import video_labels
from evaluation.video_labels import read_csv
from pipeline.crib_state import (
    ARM, CARE_HOLD, CONFIRM, EMPTY_TH, MOTION_TH, AlertState, CribState, EdgeFeatures, read_empty_ref,
)

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "outputs" / "video_eval"
OUT_DIR = ROOT / "outputs" / "crib_context"
FPS = 5              # 要跟 eval_video.py 存每幀判定時一樣
COLORS = {"ok": (80, 180, 80), "alert": (60, 60, 220), "care": (200, 140, 40), "empty": (170, 170, 170)}


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
