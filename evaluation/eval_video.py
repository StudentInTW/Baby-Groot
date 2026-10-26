"""
eval_video.py
用娃娃影片評估「每個床位」的判定，跟 Harry 的時間軸比對。照片只能看單張判對沒，影片才量得到：
  - 危險片段有沒有發出警示（危險持續到片段結束時，系統是不是在響）
  - 安全片段會不會亂叫（誤報）

標註（labels/，讀取見 video_labels.py；不能公開的素材另放不進 repo 的 local_video_*.csv）：
  doll_video_slots.csv   每支影片、每隻娃娃一個床位方框（畫面比例座標），和床位方向 rot
                         （把床位轉正要轉幾度；安裝時設定頭朝哪邊）。頭在框裡的人就算這個床位。
  doll_video_labels.csv  每段狀態的開始、結束秒數。只列手不在畫面裡的乾淨片段；
                         手在擺娃娃的過渡期不計分（label 意義同 doll_labels.csv：ok 口鼻看得到／alert 看不到；
                         empty 空床——寶寶被抱走，系統不該響）。staged＝prone 是「趴睡但口鼻看得到」：
                         政策上要警示、但口鼻判定本來就抓不到，另外列，不算進誤報。

比較四種判定（v3、v3＋床位、v4＋床位、v5＋床位；每幀的判定在 pipeline/crib_detect.py）。
警示規則（pipeline/crib_state.py 的 AlertState）：每個床位各自一個狀態機，連續 --on 秒看不到 → 警示；連續 --off 秒看得到 → 解除；
看得到要持續 --blip 秒以上才會把「看不到」的計時歸零（防閃爍）。

用法：
    uv run python -m evaluation.eval_video                  # 跑 doll_video_labels.csv 裡所有娃娃影片（不含 NOT_DOLL 和 local 標註的影片）
    uv run python -m evaluation.eval_video IMG_8150         # 只跑一支
    uv run python -m evaluation.eval_video 嬰兒翻身          # 非娃娃的影片（NOT_DOLL、local 標註）要指定名稱才跑
    uv run python -m evaluation.eval_video --reuse          # 不重跑偵測，用上次存的每幀結果重算（改 --on/--off 時用）
    uv run python -m evaluation.eval_video --gray --dim 0.3 # 模擬補光不夠的 IR 夜視（見 degrade.py）
輸出：outputs/video_eval/<影片>_frames.csv（每幀判定）、<影片>_timeline.png（時間軸對照圖）、
      <影片>_preview.mp4（標註影片，縮小版）；--gray/--dim 的結果放 outputs/video_eval_<條件>/
"""

import argparse
import csv
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 要在 import cv2 前設

import cv2
import numpy as np

from evaluation import video_labels
from evaluation.degrade import degrade, describe, tag
from evaluation.video_labels import read_csv
from pipeline import infant_face_v3, infant_face_v4, infant_face_v5
from pipeline.crib_detect import MODES, detect
from pipeline.crib_state import AlertState
from pipeline.face_common import FACE_MODEL, FACE_SCORE_TH, GREEN, RED, YELLOW, put_label
from pipeline.infant_face_v3 import load_rtmo

ROOT = Path(__file__).resolve().parent.parent
# 不是娃娃的影片要指定名稱才跑，不混進娃娃影片的合計（數字才跟之前的對得上）；local_ 標註檔裡的影片也一樣
NOT_DOLL = {"嬰兒翻身"}  # 嘉晏哥哥提供的 AI 生成影片：灰階夜視俯拍，一個嬰兒仰→側→趴（dataset/嬰兒翻身.mp4）
OUT_DIR = ROOT / "outputs" / "video_eval"
MODE_NAMES = {"v3": "v3", "v3_slot": "v3+slot", "v4_slot": "v4+slot", "v5_slot": "v5+slot"}
PREVIEW_W = 540     # 標註影片的寬度


def load_annotations():
    """回傳 {影片: {"slots": {娃娃: (x0, y0, x1, y1)}, "rot": {娃娃: 度數}, "segments": [段落, ...], "local": bool}}。"""
    return {name: {"slots": {d: s["rect"] for d, s in v["slots"].items()},
                   "rot": {d: s["rot"] for d, s in v["slots"].items()},
                   "segments": v["segments"], "local": v["local"]}
            for name, v in video_labels.load().items()}


def gt_at(segments, doll, t):
    """這一刻這隻娃娃的標註；落在過渡期（不計分）就回傳 None。"""
    return next((s for s in segments if s["doll"] == doll and s["start"] <= t <= s["end"]), None)


def alert_track(times, hidden, on, off, blip):
    """回傳每一幀是否在警示中。"""
    state = AlertState(on, off, blip)
    return [state.update(t, hid) for t, hid in zip(times, hidden)]


def evaluate(rows, segments, on, off, blip):
    """rows：每幀每個床位一列。回傳每個段落、每種判定的結果。"""
    by_doll = defaultdict(list)
    for r in rows:
        by_doll[r["doll"]].append(r)
    alerts = {}
    for doll, rs in by_doll.items():
        rs.sort(key=lambda r: r["t"])
        times = [r["t"] for r in rs]
        for m in MODES:
            alerts[doll, m] = (times, alert_track(times, [r[m] == "hidden" for r in rs], on, off, blip),
                               [r[m] == "hidden" for r in rs])
    results = []
    for s in segments:
        times = alerts[s["doll"], MODES[0]][0]
        if not any(s["start"] <= t <= s["end"] for t in times):
            continue  # 這段沒有跑到（--limit-sec 試跑）
        res = {"seg": s}
        for m in MODES:
            times, alert, hidden = alerts[s["doll"], m]
            idx = [i for i, t in enumerate(times) if s["start"] <= t <= s["end"]]
            frac_hidden = sum(hidden[i] for i in idx) / max(len(idx), 1)
            if s["label"] == "alert":  # 危險一直持續到這段結束，那一刻還在警示才算抓到
                # （過渡期手擋住畫面時可能已經響了；只看開頭有響，會把「手擋住」誤算成抓到）
                res[m] = {"hidden": frac_hidden, "caught": alert[idx[-1]],
                          "coverage": sum(alert[i] for i in idx) / len(idx)}
            else:  # 口鼻看得到：扣掉開頭 off 秒（前一段的警示還在解除中），之後還在警示就算誤報
                idx_late = [i for i in idx if times[i] >= s["start"] + off]
                res[m] = {"hidden": frac_hidden, "false_alert": any(alert[i] for i in idx_late) if idx_late else None}
        results.append(res)
    return results, alerts


def report(name, results, on, off, blip):
    def pct(a, b):
        return f"{a}/{b}（{a / b:.0%}）" if b else "—"

    print(f"\n{'=' * 78}\n== {name}｜警示規則：連續 {on:g} 秒看不到才警示、連續 {off:g} 秒看得到才解除、看得到不到 {blip:g} 秒不算數")
    print(f"\n{'時間':<11}{'娃娃':<3}{'標註':<7}" + "".join(f"{MODE_NAMES[m]:>18}" for m in MODES) + "  說明")
    for r in results:
        s = r["seg"]
        cells = []
        for m in MODES:
            x = r[m]
            if s["label"] == "alert":
                txt = "抓到" if x["caught"] else "漏掉"
            else:
                txt = "—" if x["false_alert"] is None else ("誤報" if x["false_alert"] else "正常")
            cells.append(f"{txt}（{x['hidden']:.0%}）")
        span = f"{int(s['start'] // 60)}:{s['start'] % 60:04.1f}-{int(s['end'] // 60)}:{s['end'] % 60:04.1f}"
        print(f"{span:<11}{s['doll']:<3}{s['label']:<7}" + "".join(f"{c:>16}" for c in cells) + f"  {s['note']}")
    print("    （括號＝這段裡每幀判成看不到口鼻的比例；危險片段要到結束那一刻還在警示才算抓到）")
    summarize(results)


def summarize(results, title=None):
    if title:
        print(f"\n{'=' * 78}\n== {title}")
    danger = [r for r in results if r["seg"]["label"] == "alert"]
    safe = [r for r in results if r["seg"]["label"] == "ok" and r["seg"]["staged"] != "prone"]
    prone = [r for r in results if r["seg"]["label"] == "ok" and r["seg"]["staged"] == "prone"]
    empty = [r for r in results if r["seg"]["label"] == "empty"]
    for m in MODES:
        caught = [r for r in danger if r[m]["caught"]]
        cover = sum(r[m]["coverage"] for r in danger) / max(len(danger), 1)
        scored = [r for r in safe if r[m]["false_alert"] is not None]
        fa = sum(r[m]["false_alert"] for r in scored)
        extra = "".join(f"｜{name} {sum(bool(r[m]['false_alert']) for r in segs)}/{len(segs)} 有響"
                        for name, segs in (("趴睡但口鼻看得到", prone), ("空床", empty)) if segs)
        print(f"    {MODE_NAMES[m]:<8} 危險片段抓到 {len(caught)}/{len(danger)}（{len(caught) / max(len(danger), 1):.0%}，"
              f"危險時間裡有 {cover:.0%} 在警示）｜安全片段誤報 {fa}/{len(scored)}{extra}")


def draw_timeline(name, rows, alerts, segments, duration, out_path):
    """時間軸對照圖：每隻娃娃一組——標註、三種判定（細條＝每幀判定，粗條＝警示中）。"""
    W, left, row_h, gap = 1800, 110, 26, 14
    dolls = sorted({r["doll"] for r in rows})
    n_rows = 1 + len(MODES)
    H = 60 + len(dolls) * (n_rows * row_h + gap + 24) + 40
    img = np.full((H, W, 3), 255, np.uint8)
    xs = lambda t: int(left + (W - left - 20) * t / duration)
    for t in range(0, int(duration) + 1, 10):  # 時間軸
        x = xs(t)
        cv2.line(img, (x, 40), (x, H - 30), (225, 225, 225), 1)
        cv2.putText(img, f"{t // 60}:{t % 60:02d}", (x - 16, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    cv2.putText(img, f"{name}  (green = mouth/nose visible, red = not visible / alert, gray = no person, "
                     f"blank GT = hands in frame, not scored)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    y = 50
    colors = {"ok": (80, 180, 80), "alert": (60, 60, 220), "partial": (0, 170, 255), "empty": (170, 170, 170)}
    for doll in dolls:
        cv2.putText(img, f"{doll} doll", (10, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        y += 24
        cv2.putText(img, "GT", (10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        for s in segments:
            if s["doll"] == doll:
                cv2.rectangle(img, (xs(s["start"]), y + 3), (xs(s["end"]), y + row_h - 3), colors[s["label"]], -1)
        y += row_h
        rs = sorted((r for r in rows if r["doll"] == doll), key=lambda r: r["t"])
        for m in MODES:
            cv2.putText(img, MODE_NAMES[m], (10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            times, alert, _ = alerts[doll, m]
            for i, r in enumerate(rs):  # 上半：每幀判定
                x0, x1 = xs(r["t"]), max(xs(r["t"]) + 1, xs(r["t"] + 0.2))
                c = {"visible": (80, 180, 80), "hidden": (60, 60, 220), "none": (170, 170, 170)}[r[m]]
                cv2.rectangle(img, (x0, y + 3), (x1, y + row_h // 2), c, -1)
                if alert[i]:  # 下半：警示中
                    cv2.rectangle(img, (x0, y + row_h // 2 + 1), (x1, y + row_h - 3), (30, 30, 170), -1)
            y += row_h
        y += gap
    # 影片名稱可能是中文：cv2.imwrite 在 Windows 寫不了非 ASCII 路徑（回報成功、其實沒存），改用 imencode
    cv2.imencode(".png", img)[1].tofile(str(out_path))


def draw_preview(frame, t, slots, persons, faces, alert_now, segments, mode):
    """標註影片的一幀：床位框（紅＝警示中，依 --preview 選的判定）、人框、臉（黃＝找到的臉，橘＝口鼻檢查判成被蓋住）。"""
    h, w = frame.shape[:2]
    vis = frame.copy()
    for p in persons:
        x1, y1, x2, y2 = p["box"]
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (200, 200, 200), 3)
    for f in faces:
        is_covered = getattr(f, "skin", None) is not None and f.skin < infant_face_v4.MOUTH_SKIN_MIN
        c = infant_face_v4.ORANGE if is_covered else YELLOW
        cv2.rectangle(vis, (int(f[0]), int(f[1])), (int(f[2]), int(f[3])), c, 5)
    for doll, (x0, y0, x1, y1) in slots.items():
        alarm = alert_now[doll]
        color = RED if alarm else GREEN
        cv2.rectangle(vis, (int(x0 * w), int(y0 * h)), (int(x1 * w) - 1, int(y1 * h)), color, 8)
        seg = gt_at(segments, doll, t)
        gt = "GT: -" if seg is None else "GT: " + {"ok": "visible", "alert": "NOT visible"}.get(seg["label"], seg["label"])
        put_label(vis, f"{doll} {'ALERT' if alarm else 'OK'}", x0 * w + 10, y0 * h + 60, color, 1.8, 5)
        put_label(vis, gt, x0 * w + 10, y1 * h - 20, (90, 90, 90), 1.1, 3)
    put_label(vis, f"{int(t // 60)}:{t % 60:04.1f}", 20, 80, (0, 0, 0), 2.2, 5)
    put_label(vis, f"slot color: {MODE_NAMES[mode]} | orange face = mouth/nose covered (v4 check)",
              20, h - 30, (60, 60, 60), 1.0, 2)
    s = PREVIEW_W / w
    return cv2.resize(vis, (PREVIEW_W, round(h * s)), interpolation=cv2.INTER_AREA)


def run_video(name, ann, args, run_pose, detector, det_v5):
    """逐幀偵測（每秒 --fps 幀），存每幀結果和標註影片。回傳 rows、影片長度。"""
    cap = cv2.VideoCapture(str(video_labels.find_video(name)))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(src_fps / args.fps))
    slots, rots, segments = ann["slots"], ann["rot"], ann["segments"]
    writer, rows, i, t0 = None, [], 0, time.perf_counter()
    live = {d: AlertState(args.on, args.off, args.blip) for d in slots}  # 標註影片的床位框顏色（--preview 選的判定）
    while True:
        if not cap.grab() or (args.limit_sec and i / src_fps > args.limit_sec):
            break
        if i % step == 0:
            _, frame = cap.retrieve()
            frame = degrade(frame, args.gray, args.dim, seed=i)  # 雜訊每幀不同（像真的感測器），seed＝幀號可重現
            t = i / src_fps
            states, persons, faces = detect(run_pose, detector, det_v5, frame, slots, rots)
            alert_now = {}
            for doll, st in states.items():
                seg = gt_at(segments, doll, t)
                rows.append({"t": round(t, 3), "doll": doll, "gt": "" if seg is None else seg["label"],
                             "persons": st["persons"], **{m: st[m] for m in MODES},
                             "v5_tier": st["v5_tier"],
                             "face_scores": st["face_scores"], "slot_face_scores": st["slot_face_scores"],
                             "mouth_skin": st["mouth_skin"]})
                alert_now[doll] = live[doll].update(t, st[args.preview] == "hidden")
            pv = draw_preview(frame, t, slots, persons, faces, alert_now, segments, args.preview)
            if writer is None:
                writer = cv2.VideoWriter(str(args.out_dir / f"{name}_preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (pv.shape[1], pv.shape[0]))
            writer.write(pv)
            if len(rows) // len(slots) % 50 == 0:
                print(f"  {name} {t:5.1f}/{n / src_fps:.1f}s（{time.perf_counter() - t0:.0f}s）", flush=True)
        i += 1
    cap.release()
    if writer is not None:
        writer.release()
    with open(args.out_dir / f"{name}_frames.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    return rows, n / src_fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", help="影片名稱（不含副檔名）；不給就跑標註檔裡全部")
    ap.add_argument("--fps", type=float, default=5, help="每秒取幾幀跑偵測（預設 5）")
    ap.add_argument("--on", type=float, default=3, help="連續幾秒看不到口鼻才警示（預設 3；2026/10/9 由 2 改 3，娃娃影片結果相同）")
    ap.add_argument("--off", type=float, default=2, help="連續幾秒看得到口鼻才解除警示（預設 2）")
    ap.add_argument("--blip", type=float, default=1, help="看得到要持續幾秒才把看不到的計時歸零（預設 1；0＝不防閃爍）")
    ap.add_argument("--det-th", type=float, default=0.15, help="RTMO 找人門檻（預設 0.15，見 docs/娃娃測試結果.md §7）")
    ap.add_argument("--reuse", action="store_true", help="不重跑偵測，讀上次存的 <影片>_frames.csv 重算")
    ap.add_argument("--limit-sec", type=float, default=0, help="只跑前幾秒（試跑用；會覆蓋上次的輸出）")
    ap.add_argument("--preview", choices=MODES[1:], default="v3_slot",
                    help="標註影片的床位框顏色依哪一種判定（預設 v3_slot；v4 還沒通過真實嬰兒的誤報測試）")
    ap.add_argument("--gray", action="store_true", help="模擬 IR 夜視：轉成灰階（見 degrade.py）")
    ap.add_argument("--dim", type=float, default=0, help="模擬關燈：亮度乘上這個倍率（0＝不處理）")
    args = ap.parse_args()

    infant_face_v3.DET_TH = args.det_th
    videos = load_annotations()
    names = args.videos or sorted(v for v, a in videos.items() if v not in NOT_DOLL and not a["local"])
    cond = tag(args.gray, args.dim)  # 灰階／關燈的結果另外放，不蓋掉原圖的
    args.out_dir = OUT_DIR.with_name(f"{OUT_DIR.name}_{cond}") if cond else OUT_DIR
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"影像條件：{describe(args.gray, args.dim)}")
    run_pose = detector = det_v5 = None
    all_results = []
    for name in names:
        ann = videos[name]
        if args.reuse:
            rows = read_csv(args.out_dir / f"{name}_frames.csv")
            for r in rows:
                r["t"] = float(r["t"])
            duration = max(r["t"] for r in rows) + 1
        else:
            if run_pose is None:
                run_pose = load_rtmo()
                detector = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), FACE_SCORE_TH)
                det_v5 = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), infant_face_v5.FACE_TH_V5)
            rows, duration = run_video(name, ann, args, run_pose, detector, det_v5)
        results, alerts = evaluate(rows, ann["segments"], args.on, args.off, args.blip)
        report(name, results, args.on, args.off, args.blip)
        draw_timeline(name, rows, alerts, ann["segments"], duration, args.out_dir / f"{name}_timeline.png")
        all_results += results
    if len(names) > 1:
        summarize(all_results, f"合計 {len(names)} 支影片｜{describe(args.gray, args.dim)}")
    print(f"\n輸出：outputs/{args.out_dir.name}/（每幀判定 *_frames.csv、時間軸對照 *_timeline.png、標註影片 *_preview.mp4）")


if __name__ == "__main__":
    main()
