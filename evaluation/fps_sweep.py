"""
fps_sweep.py
每床每秒要判讀幾幀才夠：把 eval_video 存的每秒 5 幀每幀判定（<影片>_frames.csv）每隔 k 幀取一幀，
重算 v6＋床位的警示結果（每秒 5/k 幀）。主流程每一幀的判定各自獨立（pipeline/crib_detect.py 沒有跨幀狀態），
所以抽幀的結果就等於用每秒 5/k 幀跑；不用重跑模型、不用晶片。
每個 k 都把 k 種起點（第 0、1、…、k-1 幀開始取）全部算過，列出最好到最差。
警示規則跟 eval_video 一樣用時間算（連續 3 秒看不到才警示、2 秒看得到才解除、看得到不到 1 秒不算數），
幀率低時湊滿 3 秒需要的幀數就少（每秒 1 幀：連續 4 幀看不到）。

只算 v6 沒訓練過的影片（hold＋場域；拿掉 hailo/prep_calib.py SPLITS 裡 calib 那一半）。
情況依標註的 note 分類（關鍵字），分類規則在 category()。
空床這裡一定會響：空床判斷是 crib_context 另外做的（用前後幀的變化），沒有在這裡測。

用法：
    uv run python -m evaluation.fps_sweep                       # CPU 版（outputs/video_eval/）
    uv run python -m evaluation.fps_sweep --eval-dir outputs/video_eval_chip-rtmo_D1_chip-mouth_F   # 晶片 F 版
輸出：印出每種幀率、每種情況的結果；--json 另存明細。
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from evaluation import eval_video, video_labels
from evaluation.video_labels import read_csv
from hailo.prep_calib import SPLITS

ROOT = Path(__file__).resolve().parent.parent
MODE = "v6_slot"
STEPS = (1, 2, 3, 4, 5, 10, 15, 25)   # 每隔幾幀取一幀：每秒 5、2.5、1.67、1.25、1、0.5、0.33、0.2 幀
# 嬰兒翻身：非娃娃影片。pp_main_215118：11/26 起拿來測照護中、空床（crib_context）的場域錄影，
# 不在 docs/影片測試結果.md §3m 那 30 支裡；加進來合計就會跟文件對不上
NOT_SCORED = {"嬰兒翻身", "pp_main_215118"}


def category(seg):
    note, label, staged = seg["note"], seg["label"], seg["staged"]
    if label == "empty":
        return "空床（這裡一定會響，見檔頭）"
    if label == "ok" and staged == "prone":
        return "趴睡但口鼻看得到（另列）"
    if label == "alert" and staged == "unseen":
        return "黃燈：口鼻沒蓋但看不到（該響）"
    if label == "alert":
        return "危險：趴睡臉朝下（該響）" if "趴" in note else "危險：布料蓋住口鼻（該響）"
    if any(w in note for w in ("毛巾", "包巾", "棉被", "被", "外套", "玩偶")):
        return "安全：有布料或玩偶、口鼻露出（誤報）"
    if "側" in note:
        return "安全：側躺口鼻看得到（誤報）"
    return "安全：仰躺口鼻看得到（誤報）"


def load(eval_dir):
    ann = video_labels.load()
    data = {}
    for p in sorted(eval_dir.glob("*_frames.csv")):
        name = p.name[: -len("_frames.csv")]
        if name not in ann or name in SPLITS["calib"]["videos"] or name in NOT_SCORED:
            continue
        rows = read_csv(p)
        if not rows or MODE not in rows[0]:
            continue
        for r in rows:
            r["t"] = float(r["t"])
        idx = {t: i for i, t in enumerate(sorted({r["t"] for r in rows}))}
        for r in rows:
            r["i"] = idx[r["t"]]
        data[name] = rows
    return ann, data


def score(ann, data, k, phase, on, off, blip):
    """回傳 {情況: [響了幾段, 幾段]}、每段危險第一次響的時間、漏掉的危險片段。"""
    per, first, misses = defaultdict(lambda: [0, 0]), {}, []
    for name, rows in data.items():
        rows = [r for r in rows if r["i"] % k == phase]
        results, alerts = eval_video.evaluate(rows, ann[name]["segments"], on, off, blip)
        for res in results:
            s, x, c = res["seg"], res[MODE], category(res["seg"])
            if s["label"] == "alert":
                per[c][0] += bool(x["caught"])
                per[c][1] += 1
                if s["staged"] == "unseen":
                    continue
                key = f'{name}|{s["doll"]}|{s["start"]}'
                if x["caught"]:
                    times, alert, _ = alerts[s["doll"], MODE]
                    first[key] = next(t for t, a in zip(times, alert) if s["start"] <= t <= s["end"] and a)
                else:
                    misses.append(f'{name} {s["doll"]} {s["start"]:g}–{s["end"]:g} 秒')
            elif x["false_alert"] is not None:   # 片段太短、扣掉開頭 off 秒後沒有幀就不計
                per[c][0] += bool(x["false_alert"])
                per[c][1] += 1
    return dict(per), first, misses


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", default="outputs/video_eval", help="eval_video 的輸出資料夾（每秒 5 幀的 *_frames.csv）")
    ap.add_argument("--on", type=float, default=3)
    ap.add_argument("--off", type=float, default=2)
    ap.add_argument("--blip", type=float, default=1)
    ap.add_argument("--json", help="另存明細")
    args = ap.parse_args()
    eval_video.MODES = [MODE]   # evaluate() 只算 v6＋床位
    ann, data = load(ROOT / args.eval_dir)
    print(f"{args.eval_dir}：v6 沒訓練過的 {len(data)} 支｜警示規則 {args.on:g}／{args.off:g}／{args.blip:g} 秒")
    out, base = [], None
    for k in STEPS:
        phases = [score(ann, data, k, p, args.on, args.off, args.blip) for p in range(k)]
        if base is None:
            base = phases[0][1]
        cats = sorted({c for per, _, _ in phases for c in per})
        print(f"\n每床每秒 {5 / k:.2f} 幀（每 {k} 幀取 1 幀，{k} 種起點）")
        for c in cats:
            fr = sorted({f"{per.get(c, [0, 0])[0]}/{per.get(c, [0, 0])[1]}" for per, _, _ in phases},
                        key=lambda s: (int(s.split("/")[1]), int(s.split("/")[0])))
            print(f"  {c:<28}{'、'.join(fr)}")
        late = [f[key] - base[key] for _, f, _ in phases for key in f if key in base]
        print(f"  同一段危險，比每秒 5 幀晚響：中位 {statistics.median(late):.1f} 秒、最多 {max(late):.1f} 秒"
              f"（{len(late)} 段 × 起點）")
        miss = sorted({m for _, _, ms in phases for m in ms})
        print(f"  漏掉的危險片段（任一種起點）：{'；'.join(miss) or '無'}")
        out.append({"step": k, "fps": 5 / k, "phases": [{"per": per, "first_alert": f, "misses": ms}
                                                        for per, f, ms in phases]})
    if args.json:
        Path(args.json).write_text(json.dumps({"eval_dir": args.eval_dir, "videos": sorted(data), "rates": out},
                                              ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
