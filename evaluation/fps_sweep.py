"""
fps_sweep.py
每床每秒要判讀幾幀才夠：把 eval_video 存的每秒 5 幀每幀判定（<影片>_frames.csv）每隔 k 幀取一幀，
重算主流程（v7＋床位；--mode 可換 v6_slot）的警示結果（每秒 5/k 幀）。主流程每一幀的判定各自獨立（pipeline/crib_detect.py 沒有跨幀狀態），
所以抽幀的結果就等於用每秒 5/k 幀跑；不用重跑模型、不用晶片。
每個 k 都把 k 種起點（第 0、1、…、k-1 幀開始取）全部算過，列出最好到最差。
警示規則跟 eval_video 一樣用時間算（連續 3 秒看不到才警示、2 秒看得到才解除、看得到不到 1 秒不算數），
幀率低時湊滿 3 秒需要的幀數就少（每秒 1 幀：連續 4 幀看不到）。

只算 v6 沒訓練過的影片（hold＋場域；拿掉 hailo/prep_calib.py SPLITS 裡 calib 那一半）。
情況依標註的 note 分類（關鍵字），分類規則在 category()。
空床這裡一定會響：空床判斷是 crib_context 另外做的（用前後幀的變化），沒有在這裡測。

小數幀率（2026/11/28 加，--rates）：多床時每床實際分到的幀率不是 5 的因數（例如 25 床合計 24 幀/秒＝每床 0.96 幀），
用時間取樣：每 1/幀率 秒取最接近的那一幀，起點平均取 5 種。--gap G：再加一種最壞情況——每段危險一開始就斷線 G 秒
（那床 G 秒沒有判讀，長時間測試量到的最長空檔），看還抓不抓得到、晚多少。這裡的警示規則不會把斷線那段補算成看不到
（live_view 的 crib_state.CribState.resume 會），所以晚響是上限。

用法：
    uv run python -m evaluation.fps_sweep                       # CPU 版（outputs/video_eval/）
    uv run python -m evaluation.fps_sweep --rates 0.96 0.8 --gap 3.6   # 每床每秒 0.96、0.8 幀，加斷線 3.6 秒
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
MODE = "v7_slot"   # --mode 可換；11/25 的 docs/影片測試結果.md §3m 是 v6_slot
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


def pick_step(k, phase):
    """每 k 幀取一幀（第 phase 幀開始）。"""
    return lambda name, rows: [r for r in rows if r["i"] % k == phase]


def pick_rate(rate, phase, gap=0.0, ann=None):
    """每床每秒 rate 幀（小數也可以）：從第 phase 秒起，每 1/rate 秒取最接近的那一幀（每秒 5 幀的格子）。
    gap > 0：每段危險開始後 gap 秒內那床的幀全部拿掉（斷線的最壞情況）。"""
    def pick(name, rows):
        n = max(r["i"] for r in rows) + 1
        want, j = set(), 0
        while (x := round((phase + j / rate) * 5)) < n:
            want.add(x)
            j += 1
        out = [r for r in rows if r["i"] in want]
        if gap > 0:
            cut = [(s["doll"], s["start"], s["start"] + gap) for s in ann[name]["segments"]
                   if s["label"] == "alert" and s["staged"] != "unseen"]
            out = [r for r in out if not any(r["doll"] == d and a <= r["t"] < b for d, a, b in cut)]
        return out
    return pick


def score(ann, data, pick, on, off, blip):
    """pick(影片名稱, 每幀判定) → 抽出來的幀。回傳 {情況: [響了幾段, 幾段]}、每段危險第一次響的時間、漏掉的危險片段。"""
    per, first, misses = defaultdict(lambda: [0, 0]), {}, []
    for name, rows in data.items():
        rows = pick(name, rows)
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
    global MODE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", default="outputs/video_eval", help="eval_video 的輸出資料夾（每秒 5 幀的 *_frames.csv）")
    ap.add_argument("--on", type=float, default=3)
    ap.add_argument("--off", type=float, default=2)
    ap.add_argument("--blip", type=float, default=1)
    ap.add_argument("--json", help="另存明細")
    ap.add_argument("--mode", default=MODE, choices=["v7_slot", "v6_slot"], help=f"依哪一種判定（預設主流程 {MODE}）")
    ap.add_argument("--rates", type=float, nargs="*", default=[], help="另外算這些每床每秒幀數（小數也可以，5 種起點）")
    ap.add_argument("--gap", type=float, default=0, help="--rates 再各加一次：每段危險一開始就斷線這麼多秒")
    ap.add_argument("--phases", type=int, default=5, help="--rates 每種幀率取幾種起點（預設 5）")
    args = ap.parse_args()
    MODE = args.mode
    eval_video.MODES = [MODE]   # evaluate() 只算這一種
    ann, data = load(ROOT / args.eval_dir)
    print(f"{args.eval_dir}：v6 沒訓練過的 {len(data)} 支｜警示規則 {args.on:g}／{args.off:g}／{args.blip:g} 秒")
    out, base = [], None
    runs = [(k, f"每 {k} 幀取 1 幀，{k} 種起點", 5 / k, [pick_step(k, p) for p in range(k)], 0.0) for k in STEPS]
    for rate in args.rates:
        for gap in ([0.0, args.gap] if args.gap else [0.0]):
            runs.append((None, f"時間取樣，{args.phases} 種起點" + (f"，每段危險一開始斷線 {gap:g} 秒" if gap else ""), rate,
                         [pick_rate(rate, p / (args.phases * rate), gap, ann) for p in range(args.phases)], gap))
    for k, how, fps, picks, gap in runs:
        phases = [score(ann, data, pk, args.on, args.off, args.blip) for pk in picks]
        if base is None:
            base = phases[0][1]
        cats = sorted({c for per, _, _ in phases for c in per})
        print(f"\n每床每秒 {fps:.2f} 幀（{how}）")
        for c in cats:
            fr = sorted({f"{per.get(c, [0, 0])[0]}/{per.get(c, [0, 0])[1]}" for per, _, _ in phases},
                        key=lambda s: (int(s.split("/")[1]), int(s.split("/")[0])))
            print(f"  {c:<28}{'、'.join(fr)}")
        late = [f[key] - base[key] for _, f, _ in phases for key in f if key in base]
        print(f"  同一段危險，比每秒 5 幀晚響：中位 {statistics.median(late):.1f} 秒、最多 {max(late):.1f} 秒"
              f"（{len(late)} 段 × 起點）")
        miss = sorted({m for _, _, ms in phases for m in ms})
        print(f"  漏掉的危險片段（任一種起點）：{'；'.join(miss) or '無'}")
        out.append({"step": k, "fps": fps, "gap": gap, "phases": [{"per": per, "first_alert": f, "misses": ms}
                                                                  for per, f, ms in phases]})
    if args.json:
        Path(args.json).write_text(json.dumps({"eval_dir": args.eval_dir, "videos": sorted(data), "rates": out},
                                              ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
