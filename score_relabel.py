"""
score_relabel.py
用人工重標（docs/任務_SyRIP重標.md 的 labels.csv）重算 SyRIP 的誤報率和「臉看不到→ALERT」比例。

真實標註怎麼來：
  - 重標資料包裡的圖（誤報圖、GT 臉看不到的圖、30 張抽查）：用人工標註 1 看得到／2 部分或不確定／3 看不到
  - 不在資料包裡的圖＝所有評估都判 OK 的「GT 臉看得到」圖，當作 1（30 張抽查用來驗證這個假設）
  - 評估結果裡如果有 ALERT 的圖不在資料包，也暫時當作 1，並印出提醒（代表要補標）

用法：
    uv run python score_relabel.py                # 標註檔預設 outputs/relabel_syrip_labels.csv
    uv run python score_relabel.py 別的標註.csv
    uv run python score_relabel.py --eval-dir outputs/syrip_eval_gray   # 算灰階／關燈那份（eval_syrip.py --gray/--dim）
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from eval_syrip import OUT_DIR as EVAL_DIR

ROOT = Path(__file__).parent
KEY = ROOT / "outputs" / "relabel_syrip_key.csv"
LABELS = ROOT / "outputs" / "relabel_syrip_labels.csv"
NAMES = {1: "看得到", 2: "部分", 3: "看不到"}


def read_csv(path, encoding="utf-8"):
    with open(path, encoding=encoding) as fh:
        return list(csv.DictReader(fh))


def load_truth(labels_path):
    """回傳 {SyRIP 檔名: 人工標註}。Excel 會把編號 001 存成 1，所以編號一律轉成數字比對。"""
    rows = read_csv(labels_path, "utf-8-sig")
    col = next(k for k in rows[0] if k.startswith("口鼻"))
    label_by_id = {int(r["id"]): int(r[col]) for r in rows if r[col].strip()}
    key = read_csv(KEY)
    return key, {k["file"]: label_by_id[int(k["id"])] for k in key if int(k["id"]) in label_by_id}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="?", type=Path, default=LABELS, help="人工重標檔（預設 outputs/relabel_syrip_labels.csv）")
    ap.add_argument("--eval-dir", type=Path, default=EVAL_DIR,
                    help="要重算的評估結果資料夾（預設 outputs/syrip_eval；灰階／關燈的在 outputs/syrip_eval_<條件>）")
    args = ap.parse_args()
    key, truth = load_truth(args.labels)
    print(f"人工標註 {len(truth)}/{len(key)} 張｜評估結果：{args.eval_dir}")
    for reason, desc in (("false_alarm", "系統判 ALERT、原標註看得到"), ("gt_hidden", "原標註看不到"),
                         ("ok_sample", "系統都判 OK 的抽查")):
        c = Counter(truth[k["file"]] for k in key if k["reason"] == reason and k["file"] in truth)
        print(f"  {desc}：" + "、".join(f"{NAMES[lab]} {c[lab]}" for lab in (1, 2, 3)))

    yes = lambda v: v == "True"
    for path in sorted(args.eval_dir.glob("*.csv")):
        found = [r for r in read_csv(path) if yes(r["found"])]
        label = {r["file"]: truth.get(r["file"], 1 if yes(r["gt_face_visible"]) else 3) for r in found}
        by = {lab: [r for r in found if label[r["file"]] == lab] for lab in (1, 2, 3)}
        fa = sum(yes(r["alert"]) for r in by[1])
        hit = sum(yes(r["alert"]) for r in by[3])
        part = sum(yes(r["alert"]) for r in by[2])
        unlabeled = sum(yes(r["alert"]) and r["file"] not in truth for r in found)
        print(f"\n[{path.stem}] 誤報率 {fa / len(by[1]):.1%}（{fa}/{len(by[1])}）"
              f"｜看不到→ALERT {hit / len(by[3]):.1%}（{hit}/{len(by[3])}）"
              f"｜部分→ALERT {part}/{len(by[2])}"
              + (f"｜⚠ {unlabeled} 張 ALERT 的圖不在重標名單，暫當看得到" if unlabeled else ""))
        posture = defaultdict(lambda: [0, 0])
        for r in by[1]:
            posture[r["posture"]][0] += yes(r["alert"])
            posture[r["posture"]][1] += 1
        print("    誤報依睡姿：" + "、".join(f"{p} {a}/{t}（{a / t:.0%}）" for p, (a, t) in sorted(posture.items())))
        # 場景相關子集：午睡室針對躺姿（Supine+Prone）；validate100 是調參時的保留集（train600 調、validate 報）
        def rate(rows):
            a, n = sum(yes(r["alert"]) for r in rows), len(rows)
            return f"{a}/{n}（{a / n:.1%}）" if n else "—"
        vis = by[1]
        lying = [r for r in vis if r["posture"] in ("Supine", "Prone")]
        if any(r.get("split") for r in vis):  # 有 split 欄才印（新版 eval_syrip 才有）
            val = [r for r in vis if r.get("split") == "validate100"]
            print(f"    誤報（場景子集）：躺姿 {rate(lying)}｜validate100 {rate(val)}"
                  f"｜validate100+躺姿 {rate([r for r in val if r['posture'] in ('Supine', 'Prone')])}")
        else:
            print(f"    誤報（場景子集）：躺姿 {rate(lying)}")


if __name__ == "__main__":
    main()
