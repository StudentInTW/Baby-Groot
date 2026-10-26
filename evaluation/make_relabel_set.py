"""
make_relabel_set.py
產生「SyRIP 誤報圖人工重標」的資料包，任務說明見 docs/任務_SyRIP重標.md。

挑哪些圖（讀 outputs/syrip_eval/ 裡所有評估結果的 CSV）：
  - 誤報圖：GT 標「臉看得到」、但任一次評估判 ALERT 的圖——要確認臉到底看不看得到
  - GT 標「臉看不到」的圖——順便確認
  - 隨機 30 張系統都判 OK 的圖——抽查「判 OK 的是不是真的看得到」，也讓標註的人猜不到哪些是誤報
全部打亂、改成編號檔名；圖上只畫藍框標出要判斷的嬰兒，不顯示系統的判斷。

輸出：
  outputs/relabel_syrip/          整包給標註的人（images/、labels.csv、說明.md）
  outputs/relabel_syrip.zip       上面那包壓成 zip
  outputs/relabel_syrip_key.csv   編號 → SyRIP 檔名、挑選原因（自己留著，不要給標註的人）

用法：
    uv run python -m evaluation.make_relabel_set
"""

import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from evaluation.eval_syrip import OUT_DIR as EVAL_DIR, load_syrip

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "outputs" / "relabel_syrip"
KEY = ROOT / "outputs" / "relabel_syrip_key.csv"
N_OK = 30         # 抽查幾張系統都判 OK 的圖
MAX_SIDE = 1280   # 圖縮到長邊不超過這個大小，壓小檔案
SEED = 0          # 固定亂數，重跑會得到一樣的資料包


def read_eval_rows():
    rows = []
    for path in sorted(EVAL_DIR.glob("*.csv")):
        with open(path, encoding="utf-8") as fh:
            rows += list(csv.DictReader(fh))
    return rows


def main():
    rows = read_eval_rows()
    yes = lambda v: v == "True"
    alerted = {r["file"] for r in rows if yes(r["gt_face_visible"]) and yes(r["alert"])}
    hidden = {r["file"] for r in rows if not yes(r["gt_face_visible"])}
    ok_pool = sorted({r["file"] for r in rows if yes(r["gt_face_visible"]) and yes(r["found"])} - alerted)

    rng = random.Random(SEED)
    picks = ([(f, "false_alarm") for f in sorted(alerted)] + [(f, "gt_hidden") for f in sorted(hidden)]
             + [(f, "ok_sample") for f in rng.sample(ok_pool, min(N_OK, len(ok_pool)))])
    rng.shuffle(picks)

    z, samples = load_syrip()
    by_file = {s["file"]: s for s in samples}
    shutil.rmtree(PKG, ignore_errors=True)
    (PKG / "images").mkdir(parents=True)
    key = []
    for i, (f, reason) in enumerate(picks, 1):
        s = by_file[f]
        img = cv2.imdecode(np.frombuffer(z.read(s["path"]), np.uint8), cv2.IMREAD_COLOR)
        x1, y1, x2, y2 = s["box"].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), max(3, round(max(img.shape[:2]) / 300)))
        h, w = img.shape[:2]
        scale = min(1.0, MAX_SIDE / max(h, w))
        if scale < 1:
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(PKG / "images" / f"{i:03d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        key.append({"id": f"{i:03d}", "file": f, "reason": reason, "posture": s["posture"]})

    with open(PKG / "labels.csv", "w", newline="", encoding="utf-8-sig") as fh:  # utf-8-sig：Excel 開才不會亂碼
        writer = csv.writer(fh)
        writer.writerow(["id", "口鼻（1看得到／2部分或不確定／3看不到）", "備註"])
        writer.writerows([[k["id"], "", ""] for k in key])
    with open(KEY, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(key[0]))
        writer.writeheader()
        writer.writerows(key)
    shutil.copy(ROOT / "docs" / "任務_SyRIP重標.md", PKG / "說明.md")
    archive = Path(shutil.make_archive(str(PKG), "zip", PKG))

    counts = {r: sum(k["reason"] == r for k in key) for r in ("false_alarm", "gt_hidden", "ok_sample")}
    print(f"共 {len(key)} 張：誤報 {counts['false_alarm']}、GT 臉看不到 {counts['gt_hidden']}、抽查 OK {counts['ok_sample']}")
    print(f"→ {archive.relative_to(ROOT)}（{archive.stat().st_size / 1e6:.0f} MB）；"
          f"對照表 {KEY.relative_to(ROOT)}（自己留著，不要給標註的人）")


if __name__ == "__main__":
    main()
