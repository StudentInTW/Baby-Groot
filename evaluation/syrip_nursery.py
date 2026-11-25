"""
syrip_nursery.py（2026/11/24）
SyRIP 的主要誤報數字改看「像嬰兒室」的子集：SyRIP 多半是 2–10 個月、在家裡手持平拍的寶寶，很多會坐、爬、站，
月子中心的新生兒不會有這些姿勢，全部 700 張的誤報率混了太多不相干的畫面。子集是看結果前逐張定的
（2026/11/24；清單在 outputs/syrip_subsets/，跟人工重標一樣不進 repo）：
  像嬰兒室  仰躺、口鼻看得到、沒有大人的手入鏡、一個寶寶單獨躺著、臉夠大（眼到嘴 ≥ 46 像素，約等於 1080p 每床鏡頭上的臉）
  拿掉      奶嘴、玩具在嘴前（算看得到還是看不到沒定，不計分）
  手擋口鼻  算危險，另外列「有警示」
真實標註跟 score_relabel.py 一樣（人工重標；不在重標名單裡的照 SyRIP 原標）。只算 eval_syrip 有找到這個寶寶的圖。

用法（先跑 eval_syrip，例如 --face v3、v4、v6 --models rtmo）：
    uv run python -m evaluation.syrip_nursery v3 v4 v6
    uv run python -m evaluation.syrip_nursery v6 --eval-dir outputs/syrip_eval_chip-rtmo_D1_chip-mouth_A   # 模型在晶片上的那份
輸出：終端機表格（讀 outputs/syrip_eval/rtmo_<版本>.csv；--eval-dir 換資料夾，跟 score_relabel.py 一樣）
"""
import argparse
import csv
from pathlib import Path

from evaluation.eval_syrip import OUT_DIR as EVAL_DIR
from evaluation.score_relabel import LABELS, load_truth

ROOT = Path(__file__).resolve().parent.parent
SUBSETS = ROOT / "outputs" / "syrip_subsets"
NURSERY = SUBSETS / "nursery_tags.csv"        # 仰躺、口鼻看得到的每一張：nursery_like＝1 是像嬰兒室
MOUTH_OBJECTS = SUBSETS / "mouth_objects.csv"  # kind：pacifier 奶嘴／toy 玩具（拿掉）／hand 手擋口鼻（算危險）


def read_csv(path, enc="utf-8-sig"):
    with open(path, encoding=enc) as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("versions", nargs="*", default=["v3", "v4", "v6"])
    ap.add_argument("--eval-dir", type=Path, default=EVAL_DIR,
                    help="eval_syrip 的輸出資料夾（灰階、換了模型或放晶片時是 outputs/syrip_eval_<條件>）")
    args = ap.parse_args()
    vers = args.versions
    for p in (NURSERY, MOUTH_OBJECTS, LABELS):
        if not p.exists():
            raise SystemExit(f"找不到 {p}（子集清單和人工重標不在 repo 裡）")
    _, truth = load_truth(LABELS)
    kinds = {r["file"]: r["kind"] for r in read_csv(MOUTH_OBJECTS)}
    nursery = {r["file"] for r in read_csv(NURSERY) if r["nursery_like"] == "1"}
    data = {v: {r["file"]: r for r in read_csv(args.eval_dir / f"rtmo_{v}.csv", "utf-8")} for v in vers}
    first = data[vers[0]]

    def visible(f):   # 1 看得到／2 部分／3 看不到（人工重標優先）
        return truth.get(f, 1 if first[f]["gt_face_visible"] == "True" else 3)

    base = [f for f in first if kinds.get(f) not in ("pacifier", "toy")]
    sup_vis = [f for f in base if first[f]["posture"] == "Supine" and kinds.get(f) != "hand" and visible(f) == 1]
    rows = [
        ("像嬰兒室的仰躺、口鼻看得到：誤報", [f for f in sup_vis if f in nursery]),
        ("其他仰躺、口鼻看得到：誤報", [f for f in sup_vis if f not in nursery]),
        ("全部睡姿、口鼻看得到：誤報", [f for f in base if kinds.get(f) != "hand" and visible(f) == 1]),
        ("手擋住口鼻：有警示", [f for f in base if kinds.get(f) == "hand"]),
        ("口鼻看不到：有警示", [f for f in base if kinds.get(f) != "hand" and visible(f) == 3]),
    ]
    print("| 項目 | " + " | ".join(vers) + " |\n|---|" + "---|" * len(vers))
    for name, keep in rows:
        cells = []
        for v in vers:
            fs = [f for f in keep if data[v][f]["found"] == "True"]
            a = sum(data[v][f]["alert"] == "True" for f in fs)
            cells.append(f"{a}/{len(fs)}（{a / max(len(fs), 1):.1%}）")
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
