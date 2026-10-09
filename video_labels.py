"""
video_labels.py
影片標註的共用讀取：eval_video.py、vlm_spike.py、crib_context.py 都用這支。
只用標準函式庫——vlm_spike.py 跑在另一個虛擬環境（.venv-vlm，沒有 rtmlib），也能 import。

標註檔（repo 根目錄）：
  doll_video_slots.csv / doll_video_labels.csv     進 repo：自己拍的娃娃影片、AI 生成影片
  local_video_slots.csv / local_video_labels.csv   不進 repo（.gitignore）：不能公開的素材（例如網路上找到的影片）。
                                                   格式相同，只在本機評估；名稱不寫進程式和公開文件

床位檔欄位：video, doll（床位名稱）, x0, y0, x1, y1（床位範圍，畫面比例）, rot（把床位轉正要轉幾度）,
           head_x0, head_y0, head_x1, head_y1（床頭那一端，VLM 看這裡；空白＝用床位範圍）,
           empty_ref（空床參考畫面在影片第幾秒，crib_context.py 用；空白＝沒有）, note
時間軸欄位：video, doll, start, end（秒）, label（ok 口鼻看得到／alert 看不到／empty 空床）,
           staged（拍攝意圖：safe／warn／danger／prone＝趴睡但口鼻看得到）, note
"""

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
SLOT_FILES = (ROOT / "doll_video_slots.csv", ROOT / "local_video_slots.csv")
LABEL_FILES = (ROOT / "doll_video_labels.csv", ROOT / "local_video_labels.csv")
VIDEO_DIRS = (ROOT / "dataset" / "doll_video", ROOT / "dataset")
VIDEO_EXTS = (".MOV", ".mov", ".mp4", ".MP4")


def read_csv(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _num(value):
    return float(value) if value not in (None, "") else None


def load():
    """回傳 {影片: {"slots": {床位: 設定}, "segments": [段落, ...], "local": 是否來自 local_ 標註檔}}。
    床位設定：rect (x0, y0, x1, y1)、rot、head（床頭範圍或 None）、empty_ref（秒或 None）、note。
    段落的 start/end 已轉成秒數。"""
    videos = defaultdict(lambda: {"slots": {}, "segments": [], "local": False})
    for local, path in enumerate(SLOT_FILES):
        if not path.exists():
            continue
        for r in read_csv(path):
            head = tuple(_num(r.get(k)) for k in ("head_x0", "head_y0", "head_x1", "head_y1"))
            videos[r["video"]]["slots"][r["doll"]] = {
                "rect": tuple(float(r[k]) for k in ("x0", "y0", "x1", "y1")),
                "rot": int(r["rot"]),
                "head": head if None not in head else None,
                "empty_ref": _num(r.get("empty_ref")),
                "note": r.get("note", ""),
            }
            videos[r["video"]]["local"] |= bool(local)
    for local, path in enumerate(LABEL_FILES):
        if not path.exists():
            continue
        for r in read_csv(path):
            r["start"], r["end"] = float(r["start"]), float(r["end"])
            videos[r["video"]]["segments"].append(r)
            videos[r["video"]]["local"] |= bool(local)
    return dict(videos)


def find_video(name):
    """在 dataset/doll_video/、dataset/ 找同名影片（.MOV／.mp4）。"""
    for folder in VIDEO_DIRS:
        for ext in VIDEO_EXTS:
            path = folder / f"{name}{ext}"
            if path.exists():
                return path
    raise FileNotFoundError(f"找不到影片 {name}（找過 {', '.join(str(d) for d in VIDEO_DIRS)}）")
