"""
vlm_spike.py
VLM 複判 spike（2026/10/9，實驗，還沒進主流程）——兩個問題：
  1. 跳紅燈的那幾幀，叫小型 VLM 再看一次，分不分得出「口鼻真的被蓋住」和側臉、抱枕圖案、寢具印花？
     （設計界線：VLM 只能加註、排序，不能把紅燈關掉——這裡也量「如果拿來關紅燈會漏掉幾個」，證明為什麼不行）
  2. VLM 判不判得出趴睡？臉側向、口鼻看得到的趴睡，現在的流程判不出來（關鍵點判仰／趴召回只有 7–21%）。

模型：Qwen2-VL-2B-Instruct（Apache-2.0）。Hailo 官方 GenAI model zoo 有為 Hailo-10H 編好的版本，
所以筆電上的結果可以當「搬上晶片」的參考（晶片上是量化版，數字會不同）。
筆電的 RTX 3050 只有 4 GB，fp16 整個模型（約 4.4 GB）放不下：語言模型用 bitsandbytes 4-bit，視覺編碼器維持 fp16。

取樣：doll_video_labels.csv 每個乾淨片段取 1/4、1/2、3/4 三幀，長片段每 8 秒再多取一幀；
      裁出床位範圍、依床位方向轉正（跟 eval_video.py 的床位找臉一樣）。
      每幀的 v3＋床位判定從 outputs/video_eval/<影片>_frames.csv 讀最接近的一幀（要先跑過 eval_video.py）。
問法：同一張圖問三題（口鼻看不看得到、臉有沒有被蓋住、是不是趴睡），各取下一個字是 Yes／No 的機率（不讓模型自由生成，才能調門檻）。

環境（裝在另一個虛擬環境，主程式不需要 PyTorch）：
    uv venv .venv-vlm --python 3.12
    uv pip install --python .venv-vlm torch torchvision --index-url https://download.pytorch.org/whl/cu126
    uv pip install --python .venv-vlm transformers accelerate bitsandbytes opencv-python pillow
用法：
    .venv-vlm/Scripts/python vlm_spike.py                # 全部片段
    .venv-vlm/Scripts/python vlm_spike.py --limit 10     # 試跑前 10 幀
    .venv-vlm/Scripts/python vlm_spike.py --reuse        # 不重跑 VLM，用上次的 frames.csv 重算摘要
    .venv-vlm/Scripts/python vlm_spike.py IMG_8429 IMG_8432   # 只跑指定影片（驗證門檻用），存成 frames_<影片>.csv
輸出：outputs/vlm_spike/frames.csv（每幀的機率）、crops/（餵給 VLM 的裁切圖，方便人工檢查）、終端機摘要
"""

import argparse
import csv
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

import video_labels
from video_labels import read_csv

ROOT = Path(__file__).parent
ANNOTATIONS = video_labels.load()
NOT_DOLL = {"嬰兒翻身"} | {v for v, a in ANNOTATIONS.items() if a["local"]}  # AI 生成影片、local 標註的影片不算娃娃
EVAL_DIR = ROOT / "outputs" / "video_eval"
OUT_DIR = ROOT / "outputs" / "vlm_spike"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
VARIANTS = ("raw", "jpeg", "flip")  # 見 variants()
# 每床鏡頭（10/10 補拍）整個畫面就是床位，整張圖縮到 448 臉只剩約 20 像素，VLM 看不出來（10/10 試過：
# 仰躺正臉只有 41% 判成看得到、非趴睡 98% 被說成趴睡）——所以 VLM 改看「床頭那一端」：床位檔的 head_x0..head_y1，
# 安裝時跟床位一起設定。舊影片的床位本來就是頭部範圍，這幾欄空白。
CROP = 448          # 裁切圖長邊；Qwen2-VL 每 28×28 像素一個視覺 token，448² ≈ 256 個
EXTRA_EVERY = 8     # 長片段每幾秒多取一幀
ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
# 問法只在一張仰躺娃娃圖上比過一次（長句「clearly visible and not covered」P(Yes) 只有 0.70，短句 0.90），之後不再調，避免對著答案改題目
QUESTIONS = {
    "mouth": "Can you see the baby's nose and mouth? Answer Yes or No.",
    "covered": "Is the baby's face covered by a blanket, cloth, toy, or the mattress? Answer Yes or No.",
    "prone": "Is the baby lying on its stomach, with its back facing up? Answer Yes or No.",
}


def sample_points(videos=()):
    """回傳 [(影片, 娃娃, 秒, 標註列, 床位設定), ...]；videos 有給就只取這些影片。"""
    points = []
    for video, ann in ANNOTATIONS.items():
        if videos and video not in videos:
            continue
        for seg in ann["segments"]:
            start, end = seg["start"], seg["end"]
            ts = [start + (end - start) * q for q in (0.25, 0.5, 0.75)]
            ts += list(np.arange(start + EXTRA_EVERY, end - 1, EXTRA_EVERY))
            for t in sorted(set(round(t, 1) for t in ts)):
                points.append((video, seg["doll"], t, seg, ann["slots"][seg["doll"]]))
    return points


def crop_slot(frame, slot):
    """裁出床頭範圍（沒設定就用床位範圍）、依床位方向轉正，縮到長邊 CROP。"""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = slot["head"] or slot["rect"]
    crop = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    rot = slot["rot"]
    crop = cv2.rotate(crop, ROTATE[rot]) if rot else crop
    s = CROP / max(crop.shape[:2])
    return cv2.resize(crop, (round(crop.shape[1] * s), round(crop.shape[0] * s)), interpolation=cv2.INTER_AREA)


def v3_slot_at(cache, video, doll, t):
    """outputs/video_eval/<影片>_frames.csv 裡最接近 t 的那幀，v3＋床位的判定（visible／hidden）。"""
    if video not in cache:
        path = EVAL_DIR / f"{video}_frames.csv"
        cache[video] = read_csv(path) if path.exists() else []
    rows = [r for r in cache[video] if r["doll"] == doll]
    if not rows:
        return ""
    return min(rows, key=lambda r: abs(float(r["t"]) - t))["v3_slot"]


def load_vlm(cpu):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
    if cpu:
        model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.float32)
    else:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                                   llm_int8_skip_modules=["visual", "lm_head"])  # 視覺編碼器和輸出層不量化
        model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, quantization_config=quant,
                                                                dtype=torch.float16, device_map="cuda:0")
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=128 * 28 * 28, max_pixels=CROP * CROP)
    tok = processor.tokenizer
    yes_no = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No")]

    def ask(rgb, question):
        """回傳 (P(Yes)，秒數)；P 只在 Yes／No 兩個字之間正規化。"""
        from PIL import Image
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text], images=[Image.fromarray(rgb)], return_tensors="pt").to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, yes_no].float()
        p_yes = torch.softmax(logits, dim=0)[0].item()
        return p_yes, time.perf_counter() - t0

    return ask


def variants(crop):
    """同一幀的三個版本：原圖、重新 JPEG 壓縮、左右翻轉。
    10/9 試跑發現：同一張圖原圖和 JPEG（平均每像素只差 1 階）P(Yes) 會從 0.42 跳到 0.92，
    fp32 不量化也一樣（0.35／0.87）——2B 模型在邊界附近很不穩，所以每幀問三個版本取平均，也量不穩的程度。"""
    jpeg = cv2.imdecode(cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR)
    return [("raw", crop), ("jpeg", jpeg), ("flip", cv2.flip(crop, 1))]


def run(args):
    ask = load_vlm(args.cpu)
    (OUT_DIR / "crops").mkdir(parents=True, exist_ok=True)
    points = sample_points(args.videos)[:args.limit or None]
    cache, caps, rows = {}, {}, []
    for i, (video, doll, t, seg, slot) in enumerate(points, 1):
        if video not in caps:
            caps[video] = cv2.VideoCapture(str(video_labels.find_video(video)))
        cap = caps[video]
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            print(f"[略過] 讀不到 {video} {t}s")
            continue
        crop = crop_slot(frame, slot)
        name = f"{video}_{doll}_{t:06.1f}.jpg"
        cv2.imencode(".jpg", crop)[1].tofile(str(OUT_DIR / "crops" / name))  # cv2.imwrite 在中文路徑會失敗
        row = {"video": video, "doll": doll, "t": t, "label": seg["label"], "staged": seg["staged"], "note": seg["note"],
               "prone_gt": int("趴" in seg["note"]), "v3_slot": v3_slot_at(cache, video, doll, t), "crop": name}
        secs = []
        for vname, img in variants(crop):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for q, question in QUESTIONS.items():
                p, sec = ask(rgb, question)
                row[f"{q}_{vname}"] = round(p, 4)
                secs.append(sec)
        for q in QUESTIONS:  # 三個版本平均＝這一幀的分數
            row[f"p_{q}"] = round(statistics.mean(row[f"{q}_{v}"] for v in VARIANTS), 4)
        row["sec_per_question"] = round(statistics.median(secs), 3)
        rows.append(row)
        print(f"[{i}/{len(points)}] {video} {doll} {t:6.1f}s {seg['label']:<5} {seg['note'][:18]:<18}"
              f" v3+slot={row['v3_slot']:<7} 口鼻看得到 {row['p_mouth']:.2f}（原圖 {row['mouth_raw']:.2f}／JPEG "
              f"{row['mouth_jpeg']:.2f}）｜被蓋住 {row['p_covered']:.2f}｜趴睡 {row['p_prone']:.2f}")
    with open(args.frames_csv, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize(rows, th):
    for r in rows:
        for k in r:
            if k.startswith(tuple(f"{q}_" for q in QUESTIONS) + ("p_",)) and k != "prone_gt":
                r[k] = float(r[k])
        r["prone_gt"] = int(r["prone_gt"])
    doll = [r for r in rows if r["video"] not in NOT_DOLL]

    def rate(rs, cond):
        n = sum(1 for r in rs if cond(r))
        return f"{n}/{len(rs)}（{n / len(rs):.0%}）" if rs else "—"

    print(f"\n== 1. 口鼻看不看得到（VLM P(Yes) ≥ {th} 算看得到；只算娃娃影片，{len(doll)} 幀）")
    groups = {
        "安全：仰躺正臉": lambda r: r["label"] == "ok" and "側" not in r["note"],
        "安全：側臉口鼻看得到": lambda r: r["label"] == "ok" and "側" in r["note"] and not r["prone_gt"],
        "趴睡但臉側向、口鼻看得到": lambda r: r["label"] == "ok" and r["prone_gt"],
        "危險：被子／衣服／玩偶蓋住": lambda r: r["label"] == "alert" and not r["prone_gt"] and r["staged"] == "danger",
        "危險：趴睡臉朝下／埋抱枕": lambda r: r["label"] == "alert" and r["prone_gt"],
        "看不到：側躺後腦勺朝鏡頭": lambda r: r["label"] == "alert" and r["staged"] == "warn" and not r["prone_gt"],
    }
    for title, cond in groups.items():
        rs = [r for r in doll if cond(r)]
        want_visible = title.startswith(("安全", "趴睡但"))
        print(f"    {title:<16} 口鼻題判對 {rate(rs, lambda r: (r['p_mouth'] >= th) == want_visible)}"
              f"｜被蓋住題判對 {rate(rs, lambda r: (r['p_covered'] < 0.5) == want_visible)}")

    hidden = [r for r in doll if r["v3_slot"] == "hidden"]
    fa = [r for r in hidden if r["label"] == "ok" and r["staged"] != "prone"]  # 趴睡但口鼻看得到：政策本來就要響，不算誤報
    real = [r for r in hidden if r["label"] == "alert"]
    empty = [r for r in doll if r["label"] == "empty"]
    if empty:
        print(f"    {'空床':<16} 口鼻題說看得到 {rate(empty, lambda r: r['p_mouth'] >= th)}"
              f"｜趴睡題說是 {rate(empty, lambda r: r['p_prone'] >= 0.5)}")
    print(f"\n== 2. 複判：v3＋床位判「看不到」的 {len(hidden)} 幀")
    print(f"    其中標註安全（誤報）{len(fa)} 幀：VLM 說看得到 → 可以標「可能誤報」{rate(fa, lambda r: r['p_mouth'] >= th)}")
    print(f"    其中標註危險 {len(real)} 幀：VLM 說看得到 → 若拿來關紅燈會漏掉 {rate(real, lambda r: r['p_mouth'] >= th)}")
    for t2 in (0.3, 0.35, 0.4, 0.45, 0.5, 0.7, 0.9):
        print(f"      門檻 {t2}：誤報幀可標註 {sum(r['p_mouth'] >= t2 for r in fa)}/{len(fa)}"
              f"、危險幀被說成看得到 {sum(r['p_mouth'] >= t2 for r in real)}/{len(real)}")

    # 反方向（10/10 補拍發現：頭轉開時 YuNet 把耳朵／頭側當成臉 → 綠燈，漏掉的是危險）：
    # v3＋床位判「看得到」的幀，VLM 說看不到 → 能不能當第二道網、把漏掉的危險補成「請看一眼」
    shown = [r for r in doll if r["v3_slot"] == "visible"]
    missed = [r for r in shown if r["label"] == "alert"]
    fine = [r for r in shown if r["label"] == "ok" and r["staged"] != "prone"]
    print(f"\n== 2b. 補抓：v3＋床位判「看得到」的 {len(shown)} 幀（不含空床）")
    for t2 in (0.3, 0.35, 0.4, 0.45, 0.5, 0.7):
        print(f"      門檻 {t2}：漏掉的危險幀 VLM 說看不到 {sum(r['p_mouth'] < t2 for r in missed)}/{len(missed)}"
              f"、安全幀 VLM 說看不到（會多出的提醒）{sum(r['p_mouth'] < t2 for r in fine)}/{len(fine)}")

    print("\n== 3. 趴睡（VLM P(Yes) ≥ 0.5 算趴睡）")
    for title, rs in (("娃娃：標註趴睡", [r for r in doll if r["prone_gt"]]),
                      ("娃娃：趴睡但口鼻看得到", [r for r in doll if r["prone_gt"] and r["label"] == "ok"]),
                      ("娃娃：非趴睡", [r for r in doll if not r["prone_gt"] and r["label"] != "empty"]),
                      ("AI 生成翻身影片：趴", [r for r in rows if r["video"] == "嬰兒翻身" and r["prone_gt"]]),
                      ("AI 生成翻身影片：非趴", [r for r in rows if r["video"] == "嬰兒翻身" and not r["prone_gt"]])):
        want = "非" not in title
        print(f"    {title:<14} 判對 {rate(rs, lambda r: (r['p_prone'] >= 0.5) == want)}")

    print("\n== 4. 穩定度：同一幀三個版本（原圖／JPEG／翻轉）的答案落在 0.5 兩邊的比例")
    for q in QUESTIONS:
        flip_flop = sum(len({r[f"{q}_{v}"] >= 0.5 for v in VARIANTS}) > 1 for r in rows)
        spread = statistics.median(max(r[f"{q}_{v}"] for v in VARIANTS) - min(r[f"{q}_{v}"] for v in VARIANTS) for r in rows)
        print(f"    {q:<8} 不一致 {flip_flop}/{len(rows)}（{flip_flop / len(rows):.0%}），最高與最低差的中位數 {spread:.2f}")

    secs = statistics.median(float(r["sec_per_question"]) for r in rows)
    print(f"\n== 5. 速度：每問一題中位數 {secs:.2f} 秒（RTX 3050 4GB、語言模型 4-bit；不含讀影片、裁切）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前幾幀（試跑用；會覆蓋上次的 frames.csv）")
    ap.add_argument("--reuse", action="store_true", help="不重跑 VLM，讀上次的 frames.csv 重算摘要")
    ap.add_argument("--cpu", action="store_true", help="沒有 GPU 時用 CPU（fp32，很慢）")
    ap.add_argument("--th", type=float, default=0.5, help="P(口鼻看得到) 的門檻（預設 0.5）")
    ap.add_argument("videos", nargs="*", help="只跑這些影片（不給＝全部）；結果另存 frames_<影片>.csv，不蓋掉全部的")
    args = ap.parse_args()
    args.frames_csv = OUT_DIR / (f"frames_{'_'.join(args.videos)}.csv" if args.videos else "frames.csv")
    rows = read_csv(args.frames_csv) if args.reuse else run(args)
    summarize(rows, args.th)
    print(f"\n輸出：{OUT_DIR}")


if __name__ == "__main__":
    main()
