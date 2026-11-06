"""
hailo/vlm_hef.py
在 UGen300（Hailo-10H）上跑 VLM 複判：Qwen2-VL-2B-Instruct（Apache-2.0），用 Hailo GenAI Model Zoo v5.4.0 編好的 HEF。
給 experiments/vlm_spike.py --chip、evaluation/bench_multistream.py --vlm 用，也可以直接跑冒煙測試（實驗，還沒進主流程）。

- HEF：https://dev-public.hailo.ai/v5.4.0/blob/Qwen2-VL-2B-Instruct.hef（3.26 GB，放 model/genai/，不進 repo）。
  版本要跟 HailoRT 一樣（5.4.0）；論壇上版本不合的人，建立 VLM 時就失敗（HAILO_INVALID_OPERATION）。
- 介面是 HailoRT 內建的 hailo_platform.genai.VLM（Windows 版的 pyHailoRT 就有，不用另外裝）。
  Hailo-Ollama（Ollama 相容的伺服器）只支援 LLM，不支援 VLM。
- 輸入固定 336×336 RGB uint8（vlm.input_frame_shape()）：短邊縮到 336、中間裁成正方形，跟 Hailo 的範例
  （hailo-apps 的 vlm_chat）一樣。床頭特寫本來就接近正方形（448×418–446），裁掉的很少。
- **晶片的 API 只回文字、不給機率**：greedy（do_sample=False）只生 1 個字，Yes → 1、No → 0、其他 → 0.5。
  筆電版（vlm_spike）取的是 P(Yes)、門檻調到 0.4；greedy 大約等於門檻 0.5。
- 每題問完 clear_context()：不然對話會一直累積（context 只有 2048 tokens）。
- 跟 RTMO／YuNet 共用 hailo_infer.shared_vdevice()（同一個程序，HailoRT 的排程器輪流跑）。
  一顆晶片同時只能載一個 LLM／VLM（KV cache 獨占）。斷線重連（hailo_infer._reconnect）不會重建 VLM。
- **VLM 在跑的時候 pyHailoRT 不放開 GIL**（11/06 冒煙測試：同一個程序的其他 Python 執行緒只剩 3% 的速度），
  每問一題，同一個程序的主流程就停約 1 秒。Hailo 的範例把 VLM 放在另一個程序，但**在 Windows＋USB 上第二個程序
  開不了裝置**（11/06：VDevice 用同一個 group_id 也一樣，`LIBUSB_ERROR_ACCESS`、`HAILO_LIBUSB_FAILURE`；
  Hailo-10H 不需要、也不能開 multi_process_service）。所以 VLM 只能跟主流程同一個程序（Background）。
- **現況：還不能用**。兩支驗證影片上，口鼻題 273 次只答過 1 次 Yes（連清楚的正臉也說看不到），
  筆電同一個模型改成 336 輸入就分不太開；數字和下一步見 hailo/結果.md「VLM 複判也放上晶片」。

  uv run python -m hailo.vlm_hef model/genai/Qwen2-VL-2B-Instruct.hef      # 冒煙測試：載入、輸入格式、每題幾秒
"""

import argparse
import atexit
import statistics
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# 冒煙測試預設問的四張床頭特寫（vlm_spike 存的，自己拍的娃娃影片）：仰睡正臉、頭轉側邊、包巾蓋住口鼻、臉朝下趴睡
SAMPLES = [("IMG_8429_S_0009.0.jpg", "仰睡正臉"), ("IMG_8429_S_0076.0.jpg", "頭轉側邊、口鼻看得到"),
           ("IMG_8429_S_0131.2.jpg", "包巾蓋住口鼻"), ("IMG_8432_S_0094.2.jpg", "臉朝下趴睡")]


def square(img, size):
    """短邊縮到 size、中間裁成 size×size（Hailo 範例的做法）。筆電版用 --size 336 時也走這裡，兩邊才比得起來。"""
    h, w = img.shape[:2]
    s = size / min(h, w)
    img = cv2.resize(img, (max(size, round(w * s)), max(size, round(h * s))), interpolation=cv2.INTER_AREA)
    y0, x0 = (img.shape[0] - size) // 2, (img.shape[1] - size) // 2
    return np.ascontiguousarray(img[y0:y0 + size, x0:x0 + size])


class ChipVLM:
    def __init__(self, hef):
        from hailo_platform.genai import VLM

        from hailo.hailo_infer import shared_vdevice
        vdevice = shared_vdevice()
        t0 = time.perf_counter()
        self.vlm = VLM(vdevice, str(hef))
        self.load_sec = time.perf_counter() - t0
        self.shape = tuple(self.vlm.input_frame_shape())    # (H, W, C)
        self.lock = threading.Lock()                         # 同一時間只能有一個生成
        atexit.register(self.close)                          # 在 shared_vdevice 的釋放之前執行（atexit 後進先出）

    def close(self):
        if self.vlm is not None:
            try:
                self.vlm.release()
            except Exception:
                pass
            self.vlm = None

    def frame(self, rgb):
        return square(rgb, self.shape[0]) if rgb.shape[:2] != self.shape[:2] else np.ascontiguousarray(rgb)

    def ask(self, rgb, question, max_tokens=1):
        """回傳 (回答文字, 秒數)。圖在前、題目在後，跟筆電版一樣；不給 system prompt。"""
        prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        frame = self.frame(rgb)
        with self.lock:
            t0 = time.perf_counter()
            try:
                text = self.vlm.generate_all(prompt, frames=[frame], do_sample=False, max_generated_tokens=max_tokens)
            finally:
                sec = time.perf_counter() - t0
                self.vlm.clear_context()
        return text, sec

    def yes(self, rgb, question):
        """回傳 (Yes＝1／No＝0／其他＝0.5, 秒數, 回答文字)。"""
        text, sec = self.ask(rgb, question)
        a = text.strip().lower()
        return (1.0 if a.startswith("yes") else 0.0 if a.startswith("no") else 0.5), sec, text


class Background:
    """同一個程序裡另開一條執行緒問 VLM 同一題（bench_multistream --vlm 用），量「主流程和 VLM 同時用晶片」時兩邊各多慢。
    every＝每題之間隔幾秒（0＝不停地問）。start()／pause() 之間才問；window(t0, t1) 回傳這段時間內
    (每秒答幾題, 每題秒數中位數, 題數)。"""

    def __init__(self, hef, rgb, question, every=0.0):
        self.vlm = ChipVLM(hef)
        self.vlm.ask(rgb, question)          # 暖機
        self.rgb, self.question, self.every = rgb, question, every
        self.log, self.thread = [], None

    def _loop(self, stop):
        while not stop.is_set():
            _, sec = self.vlm.ask(self.rgb, self.question)
            self.log.append((time.time(), sec))
            if self.every:
                stop.wait(self.every)

    def start(self):
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, args=(self._stop,), daemon=True)
        self.thread.start()

    def pause(self):
        if self.thread is not None:
            self._stop.set()
            self.thread.join(30)
            self.thread = None

    def window(self, t0, t1):
        secs = [s for t, s in self.log if t0 <= t <= t1]
        return len(secs) / (t1 - t0), (statistics.median(secs) if secs else float("nan")), len(secs)


def gil_check(vlm, rgb, question):
    """問題在晶片上跑的時候，Python 的其他執行緒還能不能跑（pyHailoRT 有沒有放開 GIL）：
    另一條執行緒一直數數，比較「閒著」和「VLM 在跑」時每秒數到多少。"""
    stop, count = threading.Event(), [0]

    def counter():
        while not stop.is_set():
            count[0] += 1

    t = threading.Thread(target=counter)
    t.start()
    time.sleep(0.5)
    c0, t0 = count[0], time.perf_counter()
    time.sleep(1.0)
    idle = (count[0] - c0) / (time.perf_counter() - t0)
    c0, t0 = count[0], time.perf_counter()
    vlm.ask(rgb, question)
    busy = (count[0] - c0) / (time.perf_counter() - t0)
    stop.set()
    t.join()
    return busy / idle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hef")
    ap.add_argument("--images", nargs="*", help="要問的圖（預設：outputs/vlm_spike/crops 裡的四張娃娃床頭特寫）")
    ap.add_argument("--repeat", type=int, default=3, help="每張圖的口鼻題問幾次（預設 3）")
    args = ap.parse_args()
    from experiments.vlm_spike import QUESTIONS

    if args.images:
        samples = [(Path(p), Path(p).name) for p in args.images]
    else:
        samples = [(ROOT / "outputs" / "vlm_spike" / "crops" / n, note) for n, note in SAMPLES]
    images = [(cv2.cvtColor(cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), note)
              for p, note in samples if p.exists()]
    if not images:
        raise SystemExit("找不到要問的圖：先跑 experiments.vlm_spike，或用 --images 指定")

    vlm = ChipVLM(args.hef)
    v = vlm.vlm
    print(f"載入 {vlm.load_sec:.1f} 秒｜輸入 {vlm.shape}、{v.input_frame_format_type()}、{v.input_frame_format_order()}"
          f"｜context 上限 {v.max_context_capacity()} tokens")

    # 1. 自由描述一張圖：確認模型真的看得到圖，順便量第一個字多久出來、之後每秒幾個字
    rgb = images[0][0]
    prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image in one sentence."}]}]
    stamps, toks = [], []
    t0 = time.perf_counter()
    with vlm.lock:
        with v.generate(prompt, frames=[vlm.frame(rgb)], do_sample=False, max_generated_tokens=48) as gen:
            for tok in gen:
                stamps.append(time.perf_counter() - t0)
                toks.append(tok)
        v.clear_context()
    tps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else float("nan")
    print(f"\n描述（{images[0][1]}）：{''.join(toks).strip()}")
    print(f"  第一個字 {stamps[0]:.2f} 秒、之後每秒 {tps:.1f} 個字（{len(toks)} 個字）")

    # 2. 口鼻題（只生 1 個字）：每張圖問幾次，看答案和秒數
    print(f"\n口鼻題「{QUESTIONS['mouth']}」，每張問 {args.repeat} 次：")
    secs = []
    for rgb, note in images:
        answers = []
        for _ in range(args.repeat):
            _, sec, text = vlm.yes(rgb, QUESTIONS["mouth"])
            answers.append(text.strip())
            secs.append(sec)
        print(f"  {note:<14} → {' / '.join(answers)}")
    print(f"  每題 {statistics.median(secs):.2f} 秒（中位數；最快 {min(secs):.2f}、最慢 {max(secs):.2f}）")

    # 3. GIL：VLM 在跑的時候，同一個程序裡的其他執行緒（主流程）還跑不跑得動
    ratio = gil_check(vlm, images[0][0], QUESTIONS["mouth"])
    print(f"\nVLM 在跑時，另一條 Python 執行緒的速度是閒著時的 {ratio:.0%}"
          + ("（GIL 有放開）" if ratio > 0.5 else "（GIL 沒放開：同一個程序的主流程會被卡住）"))


if __name__ == "__main__":
    main()
