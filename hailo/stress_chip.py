"""
hailo/stress_chip.py
UGen300 壓力測試：裝置自己重開（`HAILO_COMMUNICATION_CLOSED`）跟哪個模型、哪種用法有關？

三種情況各跑 --sec 秒（隨機輸入、一次一張、跑完一張才送下一張，跟主流程一樣）：
  rtmo   只跑 RTMO
  yunet  只跑 YuNet
  alt    RTMO 一張、YuNet 一張輪流（每張都要切換模型）
hailo_infer 斷線會自動重連並計數；每種情況結束印：張數、每秒幾張、重連幾次。結果記在 hailo/結果.md「裝置斷線」。

  uv run python -m hailo.stress_chip model/rtmo_D1.hef model/yunet_A.hef --sec 180
"""

import argparse
import time
from pathlib import Path

import numpy as np

import hailo.hailo_infer as hi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rtmo", type=Path)
    ap.add_argument("yunet", type=Path)
    ap.add_argument("--sec", type=float, default=180, help="每種情況跑幾秒（預設 180）")
    args = ap.parse_args()

    imgs = np.random.default_rng(0).integers(0, 256, (8, 640, 640, 3), dtype=np.uint8)
    rtmo, yunet = hi.Hef(args.rtmo), hi.Hef(args.yunet)
    for name, seq in (("rtmo", [rtmo]), ("yunet", [yunet]), ("alt", [rtmo, yunet])):
        r0, t0, n = hi.reconnects, time.time(), 0
        while time.time() - t0 < args.sec:
            for model in seq:
                model(imgs[n % len(imgs)])
                n += 1
        dt = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] {name:5s} {dt:.0f} 秒：{n} 張（{n / dt:.1f} 張/秒），"
              f"重連 {hi.reconnects - r0} 次", flush=True)
    print(f"全部完成，總共重連 {hi.reconnects} 次", flush=True)


if __name__ == "__main__":
    main()
