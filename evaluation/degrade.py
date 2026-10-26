"""
degrade.py
模擬午睡室的暗光／夜視畫面，給三套評估用（eval_syrip.py、eval_doll.py、eval_video.py 的 --gray、--dim）。

  --gray   去掉顏色：午睡室拉窗簾偏暗，攝影機多半切到紅外線（IR）夜視，拍出來是灰階
  --dim    關燈：在線性亮度上乘倍率，再加感測器雜訊
兩個一起用＝補光不夠的 IR（沒有顏色、又暗又有雜訊）。

這只是近似。真的近紅外線下皮膚偏亮、偏平滑，有些深色布料會變淺，補光燈中間亮、邊緣暗，
這些都沒有模擬——要驗證還是得用夜視攝影機實拍。
"""

import cv2
import numpy as np

GAMMA = 2.2      # 模擬關燈時先把像素轉回線性亮度再乘，比直接乘像素值接近真實相機
DIM_NOISE = 2.0  # 暗處的感測器雜訊（高斯，標準差／255）


def dim_image(img, factor, seed=0):
    """模擬關燈：在線性亮度上乘 factor，再加一點雜訊（固定 seed，結果可重現）。
    這只是近似，真的暗光還要拍過才算數（現有照片全是開燈拍的）。"""
    lin = (img.astype(np.float32) / 255.0) ** GAMMA * factor
    out = 255.0 * lin ** (1 / GAMMA)
    out += np.random.default_rng(seed).normal(0, DIM_NOISE, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade(img, gray=False, dim=0.0, seed=0):
    """依條件處理一張 BGR 圖，回傳 BGR（模型輸入要 3 通道，灰階就複製三份）。
    灰階時雜訊加在單一通道上——IR 感測器的雜訊沒有顏色，三個通道分開加會讓灰階圖冒出彩色雜點。"""
    if not (gray or dim):
        return img
    out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if gray else img
    if dim:
        out = dim_image(out, dim, seed)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR) if gray else out


def tag(gray=False, dim=0.0):
    """條件的代號，接在輸出檔名／資料夾後面：gray、dim0.3、gray_dim0.3；原圖回傳空字串。"""
    return "_".join((["gray"] if gray else []) + ([f"dim{dim:g}"] if dim else []))


def describe(gray=False, dim=0.0):
    """條件的中文說明，印在報表標題。"""
    return "＋".join((["灰階（模擬 IR）"] if gray else []) + ([f"模擬關燈 ×{dim:g}"] if dim else [])) or "原圖"
