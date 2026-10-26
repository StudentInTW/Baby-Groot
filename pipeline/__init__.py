"""
pipeline：每床判讀的主流程（之後要搬上 UGen300 的部分）。

  infant_face_v3.py   找人 → 頭部轉正 → 找臉：每個嬰兒口鼻看不看得到（真實嬰兒基準）
  infant_face_v4.py   實驗：v3＋口鼻膚色檢查
  infant_face_v5.py   實驗：v4＋臉分三級（ok／warn／alert）
  face_common.py      共用：門檻、YuNet 找臉、把臉配給人、畫圖
  crib_detect.py      一幀：每個床位口鼻看不看得到（床位模式）
  crib_state.py       每床的狀態機：3 秒警示、照護中暫停計時、空床灰燈

在 repo 根目錄用 `uv run python -m pipeline.infant_face_v3 圖片.jpg` 執行。
"""
