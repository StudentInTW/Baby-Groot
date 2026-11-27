"""
pipeline：每床判讀的主流程（之後要搬上 UGen300 的部分）。

  infant_face_v7.py   主流程（2026/11/27 起）：v6＋RetinaFace 補找臉——v6 判口鼻看不到的床位，才在床位裡用第二個找臉模型再找一次
                      （模型檔 model/retinaface_v1_640x480.onnx 不在 repo 裡，見 README）
  infant_face_v6.py   v3＋口鼻小塊模型——每張臉切鼻子到下巴一小塊，判口鼻露出沒有（v7 也用它判口鼻；
                      模型檔 model/mouth_patch_ens3.onnx 不在 repo 裡）
  infant_face_v3.py   找人 → 頭部轉正 → 找臉：每個嬰兒口鼻看不看得到（v6 建在它上面；真實嬰兒照片的基準）
  infant_face_v4.py   對照：v3＋口鼻膚色檢查
  infant_face_v5.py   對照：v4＋臉分三級（ok／warn／alert）
  face_common.py      共用：門檻、YuNet 找臉、把臉配給人、畫圖
  crib_detect.py      一幀：每個床位口鼻看不看得到（床位模式；主流程看 v7_slot 那一欄，沒有模型檔時退回 v6_slot、v3_slot）
  crib_state.py       每床的狀態機：3 秒警示、照護中暫停計時、空床灰燈、離線

照片在 repo 根目錄用 `uv run python -m pipeline.infant_face_v6 圖片.jpg` 執行（沒有模型檔時用 infant_face_v3）；
v7 只在床位模式多做一步（照片不分床位，結果跟 v6 一樣），`pipeline.infant_face_v7 圖片.jpg` 只畫 RetinaFace 找到的臉。
"""
