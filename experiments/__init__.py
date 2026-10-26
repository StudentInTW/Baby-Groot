"""
experiments：早期版本和探路實驗，留著對照、不在主流程裡（結論見 docs/）。

  infant_prone_spike.py   v1：關鍵點信心度規則（已知會漏報；要先 uv sync --group yolo）
  infant_face_spike.py    v2：YOLOv8 找人＋整張圖找臉（要先 uv sync --group yolo）
  spike_posture.py        用關鍵點幾何分仰／趴（結論：此路不通）
  sideface_spike.py       臉分三級的可行性（後來成為 pipeline/infant_face_v5.py）
  vlm_spike.py            VLM 第二道檢查（另一個虛擬環境 .venv-vlm）
"""
