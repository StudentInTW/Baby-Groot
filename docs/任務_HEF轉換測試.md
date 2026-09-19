# 任務：模型轉 HEF 可行性測試（不需要晶片）

**給**：士丞　**目的**：趁晶片還沒到，先確認我們的模型轉得成 Hailo-10H 的 HEF、量化後準確度沒掉太多。卡關越早發現越好。

## 我們用到的模型

系統要跑**兩種模型**：姿態模型找出畫面裡的每個嬰兒和關鍵點，人臉模型判斷口鼻看不看得到。警報邏輯在主機 CPU 上跑，不用進 HEF。

| 模型 | 用途 | ONNX 輸入 | 授權 | 下載 |
|---|---|---|---|---|
| **RTMO-m**（主要候選） | 一次找出所有人＋17 個關鍵點 | 1×3×640×640 | Apache-2.0 | [zip](https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip) |
| **YuNet 2023mar**（必要） | 找臉 | 1×3×640×640（固定） | MIT | [onnx](https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx)，專案 `model/` 裡也有 |
| **YOLOX-m**（備案之一） | 找人 | 1×3×640×640 | Apache-2.0 | [zip](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip) |
| **RTMPose-m**（備案之二） | 對每個人各自估 17 個關鍵點 | N×3×256×192 | Apache-2.0 | [zip](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip) |

備案＝YOLOX 先找人，RTMPose 再對每個人估關鍵點。SyRIP 評估裡，RTMO 和備案的誤報率差不多（約 6%），RTMO 找到嬰兒的比例較高（97.7% 對 94.7%）。

## 先看這個：預計會卡在哪

| 模型 | ONNX 輸出 | 風險 |
|---|---|---|
| RTMO-m | `dets` [1, N, 5]、`keypoints` [1, N, 17, 3]，N 是動態的 | **最高**。解碼（框、關鍵點座標）包在 ONNX 裡，人數 N 是動態的；Hailo 要固定形狀，所以要把 end node 切在解碼之前，解碼改在主機上做。RTMO 的關鍵點解碼比較特殊（座標分類），要看懂原始碼才能重寫。 |
| YOLOX-m | `dets` [1, N, 5]、`labels` [1, N]，N 是動態的 | 中。NMS 包在 ONNX 裡，要切在 NMS 之前；YOLOX 是常見架構，解碼有現成寫法。 |
| RTMPose-m | `simcc_x` [N, 17, 384]、`simcc_y` [N, 17, 512] | **低**。輸出就是網路本身，主機端只要取最大值。 |
| YuNet | 12 個輸出（`cls／obj／bbox／kps` × stride 8／16／32） | 低。輸出是原始特徵，解碼在主機端（現在由 OpenCV 代勞，之後要自己寫）。輸入固定 640×640。 |

**所以 RTMO 和備案兩條路都要試**：RTMO 準確度較好，但部署可能最難，轉不過就改用備案。
用 [Netron](https://netron.app) 打開 ONNX，看最後幾層是什麼運算，找出要切的位置。

## 前處理（HEF 跟 ONNX 要吃一樣的輸入）

- **RTMO、YOLOX**：BGR、0–255、不做正規化；等比例縮放到 640×640、放左上角，其餘補 114。實作見 rtmlib 的 `tools/pose_estimation/rtmo.py`、`tools/object_detection/yolox.py`。
- **RTMPose**：依人框裁切成 192×256（框先放大 1.25 倍），再做 mean／std 正規化。實作見 rtmlib 的 `tools/pose_estimation/rtmpose.py`。
- **YuNet**：BGR、0–255，固定 640×640。我們會對整張圖和每個嬰兒的頭部區域各跑一次，HEF 版兩種都縮放成 640×640 再餵進去。

我們怎麼呼叫這些模型：專案裡的 `infant_face_v3.py`（`load_rtmo`、`head_faces`）和 `infant_face_spike.py`（`detect_faces`）。

## 校正（calibration）用的圖

- **不要用 SyRIP**：它只能內部評估，不能拿來產生要提交的模型。
- 可行性測試先用公開圖（例如 COCO）就好；正式版等娃娃照片拍好再換。

## 步驟

1. 到 Hailo Developer Zone 註冊，下載**支援 Hailo-10H 的 Dataflow Compiler**（要 Linux，或 Windows 上的 WSL2）。
2. 每個模型依序做 parse → optimize（量化）→ compile，記下每一步成功與否和錯誤訊息。先從最單純的 YuNet 和 RTMPose 開始，流程通了再挑戰 RTMO。
3. 編譯成功的模型，用編譯器的模擬器（emulator）跑幾張圖，跟 ONNX 的結果比（同一張圖，框和關鍵點位置差多少）。需要 ONNX 的標準輸出就跟 Harry 拿。
4. 回報給 Harry：哪些模型轉成功、卡在哪一步（附錯誤訊息）、量化後差多少、編譯器估的 FPS。

## 9/30 說明會可以問 Hailo

- Hailo-10H 支不支援 RTMO、RTMPose 這類姿態模型？Model Zoo 有沒有授權寬鬆（不是 AGPL）的姿態模型或人臉模型？
- 輸出包含解碼、NMS 的 ONNX，建議怎麼切？
- 量化後準確度掉太多時，有沒有不需要標註資料的補救方法？
