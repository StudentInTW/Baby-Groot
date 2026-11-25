# CLAUDE.md

給在這個 repo 工作的 Claude（和新接手的人）。

## 這是什麼

**2026/10/8 起方向轉向（Harry 提案，全隊 10/9 確認）**：產後護理之家（月子中心）嬰兒室的即時安全輔助——「讓護理師比家長先知道」。一台邊緣裝置（ASUS UGen300／Hailo-10H）判讀嬰兒室既有的每床直播鏡頭，寶寶臉朝下、口鼻被蓋住時先安靜通知護理師，AI 判讀在本地、不另存影像。參加 ASUS UGen AI League（Battlefield Lightning 賽道）。

原題目是「托嬰中心午睡安全監看」（同一台裝置看整間午睡室的多名嬰兒）；10/5–10/8 的訪談與參訪發現托嬰中心的監視器在物理上看不到臉、孩子多半 7–8 個月以上，所以轉向。**轉向的理由與淘汰的方向見 `docs/private/decisions.md`「方向」。** 程式與評估不受影響：每支鏡頭只拍一張床，等於現有的床位模式。

**2026/11/02 入圍決賽**（作品 FirstGlance，隊名 Baby Groot）：11/06–12/04 Stage II 開發，**12/10** 交英文決賽簡報，**12/19** 決賽（ASUS 總部立功大樓：舞台提案、問答、展演）。決賽技術審核會抽驗 GitHub 的程式邏輯、README 完整度和硬體整合細節。

## 開場先讀

1. `docs/private/context.md`——薄的入口，指向其他文件。**`docs/private/` 只在 Harry 的電腦上，不在 GitHub**；沒有的話就讀 `docs/architecture.md`、`README.md` 和 `docs/` 裡其他文件。
   - 從那裡分流：`docs/private/status.md`（Stage II 進度、待辦、停損點）、`docs/private/decisions.md`（已定案與理由）、`docs/architecture.md`（公開的技術架構與 v1→v5 演進）。Stage I 的紀錄在 `docs/private/archive/stageI/`，查歷史時才讀。
2. `README.md`——英文，給評審：怎麼跑、資料夾結構、硬體整合、目前結果。

## 慣例

- **回覆用繁體中文。**
- **程式結構（2026/10/26 整理）**：主流程在 `pipeline/`、評估在 `evaluation/`、示範在 `demo/`（介面原型在 `demo/ui/`）、舊版與實驗在 `experiments/`、標註在 `labels/`、HEF 轉換在 `hailo/`。一律在 repo 根目錄用 `uv run python -m 資料夾.檔名` 執行（例如 `uv run python -m evaluation.eval_video`）。決賽技術審核會抽驗程式邏輯、README 完整度和硬體整合細節，結構或指令改了 README 要跟著改。
- **主流程是 v6（2026/11/24 Harry 定）＝`pipeline/infant_face_v6.py`**：v3（RTMO 找出所有人 → 頭部依身體方向轉正 → YuNet 找臉，`pipeline/infant_face_v3.py`）再加口鼻小塊模型——每張臉依 YuNet 的 5 個特徵點切鼻子到下巴 96×96，三顆 ResNet18 平均判口鼻露出，床位裡任一張 ≥ 0.5 才算看得到（門檻沒調、不能在 hold 上調）。模型 `model/mouth_patch_ens3.onnx`（134 MB，不進 repo；預設 onnxruntime 在主機 CPU 跑、1 條執行緒，`MOUTH_ORT_THREADS`）。也編成了 HEF：設 `FG_MOUTH_HEF=model/mouth_A.hef` 就在晶片上跑（`hailo/mouth_hef.py`）。評估的輸出會另外放 `*_chip-mouth_A`，結果見 `hailo/結果.md`「口鼻小塊模型」。每床的判定在 `pipeline/crib_detect.py`（床位模式；主流程看 `MAIN_MODE`＝`v6_slot`，沒有模型檔或舊的每幀判定時退回 `v3_slot`），燈號狀態機在 `pipeline/crib_state.py`（3 秒警示、照護中、空床、離線）。`demo/live_view.py`、`experiments/rtsp_check.py` 預設 v6（`--mode` 可換對照）。結果和限制見 `docs/architecture.md`「口鼻小塊模型」。`experiments/infant_face_spike.py`（v2）和 `experiments/infant_prone_spike.py`（v1）留著對照，改版時不要直接覆蓋。
- **`pipeline/infant_face_v4.py`（v3＋口鼻膚色檢查）、`pipeline/infant_face_v5.py`（v4＋臉三級 ok/warn/alert）留作對照**：v4 擋「只露眼睛」的假臉（SyRIP 誤報 10.9%），但雙眼距離 < 22 px 的臉不檢查，每床鏡頭上的側躺等於 v3；v5 加側臉 warn（娃娃影片好，但真實嬰兒召回/誤報仍輸 v3）。細節見 `docs/architecture.md`、`docs/影片測試結果.md` §3d、`docs/娃娃測試結果.md` §5。
- **YOLOv8 是選配**（AGPL，只有 v1、v2 和評估比較會用到）：要用時 `uv sync --group yolo`。平常 `uv sync` 不會裝它，也不需要 PyTorch。
- **這些不進 repo**：`dataset/`、`outputs/`、模型權重（YuNet 除外）、`docs/private/`（含 context/decisions/status）、`docs/archive/`（給隊友的舊任務單、托嬰時期文件）、`docs/*.docx` 同意書。`.gitignore` 已設定好。**repo 從 2026/10/28 起公開**：commit 進去的東西任何人都看得到。
- **SyRIP 只能內部評估**：它的圖、以及由它產生的輸出圖，不可放進簡報、影片或 repo。
- **網路上找到的影片／圖片也只能內部評估**：標註放不進 repo 的 `labels/local_video_slots.csv`／`labels/local_video_labels.csv`（格式同 `labels/doll_video_*.csv`，`evaluation/video_labels.py` 會一起讀），名稱不寫進程式和公開文件。
- **評估流程**（改了 `pipeline/` 就三套都要重跑；指令前面都是 `uv run python -m evaluation.`）：
  - SyRIP（測誤報）：`eval_syrip --face v6 --models rtmo`（對照 `--face v3`、`v4`）跑分 → `score_relabel` 用人工標註重算全部 700 張的誤報率 → **主要數字看 `syrip_nursery v3 v4 v6`**（像嬰兒室的仰躺 81 張：沒有大人的手、單獨躺著、臉夠大；拿掉奶嘴玩具，手擋口鼻算危險；清單在 `outputs/syrip_subsets/`，不進 repo）。全部睡姿的誤報 v6 26.3%，多半是坐爬站的大嬰兒。注意用 `--limit` 試跑會覆蓋 `outputs/syrip_eval/` 裡的 CSV。
  - 娃娃影片（測床位模式和警示狀態機）：`eval_video`（有模型檔就自動多比 v6＋床位，`--no-v6` 不比；calib 那一半的影片是 v6 的訓練資料，合計後面另外印「只算 v6 沒訓練過的影片」，v6 的數字看那一行），床位在 `labels/doll_video_slots.csv`、時間軸在 `labels/doll_video_labels.csv`（只列手不在畫面裡的片段；`label` 另有 `empty` 空床、`staged` 另有 `prone` 趴睡但口鼻看得到，都不算進誤報；`label` alert 加 `staged` unseen 是口鼻沒被蓋、但鏡頭看不到（例如側躺背對鏡頭），不算危險也不算誤報，另外列，2026/11/14 起）。結果見 `docs/影片測試結果.md`。不給影片名稱會連台南 11/08 那 6 支一起跑；那批單獨用 `--fps 1` 計分（§3i）。
  - 每床鏡頭的床位情境（照護中暫停計時、空床灰燈，原型）：`crib_context`，讀 `eval_video` 存的每幀判定，空床參考在床位檔的 `empty_ref` 欄。結果見 `docs/影片測試結果.md` §3h。
  - VLM 複判（實驗，另一個虛擬環境 `.venv-vlm`）：`.venv-vlm/Scripts/python -m experiments.vlm_spike`，VLM 看床位檔 `head_x0..head_y1` 的床頭範圍。
  - 口鼻小塊模型（v6 的模型，11/23 起）：`experiments.mouth_patch_crops` 挑幀＋存頭部特寫（布料）→ `experiments.mouth_patch_build` 照主流程切口鼻小塊 → `.venv-vlm/Scripts/python -m experiments.mouth_patch_train`（三顆種子）→ `experiments.mouth_patch_export` 合成 ONNX；指令在各檔開頭。**只能用 `hailo/prep_calib.py` 的 `SPLITS` 裡 calib 那一半訓練**；hold、場域只評估，門檻 0.5 不在 hold 上調。中間檔在 `outputs/airway3/`；11/19 側睡角度序列的標註在 `labels/train_20261119.csv`（只訓練，不算進 `eval_video`）。舊的頭部特寫分類器（`experiments.airway_crops`、`airway_train`，11/06）沒採用，見 `docs/architecture.md`「口鼻分類器」。
  - 娃娃擺拍（測危險情境召回）：`eval_doll`（v6 加 `--face v6`；第 2–4 批是 v6 的訓練資料，v6 只看第 1 批），標註在 `labels/doll_labels.csv`。**計分單位是「每隻娃娃」**，不是每張照片——同框兩隻都危險、只抓到一隻時，用整張圖算會被判對。結果和失敗模式見 `docs/娃娃測試結果.md`。基準是第 1–3 批（`--batch 1,2,3`）；第 4 批（台南 11/08）另外用 `--batch 4` 計分。
  - 灰階／暗光（模擬 IR 夜視，選跑）：三支都有 `--gray`、`--dim 0.3`（`evaluation/degrade.py`），結果另外放 `_<條件>` 的檔名／資料夾、不蓋掉原圖基準；SyRIP 用 `score_relabel --eval-dir outputs/syrip_eval_<條件>` 算。**灰階下口鼻膚色檢查（v4/v5）失效**，結果見 `docs/娃娃測試結果.md` §8。
  - 框床 RTMO（實驗，`eval_video --slot-rtmo`）：RTMO 只看各床位的裁切。床只佔畫面一角的廣角鏡頭幫助大；每床鏡頭沒好處，兩床緊鄰時多出側臉誤報。結果見 `docs/影片測試結果.md` §3j。
  - 模擬監視器畫質（實驗，`evaluation/cctv_sim.py`）：把娃娃影片縮到托嬰廣角鏡頭的臉寬（32 px）、擺進 1080p 畫面、調色調和銳化、H.264 壓縮，再 `eval_video --cctv daycare`。ffmpeg 用 `uv run --with imageio-ffmpeg` 臨時掛（不進 pyproject，免得 `uv sync` 動到 pyHailoRT）。**只能當壓力測試和訓練用的增強資料，不能代替真實鏡頭評估**：做不出場域「被子蓋到身體、臉露出」判不到的失敗。結果見 §3k。
- **可以訓練模型（2026/11/05 起）**：原本為了省事只用現成、授權寬鬆的模型加上演算法設計；表現會更好就訓練或微調（例如 RTMO 量化掉太多時）。訓練資料的授權規定見 `docs/private/decisions.md`「技術與產品政策」：要允許商用，SyRIP、網路影像、評估用的資料都不能拿來訓練。

## 範圍護欄

UGen300 11/02 起寄送，實機工作（HEF 轉換、HailoRT、晶片上的速度／功耗）現在都可以做：計畫在 `hailo/轉換計畫.md`，進度與停損點在 `docs/private/status.md`「B 部署計畫」，每一步的指令、錯誤訊息和數字寫進 `hailo/結果.md`。**11/06 現況**：UGen300 裝好（HailoRT 5.4.0，步驟和陷阱見 `hailo/安裝筆記.md`）。RTMO 定案用 D1（量化感知訓練、全部 8 位元），設 `FG_RTMO_HEF=model/rtmo_D1.hef`，主流程和三套評估的 RTMO 就改在晶片上跑（`hailo/hailo_infer.py`）。YuNet 也接上了（設 `FG_YUNET_HEF=model/yunet_A.hef`，`hailo/yunet_hef.py`），但準確度差一點、單床反而變慢、會讓 USB 偶爾斷線，要不要放晶片還沒決定（速度和斷線見 `hailo/結果.md`「主流程的速度」「裝置斷線」）。VLM（Qwen2-VL-2B，Hailo 官方編好的 HEF，放 `model/genai/`）也在晶片上跑起來了（`hailo/vlm_hef.py`，每題約 1 秒），但輸入固定 336×336、API 不給機率，床頭裁切在晶片上幾乎都回答「看不到」，改看頭部特寫（`vlm_spike --crop head`）後正臉好了、側臉仍全部說看不到，還不能用；它跑的時候 pyHailoRT 不放開 GIL，Windows＋USB 又不能放另一個程序，每問一題主流程停約 1 秒（見 `hailo/結果.md`「VLM 複判也放上晶片」）。所以 11/06 改試訓練過的口鼻分類器，11/24 定案成 v6 的口鼻小塊模型（見「評估流程」）。同一天也編成 HEF（`model/mouth_A.hef`）：
- 娃娃上跟 CPU 一樣，SyRIP 像嬰兒室的 81 張誤報 9 → 11。
- 16 床時晶片版不拖慢吞吐量；CPU 版原本從 27.6 掉到 19.9 幀/秒（onnxruntime 預設開滿核心、各床互搶）。11/25 起主流程改成 1 條執行緒（輸出完全一樣，三套重跑逐列相同），同一段時間量 CPU 版 29.6、晶片版 31.2（不跑口鼻模型 31.2；筆電插電）。
- 兩顆都在晶片（子程序、8 床、一次一張）跑了 8 小時，0 次斷線（11/24 晚上）。放晶片還是 CPU 還沒定。
- 編譯指令和陷阱在 `hailo/結果.md`、`hailo/安裝筆記.md`：DFC log 的「Mixed Precision」不代表被壓成 4 位元；DFC 的微調反而更差。pyHailoRT 不在 `pyproject.toml` 裡，`uv sync` 會把它移除，要重裝。晶片上沒實測過的數字（速度、路數、功耗）不能寫成已達成；目前量過單一模型的速度和主流程的吞吐量（`evaluation.bench_multistream --threads`：筆電主機、720p、用同一支影片模擬多床、不含解串流；RTMO 在晶片約 30 幀/秒、兩個都在晶片約 15），功耗還沒量。
