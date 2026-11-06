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
- **主力程式是 `pipeline/infant_face_v3.py`**：RTMO 找出所有人 → 頭部依身體方向轉正 → YuNet 找臉 → 判斷每個嬰兒的口鼻看不看得到。每床的判定在 `pipeline/crib_detect.py`（床位模式），燈號狀態機在 `pipeline/crib_state.py`（3 秒警示、照護中、空床）。`experiments/infant_face_spike.py`（v2）和 `experiments/infant_prone_spike.py`（v1）留著對照，改版時不要直接覆蓋。
- **`pipeline/infant_face_v4.py`（v3＋口鼻膚色檢查）、`pipeline/infant_face_v5.py`（v4＋臉三級 ok/warn/alert）都是實驗版，還沒採用**：v4 擋「只露眼睛」的假臉（2026/9/23 改良口鼻膚色後 SyRIP 誤報 16.6%→10.9%）；v5 加側臉 warn（娃娃影片好，但真實嬰兒召回/誤報仍輸 v3）。**v3 仍是真實嬰兒基準。** 細節見 `docs/architecture.md`、`docs/影片測試結果.md` §3d、`docs/娃娃測試結果.md` §5。
- **YOLOv8 是選配**（AGPL，只有 v1、v2 和評估比較會用到）：要用時 `uv sync --group yolo`。平常 `uv sync` 不會裝它，也不需要 PyTorch。
- **這些不進 repo**：`dataset/`、`outputs/`、模型權重（YuNet 除外）、`docs/private/`（含 context/decisions/status）、`docs/archive/`（給隊友的舊任務單、托嬰時期文件）、`docs/*.docx` 同意書。`.gitignore` 已設定好。**repo 從 2026/10/28 起公開**：commit 進去的東西任何人都看得到。
- **SyRIP 只能內部評估**：它的圖、以及由它產生的輸出圖，不可放進簡報、影片或 repo。
- **網路上找到的影片／圖片也只能內部評估**：標註放不進 repo 的 `labels/local_video_slots.csv`／`labels/local_video_labels.csv`（格式同 `labels/doll_video_*.csv`，`evaluation/video_labels.py` 會一起讀），名稱不寫進程式和公開文件。
- **評估流程**（改了 `pipeline/` 就三套都要重跑；指令前面都是 `uv run python -m evaluation.`）：
  - SyRIP（測誤報）：`eval_syrip --face v3` 跑分 → `score_relabel` 用人工標註重算真實的誤報率。注意用 `--limit` 試跑會覆蓋 `outputs/syrip_eval/` 裡的 CSV。
  - 娃娃影片（測床位模式和警示狀態機）：`eval_video`，床位在 `labels/doll_video_slots.csv`、時間軸在 `labels/doll_video_labels.csv`（只列手不在畫面裡的片段；`label` 另有 `empty` 空床、`staged` 另有 `prone` 趴睡但口鼻看得到，都不算進誤報）。結果見 `docs/影片測試結果.md`。
  - 每床鏡頭的床位情境（照護中暫停計時、空床灰燈，原型）：`crib_context`，讀 `eval_video` 存的每幀判定，空床參考在床位檔的 `empty_ref` 欄。結果見 `docs/影片測試結果.md` §3h。
  - VLM 複判（實驗，另一個虛擬環境 `.venv-vlm`）：`.venv-vlm/Scripts/python -m experiments.vlm_spike`，VLM 看床位檔 `head_x0..head_y1` 的床頭範圍。
  - 口鼻分類器（實驗，11/06）：`uv run python -m experiments.airway_crops` 存頭部特寫（先跑 `eval_video`）→ `.venv-vlm/Scripts/python -m experiments.airway_train` 訓練＋評估。**只能用 `hailo/prep_calib.py` 的 `SPLITS` 裡 calib 那一半訓練**；hold 只評估，門檻 0.5 不在 hold 上調。結果和失敗模式（側躺側臉判成看不到）見 `docs/architecture.md`「口鼻分類器」。
  - 娃娃擺拍（測危險情境召回）：`eval_doll`，標註在 `labels/doll_labels.csv`。**計分單位是「每隻娃娃」**，不是每張照片——同框兩隻都危險、只抓到一隻時，用整張圖算會被判對。結果和失敗模式見 `docs/娃娃測試結果.md`。
  - 灰階／暗光（模擬 IR 夜視，選跑）：三支都有 `--gray`、`--dim 0.3`（`evaluation/degrade.py`），結果另外放 `_<條件>` 的檔名／資料夾、不蓋掉原圖基準；SyRIP 用 `score_relabel --eval-dir outputs/syrip_eval_<條件>` 算。**灰階下口鼻膚色檢查（v4/v5）失效**，結果見 `docs/娃娃測試結果.md` §8。
- **可以訓練模型（2026/11/05 起）**：原本為了省事只用現成、授權寬鬆的模型加上演算法設計；表現會更好就訓練或微調（例如 RTMO 量化掉太多時）。訓練資料的授權規定見 `docs/private/decisions.md`「技術與產品政策」：要允許商用，SyRIP、網路影像、評估用的資料都不能拿來訓練。

## 範圍護欄

UGen300 11/02 起寄送，實機工作（HEF 轉換、HailoRT、晶片上的速度／功耗）現在都可以做：計畫在 `hailo/轉換計畫.md`，進度與停損點在 `docs/private/status.md`「B 部署計畫」，每一步的指令、錯誤訊息和數字寫進 `hailo/結果.md`。**11/06 現況**：UGen300 裝好（HailoRT 5.4.0，步驟和陷阱見 `hailo/安裝筆記.md`）。RTMO 定案用 D1（量化感知訓練、全部 8 位元），設 `FG_RTMO_HEF=model/rtmo_D1.hef`，主流程和三套評估的 RTMO 就改在晶片上跑（`hailo/hailo_infer.py`）。YuNet 也接上了（設 `FG_YUNET_HEF=model/yunet_A.hef`，`hailo/yunet_hef.py`），但準確度差一點、單床反而變慢、會讓 USB 偶爾斷線，要不要放晶片還沒決定（速度和斷線見 `hailo/結果.md`「主流程的速度」「裝置斷線」）。VLM（Qwen2-VL-2B，Hailo 官方編好的 HEF，放 `model/genai/`）也在晶片上跑起來了（`hailo/vlm_hef.py`，每題約 1 秒），但輸入固定 336×336、API 不給機率，床頭裁切在晶片上幾乎都回答「看不到」，改看頭部特寫（`vlm_spike --crop head`）後正臉好了、側臉仍全部說看不到，還不能用；它跑的時候 pyHailoRT 不放開 GIL，Windows＋USB 又不能放另一個程序，每問一題主流程停約 1 秒（見 `hailo/結果.md`「VLM 複判也放上晶片」）。所以 11/06 改試訓練過的口鼻分類器（還在筆電上，見「評估流程」）。pyHailoRT 不在 `pyproject.toml` 裡，`uv sync` 會把它移除，要重裝。晶片上沒實測過的數字（速度、路數、功耗）不能寫成已達成；目前量過單一模型的速度和主流程的吞吐量（`evaluation.bench_multistream --threads`：筆電主機、720p、用同一支影片模擬多床、不含解串流；RTMO 在晶片約 30 幀/秒、兩個都在晶片約 15），功耗還沒量。
