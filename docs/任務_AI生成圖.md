# 任務：用 Stable Diffusion 生成測試圖

**給**：嘉晏哥哥　**預估時間**：半天內（選做，量不用多）　**交回**：分好資料夾的圖片壓成 zip，加一行「用了哪個模型」

## 這些圖拿來做什麼

測試我們的系統：畫面裡嬰兒的口鼻看得到時要判「正常」，看不到時要發「警示」。
**不是拿來訓練模型**，所以不用標框、不用標點——**放對資料夾就是標註**。

## 步驟

1. **選模型，先確認授權**：選授權**明確允許商用**的模型，例如 SDXL 1.0 base。其他模型（包括 Civitai 上的社群模型）不確定授權的話，先問 Harry。
   理由：比賽規定作品用到的素材都要有合法授權，而且要能提出證明。
2. **照下表生成**：6 種情境，每種 **10–20 張**（總共約 60–120 張）。每種情境多換幾個 seed，讓嬰兒、床單、光線都不一樣。
3. **一張一張看，照「圖裡實際的樣子」分資料夾**（這一步最重要）：
   - 放哪個資料夾，看的是**圖裡嬰兒的口鼻實際看不看得到**，不是提示詞寫了什麼。例如提示詞寫「被子蓋住口鼻」，但生出來口鼻還露在外面，就放到 `ok_side` 或刪掉。
   - 刪掉：嬰兒變形（多手多腳、臉糊掉）、不像照片、看不出是嬰兒的。
4. **保留生成參數**：直接存 PNG（WebUI 和 ComfyUI 會把提示詞、seed、模型寫進 PNG 裡），不要轉成 JPG。
5. **交回**：整個資料夾壓成 zip，附一行字寫用了哪個模型（名稱＋版本）。

## 6 種情境與資料夾

| 資料夾 | 情境 | 提示詞裡的 `[情境]` |
|---|---|---|
| `ok_supine` | 仰睡、臉朝上 | lying on its back, face up |
| `ok_side` | 側睡或頭轉向一邊，口鼻看得到 | lying on its side, face turned toward the camera |
| `alert_facedown` | 趴睡、臉朝下埋進床墊 | lying on its stomach, face pressed down into the mattress |
| `alert_blanket` | 被子蓋住口鼻或整張臉 | a blanket covering its nose and mouth |
| `alert_cover` | 口鼻被絨毛玩偶擋住 | a teddy bear lying over its face |
| `warn_prone` | 趴睡、頭轉向一邊，口鼻看得到 | lying on its stomach, head turned to one side, face visible |

## 提示詞範本

**正向**（把 `[情境]` 換成上表最右欄）：

> A realistic photo from a ceiling-mounted security camera in a daycare nap room, looking down at a 6-month-old baby asleep on a crib mattress with eyes closed, [情境], soft dim lighting, plain crib sheet and a small blanket

**反向（negative prompt）**：

> cartoon, illustration, painting, 3d render, deformed, extra limbs, extra fingers, blurry face, text, watermark

- SDXL 建議解析度 1024×1024。
- 每種情境先試生 4 張，看起來像再大量生。
- 生不出來或被工具擋掉的情境就跳過，交給娃娃拍。

## 加分（有空再做）

- **用 ControlNet**（depth 或 OpenPose），拿隊上拍的娃娃照片當構圖參考，生成姿勢一樣、但外觀像真嬰兒的圖。危險姿勢用這個方法比較容易生得出來。
- **多個嬰兒同框**（2–3 張嬰兒床並排）：放 `multi` 資料夾。每張最多 1 個嬰兒在危險狀態，檔名寫出危險的是左、中、右哪一個，例如 `danger_left_01.png`；全部都安全就寫 `safe_01.png`。

## 注意

- 不要用網路上的真實嬰兒照片當 img2img 或 ControlNet 的輸入。
- 這些圖只供本隊內部使用；要放進簡報時，標示「AI 生成」。
