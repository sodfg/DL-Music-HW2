# DL Music HW2 · 鋼琴音樂生成

使用 POP909 從零訓練 Transformer，生成 **32 小節鋼琴 MIDI**，並比較不同生成設定。

- **Task A**：不提供旋律提示，從頭生成 32 小節。
- **Task B**：保留助教的 8 小節提示，續寫 24 小節。
- **評估**：H4、GS；另提供 SI、模型深度比較等擴充實驗工具。

## 快速開始

需要 Python 3.10 或 3.12。已在本機 CPU（PyTorch 2.5.1）與學校 A5000（PyTorch 2.6.0）執行。
GPU 訓練環境需有相容的 NVIDIA 驅動與 CUDA 版 PyTorch。

```bash
git clone https://github.com/sodfg/DL-Music-HW2.git
cd DL-Music-HW2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r baseline/requirements.txt
```

若在 EDA Union 使用既有共用套件，請改看[遠端環境說明](baseline/README.md#eda-union)，不必重建環境或重裝 PyTorch。
以下指令都從專案根目錄執行。

### 已有模型：直接生成

模型與資料集不包含在 Git 儲存庫中。先取得本專案相容的 `best.pt`，放在
`runs/baseline/best.pt`；目前尚未提供公開的模型下載連結。
Task A 推論不需要 POP909 訓練資料，checkpoint 已包含模型設定與 token 字典。

```bash
python baseline/generate.py \
  --checkpoint runs/baseline/best.pt \
  --output runs/my_samples \
  --count 2 --bars 32
```

結果為 `runs/my_samples/*.mid`，使用 MIDI 播放器試聽。
程式預設使用可用的 GPU，沒有 CUDA 時使用 CPU；也可明確加 `--device cpu`。

### 沒有模型：自行訓練

使用**課程提供的資料切分**，將解壓後的 `POP909/` 放在根目錄：

```text
POP909/
  training/001/001.mid       # 共 896 首，各有 beat_midi.txt / chord_midi.txt
  prompts/basic-b/001/       # 3 首續寫提示
```

```bash
python baseline/prepare.py
python baseline/train.py --steps 20000 --validation-fraction 0 --output runs/baseline
```

第一行輸出訓練 tokens 與 `roundtrip.mid`；請先檢查還原效果。
第二行使用全部訓練資料，儲存 `best.pt` 與 `last.pt`，之後即可執行上面的生成指令。
訓練需要時間與適當硬體；GPU 選擇、tmux、續訓及快速試跑見詳細指南。

## 接下來做什麼？

| 需求 | 操作說明 |
|---|---|
| 續寫提示、跑三組設定、計算 H4／GS、產生總表 | [完整操作指南](baseline/README.md) |
| MIDI 轉鋼琴 WAV、SI 小量／批次實驗 | [Optional 擴充實驗流程](baseline/docs/OPTIONAL_WORKFLOW.md) |
| 比較 3 層與 6 層模型 | [模型深度實驗](baseline/docs/DEPTH_EXPERIMENT.md) |
| 找程式、理解目錄與 smoke test | [資料夾導覽](baseline/docs/PROJECT_LAYOUT.md) |

## 檔案存在哪裡？

| 目錄 | 內容 | Git |
|---|---|---|
| `baseline/` | 程式、測試與文件 | 追蹤 |
| `HW2_scripts/` | 助教提供的參考工具 | 追蹤原始工具 |
| `POP909/`、`data/` | 原始資料、前處理資料 | 忽略，另行取得 |
| `runs/` | 模型、生成 MIDI／WAV、評估結果 | 忽略，另行備份 |

目前使用自訂的小節／位置事件表示法。H4／GS 評估採共同 64 格小節位置，
分數不能和其他前處理規則的結果直接混用。SI pilot 與正式批次結果也分開保存。

本專案參考課程的 tokenization／evaluation 範例；音樂指標使用
[MusDr](https://github.com/slSeanWU/MusDr)，音訊合成使用
[FluidSynth](https://www.fluidsynth.org/)。
