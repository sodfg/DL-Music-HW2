# HW2 first baseline

先完成 Basic A/B。這是可執行的第一版，不是已訓練好的模型或完成的作業成果。
以下指令皆從 `Hw2-DL-in-music` 根目錄執行。助教的 `HW2_scripts` 保持原樣。

## 程式與資料

- `music.py`：MIDI ↔ tokens，採用 `beat_midi.txt` 的拍點切小節。
- `prepare.py`：處理全部 896 首，嚴格檢查字典與輸入，輸出 JSON 和 roundtrip MIDI。
- `model.py`：隨機初始化、具有 causal attention 的 decoder-only Transformer。
- `train.py`：訓練、驗證、存檔及續訓。
- `generate.py`：固定小節數生成、三種設定比較、8+24 小節續寫。
- `evaluate.py`：呼叫助教提供的 H1/H4/GS 計算程式。
- `render_audio.py`：FluidSynth 合成 WAV。

表示法沿用助教提供的 Bar/Pos/Pitch/Vel/Dur 概念，但加上 `Beats_N`：
每小節標註拍點數。它不是完整的 MIDI 拍號，不推定分母。採樣生成預設每小節 4 拍。
一拍分成 16 格，音符長度最多 256 格，力度量化成 32 種，音高保留 0–127。
位置字典容納每小節最多 32 拍，遇到更多拍就報錯；不直接判定異常標註是正確拍號。
每小節速度由其拍點間隔中位數估計，量化到 5 BPM（30–210）。這有時間近似誤差，
請試聽 roundtrip；來源速度變化細節、踏板及三軌分工不會保留。首尾未完整小節
保留成標註資料中的部分小節。空小節保留。

這份資料與助教原始 JSON 的字典不同，不能混用；模型 checkpoint 內含對應字典。

## 本機準備與驗證

```bash
source .venv/bin/activate
python -m pip install -r baseline/requirements.txt
python baseline/prepare.py
```

輸出在 `data/baseline/`。先聽 `roundtrip.mid`，也可還原其他歌：

```bash
python baseline/prepare.py --roundtrip-song 149
```

小模型只用來確認流程，不會產生有品質的音樂：

```bash
python baseline/train.py --output runs/smoke --device cpu --steps 2 --context 32 --width 32 --layers 1 --heads 4 --batch-size 2 --eval-every 2 --eval-batches 1
python baseline/generate.py --checkpoint runs/smoke/best.pt --output runs/smoke/generated --device cpu --count 1 --bars 2 --max-events-per-bar 4
```

## EDA Union

先在本機傳送專案（不複製本機 .venv、Git 和生成結果）：

```bash
rsync -av --exclude='.venv/' --exclude='.git/' --exclude='runs/' --exclude='data/' --exclude='.tools/' ./ eda_union_sodfg:~/hw2_music/
ssh eda_union_sodfg
cd ~/hw2_music
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r baseline/requirements.txt
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python baseline/prepare.py
nvidia-smi
```

檢查哪張卡空閒，依學校的共享 GPU 使用規則選擇。先測試 20 步：

```bash
CUDA_VISIBLE_DEVICES=0 python baseline/train.py --steps 20 --eval-every 20 --eval-batches 2 --output runs/gpu_smoke
```

正式訓練建議放在 tmux，避免網路中斷使程序停止：

```bash
tmux new -s hw2
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python baseline/train.py --steps 20000 --validation-fraction 0 --output runs/baseline > runs_train.log 2>&1
```

按 Ctrl+B，放開後按 D，離開 tmux 而不停止訓練。重新登入後：

```bash
tmux attach -t hw2
tail -f runs_train.log
```

若 tmux 不存在，請依學校方式取得或安裝，避免用 sudo 修改共享主機。
6 層、512 維、8 heads、context 1024、batch 8 是起始設定，並未在 A5000 實測。
若 OOM，先把 `--batch-size` 改成 4 或 2。20,000 是起始步數，不保證音樂品質。
開發階段預設留約 5% 的完整歌曲作驗證，不混入助教 prompts。正式訓練指令使用
`--validation-fraction 0`，全部 896 首參與梯度更新，符合使用完整訓練集的要求。
這時 loss 記為 `training_probe_loss`，只表示訓練資料的表現，不能當成泛化驗證結果。
記錄位於 `loss.jsonl` / `split.json`；best.pt 依對應 probe loss 選擇。

續訓，模型參數、字典與 seed 必須與原 run 相同：

```bash
CUDA_VISIBLE_DEVICES=0 python baseline/train.py --steps 20000 --validation-fraction 0 --output runs/baseline --resume runs/baseline/last.pt
```

同一總步數下可恢復 RNG/optimizer；增加總步數會改變 cosine learning rate 排程。
只載入自己產生的 checkpoint（含 optimizer 狀態）。

## Basic A：三組設定，各 20 首、32 小節

```bash
python baseline/generate.py --checkpoint runs/baseline/best.pt --output runs/a_t08 --temperature 0.8 --top-k 32 --top-p 0.95
python baseline/generate.py --checkpoint runs/baseline/best.pt --output runs/a_t10 --temperature 1.0 --top-k 32 --top-p 0.95
python baseline/generate.py --checkpoint runs/baseline/best.pt --output runs/a_t12 --temperature 1.2 --top-k 32 --top-p 0.95
```

固定相同 seed，便於比較。採樣會限制 token 順序與位置單調性，每小節至少一個音符，
最多 128 個音符，並精確以 Bar/EOS 結束。這些規則必須寫入報告。
context 滑動窗口不使用 KV cache，生成可能比較慢；不是整首歌曲長度的 context。

## Basic B：每首 prompt 三組設定

以下對 001 執行；把路徑換成 002、003，總共輸出 9 首：

```bash
python baseline/generate.py --checkpoint runs/baseline/best.pt --prompt POP909/prompts/basic-b/001/001.mid --prompt-beats POP909/prompts/basic-b/001/beat_midi.txt --prompt-bars 8 --bars 32 --count 1 --temperature 0.8 --output runs/b_001_t08
python baseline/generate.py --checkpoint runs/baseline/best.pt --prompt POP909/prompts/basic-b/001/001.mid --prompt-beats POP909/prompts/basic-b/001/beat_midi.txt --prompt-bars 8 --bars 32 --count 1 --temperature 1.0 --output runs/b_001_t10
python baseline/generate.py --checkpoint runs/baseline/best.pt --prompt POP909/prompts/basic-b/001/001.mid --prompt-beats POP909/prompts/basic-b/001/beat_midi.txt --prompt-bars 8 --bars 32 --count 1 --temperature 1.2 --output runs/b_001_t12
```

模型條件是量化後的 prompt tokens；MIDI 輸出保留 prompt 原始音符的音高、力度與
秒數（越過八小節邊界的延音裁切），生成尾段移至原始 prompt 的結束時間。
輸出合併成一軌，保留邊界前的原始 MIDI tempo、控制事件與通道資訊；
續寫以 MIDI ticks 接到原始提示的精確結束時間，使用生成的速度。
跨越邊界的原曲音符在邊界終止，並釋放 channel 0 踏板，避免懸掛音符。
每首選最佳設定，MIDI、WAV 和報告中都清楚標註。

## 客觀評估與 WAV

先安裝助教指定 MusDr（官方來源 https://github.com/slSeanWU/MusDr）：

```bash
git clone https://github.com/slSeanWU/MusDr.git HW2_scripts/MusDr
python baseline/evaluate.py --data data/baseline --output runs/metrics_real.csv
python baseline/evaluate.py --data runs/a_t08 --output runs/metrics_t08.csv
python baseline/evaluate.py --data runs/a_t10 --output runs/metrics_t10.csv
python baseline/evaluate.py --data runs/a_t12 --output runs/metrics_t12.csv
```

MusDr 可能需要額外依賴，請依其 README 安裝。這部分尚未在本機驗證。
評估器把每個小節位置映射到共同的 64 格相對位置，再呼叫助教程式。
原因是完整字典預留 512 個位置，直接傳入會使 GS 分母過大、人為提高分數。
真實和生成資料都使用相同映射；Beats token 不傳入評估。
這是在小節相對時間上的節奏比較，與直接用助教原始 128 格字典得到的分數不同，
不要混用舊分數。H4 仍以四小節為窗口。真實資料含不同小節拍點數，生成固定四拍，
報告應交代這個比較限制及 64 格正規化流程。
SI 是選做，目前未實作。

WAV 需要 FluidSynth 執行檔與有授權的 `.sf2` SoundFont：

```bash
python baseline/render_audio.py --input runs/a_t08 --output runs/a_t08/wav --soundfont /path/to/piano.sf2
```

把最佳 Basic A 的 20 首與全部 Basic B 的 9 首都轉為 WAV。FluidSynth/SoundFont
目前未自動下載或在共享主機安裝，音訊合成需另行驗證。

## 繳交與版本控制

commit baseline 程式、README 和 requirements；data、runs、POP909 不提交。
模型提交資料夾請保留 best.pt、tok2idx.json、config.json、requirements、全部程式及
實際 inference 指令。不要上傳訓練或前處理資料。此版不包含 Bonus 伴奏與正式報告。
來源：課程 sample tokenization/evaluation code；PyTorch SDPA 與 cross entropy API。

## 已完成的本機檢查

CPU 小模型訓練與 checkpoint、32 小節生成、8+24 續寫可運行。
正式 A5000 訓練、MusDr 評估、FluidSynth WAV 尚待 EDA Union 環境驗證。
Smoke 結果來自只訓練幾步的模型，僅驗證程式，不代表生成品質。

也可以一次執行三組 Basic A 與全部九首 Basic B（預設共 69 首）：

```bash
CUDA_VISIBLE_DEVICES=0 python baseline/run_experiments.py --checkpoint runs/baseline/best.pt
```

本機測試用 `--device cpu --count 1 --max-events-per-bar 4`，正式成果不應使用 smoke checkpoint。

程式單元檢查（因果遮罩、長小節還原與續寫接點）：

```bash
python -m unittest discover -s baseline -p 'test_*.py'
```

PyTorch API 參考：https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

## 指標總表：mean、std、與真實資料的差距

完成 real/t08/t10/t12 的逐首評估後，在作業根目錄執行（僅需 Python 標準函式庫）：

```bash
python3 baseline/summarize_metrics.py
```

讀取 `runs/metrics/{real,t08,t10,t12}.csv`，保留原檔，新增
`runs/metrics/summary.csv`，並在終端機印出可複製的 Markdown 表格。
每個指標保存平均值、樣本標準差（n−1）、有效與缺失樣本數，及與 real 平均值的
絕對差距。標準差描述歌曲間差異，不是信賴區間，也不是不同訓練 seed 的波動。
缺失或非有限值不當成零；只有一筆有效資料時標準差留空。
目前支援 H1/H4/GS；SI 尚未計算，不會填入零或假設已完成。
H4/GS 分別比較，不合成單一分數；沿用上述 64 格正規化評估限制。

其他實驗可自行指定檔案（須包含 reference）：

```bash
python3 baseline/summarize_metrics.py --input-dir runs/metrics --files real.csv t08.csv t10.csv t12.csv --reference real --output runs/metrics/summary.csv
```

獨立測試，不必安裝 PyTorch：

```bash
python3 -m unittest discover -s baseline -p 'test_summarize_metrics.py'
```

本次功能可獨立提交 `summarize_metrics.py`、`test_summarize_metrics.py` 及 README。
產生的 summary.csv 仍在忽略的 runs/ 下，應與其他實驗成果一起另行備份。

同一指令也會自動產生 `runs/metrics/summary.md`，提供中文欄名、四位小數、
平均值與差距分表、標準差說明。日常閱讀建議開啟 summary.md，
在 VS Code 按 `Ctrl+Shift+V` 開啟 Markdown 預覽；CSV 用於後續程式分析。
若指定 `--output`，Markdown 報告會使用相同檔名、改為 `.md` 副檔名。

## Optional 擴充實驗流程

進度、下一步與統一 WAV 合成設定見 [OPTIONAL_WORKFLOW.md](OPTIONAL_WORKFLOW.md)。
新版 `render_audio.py` 支援 piano 統一音色、遞迴子目錄、少量試轉、有效 WAV 檢查、
重跑核對與合成 manifest。預設關閉效果，既有 MIDI 不變。
SI 尚未開始；不需要重新訓練模型來合成 WAV。
