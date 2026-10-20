# Token 表示法實驗：Bar/Pos vs MIDI-like

## 問題與固定條件

比較兩種描述同一份音樂的方式，固定 896 首、6 層 decoder-only Transformer、
width=512、heads=8、dropout=0.1、context=1024、batch=8、20000 steps、lr=0.0003、
seed=42、全部資料訓練。訓練程式與最佳化方法完全沿用 baseline/train.py。
新模型存在獨立目錄，不會覆蓋六層 baseline 或三層模型。

原版：`Bar / Beats / Tempo / Pos / Pitch / Vel / Dur`。
新版：`Tempo / Vel / NoteOn / NoteOff / TimeShift`，另有 PAD/BOS/EOS。
例如 `Vel_81, NoteOn_60, TimeShift_16, NoteOff_60` 表示按下中央 C，一拍後放開。
TimeShift 每格是 1/16 個標註拍點，不是毫秒；最大單次 256 格，較長間隔拆開。

新版是自製的 beat-grid MIDI-like 變體，不宣稱為標準 MidiTok MIDI-Like。
原始拍號仍不可信，因此沿用 beat_midi.txt 衍生的舊版量化結果，並未根據 MIDI
內嵌拍號重新切小節。它不是直接使用原始未量化 MIDI 的毫秒事件。

## 資料與還原檢查

從 data/baseline/sequences.json 轉換，不重新估計節拍：

```bash
python3 baseline/representations/midi_like.py
```

這會建立 data/midi_like/。每首均檢查所有 note-on/off、velocity、tempo 與總長度
完全符合 baseline 解碼的事件，包含同音高重疊、長小節、跨小節音符與結尾截斷。
另外逐首比對 H4/GS 所用的 onset/pitch/velocity，避免評分輸入意外改變。

同音高重疊保留每一個 MIDI note-on/off；MIDI 本身無逐音符配對 ID。
因此音訊解碼直接寫事件，不能用單一 pitch→note 字典蓋掉重疊音符。
僅評分用的 Bar/Pos view 以 FIFO 配對 duration；H4/GS 不使用 duration。

小節長度另存 bar_metadata.json，供來源驗證與還原檢查；不餵給 MIDI-like 模型。
每小節原有的 Tempo 事件仍保留，所以不能聲稱模型輸入完全沒有小節邊界線索。

本次 896 首轉換結果：

| 項目 | Bar/Pos baseline | MIDI-like |
|---|---:|---:|
| 詞彙表大小 | 1001 | 584 |
| token 總數（含 BOS/EOS） | 6,258,677 | 6,258,395 |
| 音符 onset 總數 | 1,508,536 | 1,508,536 |
| 六層模型參數量 | 19,952,128 | 19,738,624 |
| 全資料平均每 1024 tokens 約涵蓋拍數 | 48.85 | 48.85 |

詞彙表變小使 embedding 參數略減，不能宣稱參數量完全相同。
以上拍數是全資料平均估算，不代表每個訓練 crop 覆蓋相同音樂長度。

## 本機 smoke

```bash
python3 -m unittest discover -s baseline/tests -p 'test_*.py'
python3 baseline/train.py --data data/midi_like \
  --output runs/_smoke/midi_like/train --steps 2 --batch-size 2 \
  --context 64 --width 32 --layers 2 --heads 4 \
  --eval-every 2 --eval-batches 1 --validation-fraction 0 --device cpu
python3 baseline/representations/generate_midi_like.py \
  --checkpoint runs/_smoke/midi_like/train/last.pt \
  --output runs/_smoke/midi_like/generated --count 2 --bars 4 \
  --max-events-per-bar 3 --device cpu
python3 baseline/evaluation/evaluate.py \
  --data runs/_smoke/midi_like/generated --output runs/_smoke/midi_like/metrics.csv
```

smoke 僅測流程，不能當成正式音樂品質結果。若已有同名輸出，另選新的測試目錄。

## EDA Union 正式實驗

先在本機同步 baseline 程式。學校沿用既有共用套件，不安裝／降級 PyTorch。
EDA 專案目前沒有 .git，push GitHub 不會自動同步學校檔案。

```bash
rsync -av --exclude='__pycache__/' baseline/ eda_union_sodfg:~/DL_music_hw2/baseline/
```

在學校專案根目錄，先轉換及查看計畫：

```bash
cd ~/DL_music_hw2
python3 baseline/representations/midi_like.py
python3 baseline/experiments/run_representation_experiment.py --stage plan
```

確認 GPU 可用後，在 tmux 中啟動：

```bash
CUDA_VISIBLE_DEVICES=0 python3 baseline/experiments/run_representation_experiment.py \
  --stage all > runs/representation_midi_like/run.log 2>&1
```

外部終端查看：

```bash
tail -f ~/DL_music_hw2/runs/representation_midi_like/run.log
```

Ctrl+C 會停止 tail 的查看；不會停止另一個 tmux session 內的實驗。
主程式預設只建立／核對 plan，加 --stage all 才會執行完整流程。
未完成訓練可檢查後加 --resume；完成的 step-20000 checkpoint 不會重新訓練。
生成中斷若只剩部分檔案，生成器拒絕覆蓋，應保存現況並另選輸出或先人工處理。
若改了程式、資料或條件，計畫會拒絕混用，應換新的 --output。

## 輸出與評分

```text
runs/representation_midi_like/
  plan.json                     固定條件、資料與程式雜湊
  execution.json                新模型程序計時與環境
  run.log                       遠端執行紀錄
  train/                        新六層模型 checkpoint 與設定
  generation_identity.json      生成使用的 checkpoint 身分
  generated/
    sample_000.mid ...           20 首四拍、32 小節的 Task A 音樂
    event_sequences.json        實際 MIDI-like 模型輸出
    event_tok2idx.json           MIDI-like 詞彙表
    sequences.json              僅供 H4/GS 的 Bar/Pos onset view
    tok2idx.json                 onset view 使用的詞彙表
  metrics.csv                   逐首 H1/H4/GS
  comparison.csv                兩組平均、標準差及與真實資料的差距
  comparison.md                 可讀結果
```

生成固定 T=1.0、top-k=32、top-p=0.95、seed=123–142，與原版 t10 比較。
使用相同的 64 格小節正規化及助教 MusDr H4/GS 程式，不把 MIDI-like tokens
直接交給只理解 Bar/Pos 的評分程式。

生成器保留合法事件、每小節至少一個 onset、每小節最多 128 onsets 的限制。
音符最長 256 格，完成時關閉所有 active notes，總長度精確為 32×4 拍。
以外部時鐘確保邊界及每小節 Tempo；不是要求完全無限制的模型自行計數 32 小節。
兩種語法的候選集合不同，不能把整個實驗解釋成完全隔離表示法的純因果證明。

只做 Task A 的 optional 比較；不替換已完成的 Task B。
不同比例詞彙表的 cross-entropy 不能直接比較成音樂品質。
同 seed 也不代表兩種表示法生成同一首，因模型與抽樣事件不同。
SI 及試聽在 H4/GS 完成後補上；只有一個訓練 seed，不宣稱普遍優劣。
Git 由使用者操作；data 與 runs 仍被忽略，模型與音樂需另外備份。

## 第一輪結果已完成

EDA Union GPU 0 完成 20000 步訓練，並生成 20 首、各 32 小節的 MIDI。
訓練程序 wall time 約 900.4 秒（含初始化、讀檔、評估與存檔）；不同舊實驗的計時
定義不完全相同，不直接拿來計算速度比。環境仍是 PyTorch 2.6.0+cu124。

| 資料／表示法 | H4 平均 | GS 平均 |
|---|---:|---:|
| 真實資料（896 首全曲） | 2.6250 | 0.9019 |
| Bar/Pos baseline（20 首） | 2.5766 | 0.9662 |
| MIDI-like（20 首） | 2.5307 | 0.7885 |

本輪 Bar/Pos 的 H4 與 GS 平均都較接近真實參考。MIDI-like 的 GS 低於參考，
Bar/Pos 則高於參考；因此不能用分數越高就越好來排名，也不據此宣稱某表示法普遍較差。

本機核對 20/20 首的事件序列、重新解碼 MIDI 位元組、評分 view 與完整有限分數
均一致。MIDI 音樂長度約 61.93–107.69 秒，符合後續相同 48 秒 SI 的長度需求。
各項標準差與平均差距見 runs/representation_midi_like/comparison.md。
下一步試聽與 SI 尚未完成，不在此表列入未計算的結果。

## 48 秒 SI 與收尾

MIDI-like 的 20 首已補齊相同音色、gain=0.025 的完整 WAV，並各分析前 48 秒。
真實與 Bar/Pos 的參考分數只在完整 SI 協定（含 wrapper 雜湊、軟體版本及渲染設定）
完全相同，且來源 MIDI/WAV 雜湊一致時才沿用。重現：

```bash
.venv/bin/python baseline/audio/render_audio.py \
  --input runs/representation_midi_like/generated \
  --output runs/representation_midi_like/si48/audio/midi_like \
  --soundfont /path/to/GeneralUser-GS.sf2 --fluidsynth fluidsynth --gain 0.025
.venv-si/bin/python baseline/evaluation/evaluate_si.py --scope batch \
  --audio-root runs/representation_midi_like/si48/audio --groups midi_like \
  --limit-per-group 20 --duration 48 \
  --output runs/representation_midi_like/si48/groups/midi_like
.venv/bin/python baseline/experiments/summarize_representation_si.py
```

SI 平均：真實 short/mid/long=0.3299/0.2733/0.1625，Bar/Pos=0.2934/0.2127/0.0901，
MIDI-like=0.2973/0.1744/0.0247。MIDI-like 的 short 略近；Bar/Pos 的 mid、long 較近。
不能把這項單 seed、20 首片段比較當成統計顯著性或整體品質結論。
結果與標準差在 runs/representation_midi_like/si48/comparison.md。

Task B 的九首 WAV 已在 runs/listening/basic_b 補齊，來源/無截波檢查通過。
重新核對 MIDI tick 邊界之前的所有原始提示事件，9/9 一致；接點受 MIDI tick
解析度影響，最大與標註時間差約 1.05ms，並非任意修改提示。

本機繳交草稿可使用 assemble_submission.py 建立，預設保留 Task A 與 Task B 最佳版本
待選，不會上傳或操作 Git。若已有草稿，請使用新的 --output，避免覆蓋：

```bash
.venv/bin/python baseline/experiments/assemble_submission.py \
  --soundfont /path/to/GeneralUser-GS.sf2 --output output/submission_draft
```

試聽決定後可在新的輸出目錄加 --task-a t08（或 t10/t12），建立相對應的 20 MIDI+20 WAV。
另需記錄三個 Task B 最佳版本、加入正式報告、上傳雲端並確認讀取權限。
