# Bonus：旋律條件鋼琴伴奏

這是獨立的新模型與資料，保留原本 Task A/B、深度和表示法結果。
採用助教 Option B，從隨機權重訓練 encoder-decoder Transformer。

## 任務與防止答案洩漏

- 訓練：全部 896 首 POP909，MELODY → PIANO；BRIDGE 不混入目標。
- 推論：十首提示只讀 melody.mid 和 beat_midi.txt。
- piano.mid、chord_midi.txt 只供助教評分器計算 CA/OA，不能作為生成條件。
- 沒有和弦輸入。模型從旋律／伴奏配對學習關係，不用 GT 和弦強迫生成。
- 原始旋律 MIDI 所有軌道事件直接保留，另外加入獨立 PIANO 軌與空閒 channel。

beat_midi 的最後一個下拍是右邊界，十個提示均是 16 個完整小節，而非17個。
001 有六個標註拍點的小節，不能強制把全部提示當成四拍。
推論伴奏位置透過 beat_midi 映回原始秒數，再以原旋律 tempo map 寫回 MIDI ticks。
原旋律保留原有 tempo、力度、控制事件；伴奏只在前16個完整小節內生成。

## 模型與訓練

| 項目 | 設定 |
|---|---|
| 架構 | bidirectional encoder 3 層 + causal decoder 4 層、cross-attention |
| 維度 | width384、heads6、FFN1536、dropout0.1、pre-LayerNorm |
| 表示法 | Bar/Beats/Tempo/Pos/Pitch/Vel/Dur，每拍16格 |
| 條件選擇 | Focus_0..15、TargetBars_1..4，只作 encoder 條件 |
| 詞彙表 | 1021 |
| Context | source2560（資料最大2556），target1024 |
| 配對抽樣 | 完整最多16小節旋律 + 其中最多4小節伴奏 |
| 更新 | AdamW、lr0.0003、weight_decay0.01、warmup200 + cosine、clip1 |
| 正式預設 | 20000 steps、batch4、seed42、全部896首 |
| 評估 | 固定隨機視窗的 training probe，非獨立驗證集 |

目標若4小節超出 context，就縮短成完整3/2/1小節，不在音符 token 中間截斷。
所有16小節旋律都可作 encoder 條件，不使用滑動視窗丟掉旋律前段。
推論以4個4小節區段完成16小節，每段重新讀取全部16小節旋律與對應 Focus。
各區段不以先前生成伴奏作輸入，因此邊界連貫性仍需試聽檢查。
生成固定 T=1.0、top-k32、top-p0.95、seed123–132。
每小節最多48個 onset，保留空小節，但每個4小節區段至少有一個音符。
音符最多256格，結尾裁切到提示的16小節邊界；每段都可在模型 context 內完整解碼。

## 準備與測試

```bash
python3 baseline/bonus/data.py
python3 -m unittest discover -s baseline/tests -p 'test_*.py'
```

data/bonus/pairs.json 是訓練曲的兩軌整數 token，以 bounded bars 分開保存。
summary.json 記錄所有896個 MIDI/beat來源雜湊、長度上限、使用／排除尾端音符數。
最末無右邊界的部分不當成完整小節，記錄排除數，不悄悄截去超長序列。

小模型（獨立 smoke，不能當作正式品質結果）：

```bash
python3 baseline/bonus/train.py --output runs/_smoke/bonus/train \
  --steps 2 --batch-size 2 --width 32 --encoder-layers 1 --decoder-layers 1 \
  --heads 4 --eval-every 2 --eval-batches 1 --device cpu
python3 baseline/bonus/generate.py --checkpoint runs/_smoke/bonus/train/last.pt \
  --output runs/_smoke/bonus/generated --limit 1 --max-events-per-bar 2 --device cpu
python3 baseline/bonus/evaluate.py --generated runs/_smoke/bonus/generated \
  --output runs/_smoke/bonus/metrics.csv --expected-songs 1
```

若輸出目錄已存在，保留結果並改用新的路徑，避免誤覆蓋。

## EDA Union 正式流程

在學校現有共用套件執行，不安裝或降級 PyTorch。
先同步 baseline 新增程式與助教 eval_bonus.py，再在學校準備資料。

```bash
cd ~/DL_music_hw2
python3 baseline/bonus/data.py
python3 baseline/experiments/run_bonus.py --stage plan
```

檢查 nvidia-smi，有可用 GPU 時，在獨立 tmux 內啟動：

```bash
CUDA_VISIBLE_DEVICES=0 python3 baseline/experiments/run_bonus.py --stage all \
  > runs/bonus_accompaniment/run.log 2>&1
```

查看紀錄：

```bash
tail -f ~/DL_music_hw2/runs/bonus_accompaniment/run.log
```

預設只建／核對 plan；all 才訓練、生成10首及評分。中斷訓練檢查後加 --resume。
已完成同身分 checkpoint 不重訓，已完成同身分10首不重新生成。
部分生成輸出不自動覆蓋，程式或條件改變則需新 --output。
僅使用最後一步 checkpoint，會確認正式訓練已抽樣到全部896首歌曲。

## CA／OA

baseline/bonus/evaluate.py 呼叫未修改的 HW2_scripts/eval_bonus.py。
必須完整評估10首、各16小節；若助教腳本跳過歌曲或產生非有限值，wrapper拒絕報告平均。

- CA：伴奏估計和弦與提供標籤的一致性，估計器僅支援24種大小三和弦。
- CA-ceiling：同一估計器在GT鋼琴上的診斷結果，不是嚴格上界。
- OA：逐小節的 pitch-class、velocity、duration、inter-onset interval 分布重疊面積。
- 官方平均按小節加權；另外記錄歌曲間樣本標準差，不是信賴區間。
- GT no-chord／不支援和弦會影響 CA；OA 高也不保證伴奏聽起來自然。

輸出 runs/bonus_accompaniment/metrics.csv、metrics.summary.json、metrics.md。
尚未完成正式訓練與試聽前，不把 smoke 成績寫成 Bonus 正式結果。

## WAV 與繳交

完成遠端10首後下載 runs/bonus_accompaniment，再在本機用相同鋼琴音色轉 WAV：

```bash
python3 baseline/audio/render_audio.py --input runs/bonus_accompaniment/generated \
  --output runs/bonus_accompaniment/audio --soundfont /path/to/GeneralUser-GS.sf2 \
  --fluidsynth fluidsynth --gain 0.025
```

檢查十首無截波、旋律事件保留、完整16小節及試聽品質，再更新報告 Bonus 章節。
提交10 MIDI+10 WAV，記明模型、表示法、推論設定與 CA/OA。
不要把 Bonus 提示的 GT piano.mid／chord_midi.txt 當成生成成果。
目前的 assemble_submission.py 尚未包含 Bonus，等成果完成再擴充打包。
模型與成果放雲端、程式入repo，Git仍由使用者操作。

## 啟動前驗證紀錄

- 本機與 EDA 全部26項測試通過，涵蓋 decoder 因果性、旋律條件有效性、source padding、完整小節切分。
- 測試用無效 GT piano/chord 檔仍可條件生成，確認推論不依賴答案檔。
- 小模型成功生成001的16小節、兩軌 MIDI，原始旋律軌全部事件一致。
- 助教 CA/OA 腳本成功評分 smoke 的16小節，沒有跳過；smoke 分數不作正式成績。
- 正式模型大小的兩步 CUDA 訓練／training probe 已通過。
- 本機與 EDA 的完整 plan 相同，正式工作目錄為 runs/bonus_accompaniment，tmux session 為 hw2_bonus。
- 正式訓練／十首生成／CA/OA／十首 WAV／試聽／報告仍依序完成，啟動不等於 Bonus 已完成。
