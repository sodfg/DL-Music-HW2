# 模型深度實驗：3 層 vs 6 層

問題：固定音樂表示法與訓練 token 預算，減少 Transformer 層數會如何影響生成與成本？
這是模型深度比較，不是新的 token representation 實驗。

| 條件 | 六層 reference | 三層實驗 |
|---|---|---|
| 表示法／訓練資料 | 現有 1001 種 tokens、全部 896 首 | 相同 |
| Transformer layers | 6 | 3 |
| Width / heads / dropout | 512 / 8 / 0.1 | 相同 |
| Context / batch size | 1024 / 8 | 相同 |
| Steps / learning rate | 20000 / 0.0003，原 cosine 排程 | 相同 |
| Optimizer | AdamW，weight decay 0.01 | 相同 |
| Training seed | 42 | 42 |
| 參數量 | 19,952,128 | 10,494,976 |
| 模型選擇 | 已核對在 step 20000 的 best.pt | 固定 step 20000 的 last.pt |

train.py 的資料抽樣使用獨立的 random.Random，因此相同資料、種子與訓練步數
會使用相同的歌曲／片段抽樣順序，不受模型初始化消耗多少 torch RNG 影響。
不同架構的初始化與 dropout 隨機狀態不會完全相同，這是一次單 seed 的控制變因比較。
896 首全部參與訓練，沒有獨立驗證集，training_probe 不作泛化能力結論。
同樣步數不代表同樣運算量，也不直接代表三層模型品質一定較差。

## 計畫與安全檢查

在本機可先檢視／建立計畫：

```bash
.venv/bin/python baseline/experiments/run_depth_experiment.py --stage plan
```

在 EDA Union 使用既有的 Python 3.10 / PyTorch 2.6.0+cu124 共用套件，
不需要重裝 requirements。原本 requirements.txt 的舊版 torch 固定值不適用此環境。

程式核對六層 checkpoint 的 step、設定、字典、訓練切分、生成參數，
並保存訓練資料與來源程式的 SHA-256。既有計畫不符時會拒絕執行。
所有新結果在 `runs/depth_3_vs_6/`，不覆蓋 `runs/baseline/` 或舊 t10。

## EDA Union 執行

先以 nvidia-smi 確認空閒 GPU，再使用獨立 tmux：

```bash
cd ~/DL_music_hw2
python3 baseline/experiments/run_depth_experiment.py --stage plan
tmux new -s hw2_depth
CUDA_VISIBLE_DEVICES=0 python3 baseline/experiments/run_depth_experiment.py --stage all >> runs/depth_3_vs_6/run.log 2>&1
```

如果這個實驗已在跑，不要再啟動第二次。其他終端機查看：

```bash
tail -f ~/DL_music_hw2/runs/depth_3_vs_6/run.log
```

按 Ctrl+C 只停止 tail，不停止 tmux 中的實驗。
回去看工作區使用 `tmux attach -t hw2_depth`，離開用 Ctrl+B，放開，再按 D。

三層模型完成後，程式自動使用 last.pt 生成 20 首，設定固定為：

- 32 小節、每小節四拍。
- Temperature 1.0、top-k 32、top-p 0.95。
- 種子 123–142、每小節最多 128 個音符，與六層 t10 相同。
- 其餘 grammar constraints、context 滑動方式與原程式完全一致。

接著呼叫原本 evaluate.py 計算 H1/H4/GS，生成 comparison.csv 和 comparison.md。
六層使用既有 t10.csv，真實全曲參考使用既有 real.csv；總表核對每模型 20 首。
本輪先完成 H4/GS，不自動合成新的 SI 音訊。先檢查新曲最短長度，才能決定是否
能沿用原本的 48 秒 SI 協定；不能為湊長度而刪掉短曲或更換 seed。

## 中斷與重跑

已有部分 checkpoint 時，先確認程序停止，再明確使用：

```bash
CUDA_VISIBLE_DEVICES=0 python3 baseline/experiments/run_depth_experiment.py --stage all --resume
```

完成的 20000 步模型會保留，不重新訓練。完成的同一 checkpoint、同設定的 20 首
會保留，不重新生成。沒有可續訓 checkpoint 的殘留訓練資料夾不會自動清除，
應另用新的 --output；不要直接覆蓋未知成果。

## 結果與限制

```text
runs/depth_3_vs_6/
  plan.json                固定比較條件及輸入雜湊
  run.log                  訓練、生成與評估記錄
  train/                   三層模型、loss、設定
  execution.json           三層執行環境與程序 wall time
  generation_identity.json 生成使用的 checkpoint 與設定身分
  generated/               三層模型的20首 MIDI
  metrics.csv              三層逐首 H1/H4/GS
  comparison.csv           六層／三層的平均、標準差與真實參考差距
  comparison.md            可讀報告
```

成本比較不能把六層舊 log 的 elapsed 和三層 wrapper 的 process wall time 直接相除：
後者包含初始化、資料讀取與最後評估／存檔，定義不相同。若要報告訓練加速比或
顯存差異，下一步需以相同計時與硬體條件做獨立短 benchmark。
只有一個訓練 seed，不宣稱統計顯著性；指標差異也不直接代表聽感或原創性。

程式與文件交給 Git；runs/ 成果另行備份。測試：

```bash
.venv/bin/python -m unittest discover -s baseline/tests -p 'test_depth_experiment.py'
```

## 第一輪結果已完成

三層訓練、20 首 32 小節生成及 H4/GS 已在 EDA Union GPU 0 完成，成果已下載本機。
comparison.md 包含平均、標準差及與真實資料的差距；此輪三層 H4 更接近參考，
GS 差異很小，不宣稱單 seed 的結果足以證明三層普遍優於六層。
套件仍是既有 PyTorch 2.6.0+cu124，未在學校主機安裝或更換套件。

## 補齊深度比較的 SI（本機 CPU）

不重新訓練或生成 MIDI。沿用 `runs/metrics/si48_g0025` 中已驗證的真實參考與六層
WAV，為三層的 20 首 MIDI 轉鋼琴 WAV；三組均使用目前同一支 SI 程式重新計算。
每首只分析開頭 48 秒，音色、FluidSynth 版本與 gain=0.025 必須完全一致。
原有真實抽樣固定 seed=42，從符合長度條件的 895 首中取 20 首。

先檢查來源並產生固定計畫：

```bash
.venv/bin/python baseline/experiments/run_depth_si.py
```

執行（將音色與 FluidSynth 路徑換成自己的位置）：

```bash
.venv/bin/python baseline/experiments/run_depth_si.py \
  --execute \
  --si-python .venv-si/bin/python \
  --fluidsynth .tools/fluidsynth/bin/fluidsynth \
  --soundfont /path/to/GeneralUser-GS.sf2
```

結果在 `runs/depth_3_vs_6/si48/comparison.md`，包含三組平均、樣本標準差及與真實組
平均的絕對差距；逐首值在 `per_piece.csv`，每組完整協定、快取與紀錄也會保存。
同一輸入重跑會保留相同 WAV 並使用 SI 快取；來源或設定變更需另開輸出資料夾。

真實與六層音訊使用連到原 SI 研究目錄的符號連結，避免複製大型 WAV。
備份時請保留兩份研究資料夾，或讓備份工具展開符號連結。
H4/GS 使用全曲與全部 896 首參考，SI 使用 48 秒與 20 首參考，因此維持獨立表格。

本輪 SI 已完成（各 20 首，平均 ± 樣本標準差）：

| 組別 | SI short | SI mid | SI long |
|---|---:|---:|---:|
| 真實參考 | 0.3299 ± 0.0536 | 0.2733 ± 0.1178 | 0.1625 ± 0.1640 |
| 六層 | 0.2934 ± 0.0305 | 0.2127 ± 0.0552 | 0.0901 ± 0.0855 |
| 三層 | 0.2730 ± 0.0299 | 0.1850 ± 0.0570 | 0.0575 ± 0.0727 |

六層在三項 SI 平均都較接近這組真實參考；先前 H4 則是三層較接近。
這表示不同指標呈現不同面向，不能據此宣布單一模型整體勝出。
上述平均並非全部真實資料集的 SI 平均，也未做多訓練 seed 的顯著性檢定。

## 本機與 EDA Union 的同步

GitHub 的 push 不會自動修改 EDA Union 上的檔案。目前學校專案是一般資料夾，
沒有 `.git`；使用本機的 `rsync` 更新程式，不要在遠端直接執行 `git pull`。
先 commit，再預覽與同步：

```bash
rsync -av --dry-run --exclude='__pycache__/' baseline/ eda_union_sodfg:~/DL_music_hw2/baseline/
rsync -av --exclude='__pycache__/' baseline/ eda_union_sodfg:~/DL_music_hw2/baseline/
```

此指令僅同步 baseline 程式與文件，不同步 runs、模型或資料，也不刪除遠端檔案。
遠端舊版平面腳本會保留，後續命令請使用新版子資料夾路徑。這次 SI 在本機完成，
不必先同步才能計算；下次在 EDA Union 跑新實驗之前應完成同步。
