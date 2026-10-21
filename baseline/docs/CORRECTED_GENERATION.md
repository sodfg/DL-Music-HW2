# 修正版生成、評估與報告

正式新目錄：`runs/corrected_boundary_v2`。舊成果保留供診斷，不再作新版音樂的評分或報告主表。
沿用既有六層、三層模型。這次沒有重新訓練。

## 修正

- 最後小節可選擇模型學會的 `Bar`，程式轉成 EOS 結束，仍恰好32小節。
- 禁止同小節、同位置、同音高重複，允許和弦與稍後重彈同音。
- 每小節上限與結尾長音裁切保留。`generate.py` 預設仍為 legacy，正式新批次明確指定 boundary-v2。
- 修正會改變生成序列，不能把舊的分數或舊的聽感當成新版結果。

## EDA：生成與 H1/H4/GS

```bash
python3 baseline/experiments/run_corrected_generation.py --stage plan
CUDA_VISIBLE_DEVICES=0 python3 -u baseline/experiments/run_corrected_generation.py --stage all
```

共60首 Basic A、9首 Basic B、20首三層模型。六層深度比較使用同批 Basic A t10。
固定 top-k32、top-p0.95、seed123–142。模型及程式雜湊記錄於 plan.json。
各組 completed.json 包含 MIDI 雜湊、逐小節音符數、重複事件及 Task B 原提示保留檢查。
可驗證完整組後接續，部分組不自動覆蓋。

## 本機：WAV、SI 與總表

```bash
.venv/bin/python -u baseline/experiments/run_corrected_audio.py --watch-remote \
  --soundfont /path/to/GeneralUser-GS.sf2 --fluidsynth /path/to/fluidsynth
```

`--watch-remote` 觀察已啟動的 EDA 批次，不會啟動訓練或重複生成。
預設 alias eda_union_sodfg、遠端 DL_music_hw2/runs/corrected_boundary_v2。
各組生成完成後下載並轉WAV／計算SI。設定與協定須一致，截波或短於48秒會報錯，不悄悄排除歌曲。

- WAV：gain0.025、相同鋼琴SoundFont、44.1kHz、無正規化，記錄來源與雜湊。
- SI：前48秒，short3–8、mid8–15、long15–48秒。
- 真實參考：H4/GS為原896首全曲，SI沿用固定20首且驗證來源／協定。
- MIDI-like：不使用出問題的Bar/Pos停止邏輯。來源、模型、生成器及同協定音訊均驗證後，才重用原20首結果。
- 表示法比較仍包含合法候選／時鐘限制差異，不能當成純表示法因果實驗。

總表：summary.csv、summary.json、summary.md。新 SI 與渲染稽核在 si/、audio/。
`postprocess_completed.json` 代表89份WAV及總表完成，報告與試聽選曲另行確認。

## 報告與打包

舊報告v3及以前含舊Bar/Pos結果，保留備份而不再用於繳交。
新報告v4只引用新Bar/Pos總表，Bonus保留已完成的新獨立流程。
Task A整組20首與Task B各提示最佳版本、雲端連結仍由使用者確認。

```bash
.venv/bin/python baseline/experiments/assemble_submission.py \
  --output output/submission_corrected_draft \
  --corrected-results runs/corrected_boundary_v2 --include-bonus \
  --soundfont /path/to/GeneralUser-GS.sf2
```

未指定--task-a會保留待選，不代替使用者選曲。程式不操作Git，也不上傳雲端。
