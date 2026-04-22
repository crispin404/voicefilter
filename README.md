# VoiceFilter 元音条件鼾声增强

本项目只保留两条可复现主流程：

- 本地 CPU 小规模跑通：使用 [config/enhancement.yaml](config/enhancement.yaml)
- Platformax GPU 全量训练：使用 [README_PLATFORMAX.md](README_PLATFORMAX.md) 和 [scripts/platformax_run_all.sh](scripts/platformax_run_all.sh)

任务输入为同一被试的 5 段元音 `a/e/i/o/u` 和一段鼾声 + 环境音混合音，模型输出目标被试的增强鼾声音频。

## 目录要求

原始数据根目录下每个被试一个文件夹，至少包含：

```text
data_root/
  subject_001/
    元音/
      a1_1.wav
      e1_1.wav
      i1_1.wav
      o1_1.wav
      u1_1.wav
    鼾声/
      hs_01_1.wav
      hs_01_2.wav
      ...
    info.txt
```

环境音目录采用平铺结构：

```text
noise_root/
  jb.wav
  km.wav
  qm.wav
  ye.wav
```

预训练 speaker embedder 需要放在：

```text
pretrained/embedder.pt
```

## 本地 CPU 跑通流程

先安装依赖：

```powershell
cd F:\voicefilter
pip install -r requirements.txt
```

扫描被试：

```powershell
python scripts/scan_dataset.py --data-root data/raw --output metadata/subjects.json
```

生成被试级划分：

```powershell
python scripts/build_subject_splits.py --subjects metadata/subjects.json --output-dir splits --train-count 3 --val-count 1 --test-count 1 --seed 42
```

增量合成 `合成声`：

```powershell
python scripts/synthesize_mixed_snore.py --subjects metadata/subjects.json --noise-root data/noise --output-subdir 合成声 --target-snr-db 8.0 --seed 42 --metadata-path metadata/synthesized_mix_metadata.jsonl --metadata-csv metadata/synthesized_mix_metadata.csv
```

预处理音频：

```powershell
python scripts/preprocess_audio.py --subjects metadata/subjects.json --processed-root processed --sample-rate 16000 --vowel-seconds 1.0 --mix-dir-name 合成声 --synthesis-metadata metadata/synthesized_mix_metadata.jsonl --snr-stats-csv metadata/preprocess_snr_stats.csv
```

生成 manifest：

```powershell
python scripts/build_manifests.py --subjects metadata/subjects.json --splits-dir splits --output-dir manifests --processed-root processed --mix-dir-name 合成声 --snr-stats-csv metadata/preprocess_snr_stats.csv
```

## 环境音变更后更新数据

以后新增或删除 `data/noise/*.wav` 后，不需要全量重跑。重新执行合成、预处理和 manifest 三步即可：

```powershell
python scripts/synthesize_mixed_snore.py --subjects metadata/subjects.json --noise-root data/noise --output-subdir 合成声 --target-snr-db 8.0 --seed 42 --metadata-path metadata/synthesized_mix_metadata.jsonl --metadata-csv metadata/synthesized_mix_metadata.csv
python scripts/preprocess_audio.py --subjects metadata/subjects.json --processed-root processed --sample-rate 16000 --vowel-seconds 1.0 --mix-dir-name 合成声 --synthesis-metadata metadata/synthesized_mix_metadata.jsonl --snr-stats-csv metadata/preprocess_snr_stats.csv
python scripts/build_manifests.py --subjects metadata/subjects.json --splits-dir splits --output-dir manifests --processed-root processed --mix-dir-name 合成声 --snr-stats-csv metadata/preprocess_snr_stats.csv
```

脚本会只补新增环境音对应的混合音和预处理文件，删除已剔除环境音对应的旧文件，并跳过源文件没有变化的样本。需要彻底重跑时，在合成和预处理命令里加 `--force`。

预计算元音 embedding：

```powershell
python scripts/precompute_vowel_embeddings.py -c config/enhancement.yaml --subjects metadata/subjects.json --processed-root processed --output-dir processed/embeddings --device cpu
```

CPU 训练：

```powershell
python scripts/train_enhancement.py -c config/enhancement.yaml --device cpu
```

单条推理试听：

```powershell
$testSubject = Get-Content splits/test_subjects.txt -Encoding UTF8 | Select-Object -First 1
$mixedFile = Get-ChildItem "processed/mix/$testSubject" -Filter *.wav | Select-Object -First 1 -ExpandProperty FullName
python scripts/infer_enhancement_long.py -c config/enhancement.yaml --checkpoint-path outputs/checkpoints/best.pt --mixed-file "$mixedFile" --vowel-dir "processed/vowel/$testSubject" --output-path "outputs/enhanced_wavs/${testSubject}_demo.wav" --device cpu
```

测试集评估：

```powershell
python scripts/evaluate_enhancement.py -c config/enhancement.yaml --checkpoint-path outputs/checkpoints/best.pt --output-csv outputs/eval/metrics.csv --device cpu
```

## 关键产物

本地 CPU 流程完成后重点检查：

```text
metadata/subjects.json
splits/train_subjects.txt
splits/val_subjects.txt
splits/test_subjects.txt
processed/vowel/
processed/clean/
processed/mix/
processed/embeddings/
manifests/enhancement_manifest_train.jsonl
manifests/enhancement_manifest_val.jsonl
manifests/enhancement_manifest_test.jsonl
outputs/checkpoints/best.pt
outputs/logs/train.log
outputs/eval/metrics.csv
```

## 保留脚本

当前主流程只依赖以下脚本：

- `scripts/scan_dataset.py`
- `scripts/build_subject_splits.py`
- `scripts/synthesize_mixed_snore.py`
- `scripts/preprocess_audio.py`
- `scripts/build_manifests.py`
- `scripts/precompute_vowel_embeddings.py`
- `scripts/train_enhancement.py`
- `scripts/infer_enhancement_long.py`
- `scripts/evaluate_enhancement.py`
- `scripts/check_platform_env.py`
- `scripts/platformax_run_all.sh`

Platformax 使用说明见 [README_PLATFORMAX.md](README_PLATFORMAX.md)。
