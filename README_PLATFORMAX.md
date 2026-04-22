# Platformax GPU 运行说明

本说明用于在 Platformax JupyterLab 终端运行全量 GPU 流程。推荐项目目录：

```bash
/opt/data/private/mel-data/毕设/voicefilter
```

## 目录准备

在项目下准备数据和权重：

```text
voicefilter/
  data/
    raw/        # 完整原始被试数据，每个被试一个文件夹
    noise/      # 环境音 wav，例如 jb.wav/km.wav/qm.wav/ye.wav
  pretrained/
    embedder.pt # 预训练 speaker embedder
```

`pretrained/` 和 `data/` 默认不会进 git，需要单独上传到平台。

## 第 1 步：进入项目目录

```bash
cd /opt/data/private/mel-data/毕设/voicefilter
```

如果你使用的是其他目录，后续命令都要先 `cd` 到项目根目录再执行。

## 第 2 步：安装平台依赖

不要在平台上运行 `pip install -r requirements.txt`，它会安装 CPU/GPU 不确定的 torch。Platformax 镜像已经带 PyTorch + CUDA，本项目只补齐非 torch 依赖：

```bash
python -m pip install --user -r requirements-platform.txt
```

如果 `python -m pip` 不可用，先试：

```bash
pip3 install --user -r requirements-platform.txt
```

如果平台支持 venv，也可以使用：

```bash
python -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install -r requirements-platform.txt
```

## 第 3 步：检查 CUDA、数据和 embedder

默认检查 GPU0：

```bash
python scripts/check_platform_env.py --require-cuda --device cuda:0 --data-root data/raw --noise-root data/noise
```

如果要使用 GPU1：

```bash
python scripts/check_platform_env.py --require-cuda --device cuda:1 --data-root data/raw --noise-root data/noise
```

你应该看到：

- `CUDA available: True`
- GPU 名称
- `Raw subject directories` 大于 0
- `Noise wav files` 大于 0
- `Embedder checkpoint` 存在

如果这里失败，先不要继续训练。

## 一键全流程

如果你已经确认环境、数据和权重都正常，可以直接运行一键全流程。

默认使用 `cuda:0`：

```bash
bash scripts/platformax_run_all.sh
```

使用 `cuda:1`：

```bash
DEVICE=cuda:1 bash scripts/platformax_run_all.sh
```

如果 GPU0 被占用，最稳妥的方式是只暴露物理 GPU1：

```bash
CUDA_VISIBLE_DEVICES=1 DEVICE=cuda:0 bash scripts/platformax_run_all.sh
```

一键脚本会依次完成扫描、划分、合成、预处理、manifest、embedding、训练和评估。

## 分步全流程

如果你想一步一步看哪里出问题，就按下面顺序执行。前一步失败时先停，不要继续往后跑。

### 第 4 步：扫描 `data/raw`

```bash
python scripts/scan_dataset.py \
  --data-root data/raw \
  --output metadata/subjects.json
```

你应该看到有效被试数量，并生成：

```text
metadata/subjects.json
```

### 第 5 步：生成 80/10/10 被试级划分

```bash
python scripts/build_subject_splits.py \
  --subjects metadata/subjects.json \
  --output-dir splits \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42
```

产物：

```text
splits/train_subjects.txt
splits/val_subjects.txt
splits/test_subjects.txt
```

### 第 6 步：增量生成 `合成声`

```bash
python scripts/synthesize_mixed_snore.py \
  --subjects metadata/subjects.json \
  --noise-root data/noise \
  --output-subdir 合成声 \
  --target-snr-db 8.0 \
  --seed 42 \
  --metadata-path metadata/synthesized_mix_metadata.jsonl \
  --metadata-csv metadata/synthesized_mix_metadata.csv
```

产物：

```text
每个被试目录下的 合成声/*.wav
metadata/synthesized_mix_metadata.jsonl
metadata/synthesized_mix_metadata.csv
```

这一步是增量同步：新增环境音时只补新增 noise_type 的混合音，删除环境音时会删除对应旧混合音，源文件没有变化的样本会跳过。

### 第 7 步：预处理音频

```bash
python scripts/preprocess_audio.py \
  --subjects metadata/subjects.json \
  --processed-root processed \
  --sample-rate 16000 \
  --vowel-seconds 1.0 \
  --mix-dir-name 合成声 \
  --synthesis-metadata metadata/synthesized_mix_metadata.jsonl \
  --snr-stats-csv metadata/preprocess_snr_stats.csv
```

产物：

```text
processed/vowel/
processed/clean/
processed/mix/
metadata/preprocess_snr_stats.csv
```

这一步同样是增量同步：新增混合音时只补预处理文件，删除混合音时会删除对应的 `processed/mix` 和 `processed/clean`，源文件没有变化的样本会跳过。

### 第 8 步：生成 manifest

```bash
python scripts/build_manifests.py \
  --subjects metadata/subjects.json \
  --splits-dir splits \
  --output-dir manifests \
  --processed-root processed \
  --mix-dir-name 合成声 \
  --snr-stats-csv metadata/preprocess_snr_stats.csv
```

检查每个 split 都不是 0：

```bash
wc -l manifests/*.jsonl
```

产物：

```text
manifests/enhancement_manifest_train.jsonl
manifests/enhancement_manifest_val.jsonl
manifests/enhancement_manifest_test.jsonl
```

### 第 9 步：预计算元音 embedding

使用 GPU0：

```bash
python scripts/precompute_vowel_embeddings.py \
  -c config/platform_gpu.yaml \
  --subjects metadata/subjects.json \
  --processed-root processed \
  --output-dir processed/embeddings \
  --device cuda:0
```

使用 GPU1：

```bash
python scripts/precompute_vowel_embeddings.py \
  -c config/platform_gpu.yaml \
  --subjects metadata/subjects.json \
  --processed-root processed \
  --output-dir processed/embeddings \
  --device cuda:1
```

产物：

```text
processed/embeddings/*.npy
```

### 第 10 步：GPU 训练

使用 GPU0：

```bash
python scripts/train_enhancement.py \
  -c config/platform_gpu.yaml \
  --device cuda:0
```

使用 GPU1：

```bash
python scripts/train_enhancement.py \
  -c config/platform_gpu.yaml \
  --device cuda:1
```

如果 GPU0 被占用，推荐只暴露物理 GPU1：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_enhancement.py \
  -c config/platform_gpu.yaml \
  --device cuda:0
```

产物：

```text
outputs/platformax/checkpoints/latest.pt
outputs/platformax/checkpoints/best.pt
outputs/platformax/logs/train.log
```

### 第 11 步：断点续训

如果训练中断且 `latest.pt` 已存在，使用：

```bash
python scripts/train_enhancement.py \
  -c config/platform_gpu.yaml \
  --device cuda:1 \
  --checkpoint-path outputs/platformax/checkpoints/latest.pt
```

如果使用 `CUDA_VISIBLE_DEVICES=1`：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_enhancement.py \
  -c config/platform_gpu.yaml \
  --device cuda:0 \
  --checkpoint-path outputs/platformax/checkpoints/latest.pt
```

### 第 12 步：评估测试集

使用 GPU1：

```bash
python scripts/evaluate_enhancement.py \
  -c config/platform_gpu.yaml \
  --checkpoint-path outputs/platformax/checkpoints/best.pt \
  --output-csv outputs/platformax/eval/metrics.csv \
  --save-wavs-dir outputs/platformax/eval/enhanced_wavs \
  --device cuda:1
```

如果使用 `CUDA_VISIBLE_DEVICES=1`：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_enhancement.py \
  -c config/platform_gpu.yaml \
  --checkpoint-path outputs/platformax/checkpoints/best.pt \
  --output-csv outputs/platformax/eval/metrics.csv \
  --save-wavs-dir outputs/platformax/eval/enhanced_wavs \
  --device cuda:0
```

产物：

```text
outputs/platformax/eval/metrics.csv
outputs/platformax/eval/enhanced_wavs/
```

## 环境音变更后如何更新数据

以后如果你新增 `data/noise/*.wav`，或删除效果不好的环境音，不需要从头全量重合成。重新运行下面三步即可：

```bash
python scripts/synthesize_mixed_snore.py \
  --subjects metadata/subjects.json \
  --noise-root data/noise \
  --output-subdir 合成声 \
  --target-snr-db 8.0 \
  --seed 42 \
  --metadata-path metadata/synthesized_mix_metadata.jsonl \
  --metadata-csv metadata/synthesized_mix_metadata.csv

python scripts/preprocess_audio.py \
  --subjects metadata/subjects.json \
  --processed-root processed \
  --sample-rate 16000 \
  --vowel-seconds 1.0 \
  --mix-dir-name 合成声 \
  --synthesis-metadata metadata/synthesized_mix_metadata.jsonl \
  --snr-stats-csv metadata/preprocess_snr_stats.csv

python scripts/build_manifests.py \
  --subjects metadata/subjects.json \
  --splits-dir splits \
  --output-dir manifests \
  --processed-root processed \
  --mix-dir-name 合成声 \
  --snr-stats-csv metadata/preprocess_snr_stats.csv
```

脚本会自动补新增环境音对应的文件、删除已剔除环境音对应的旧文件，并跳过没有变化的样本。需要彻底重跑时，在合成和预处理命令里加 `--force`。

旧的 `合成声_2/` 不再参与流程，可以手动删除或忽略。

## 最后检查

确认模型和评估结果存在：

```bash
ls -lh outputs/platformax/checkpoints/best.pt
ls -lh outputs/platformax/eval/metrics.csv
```

确认训练使用 GPU：

```bash
grep -E "Using device|CUDA device name|epoch=" outputs/platformax/logs/train.log
```

你应该能看到类似：

```text
Using device: cuda:1
CUDA device name: NVIDIA GeForce RTX 4090
epoch=1 train_loss=...
```

## 常见问题

- `CUDA available: False`：当前 Python 没拿到 CUDA 版 torch，不要继续训练。
- `Embedder checkpoint not found`：确认 `pretrained/embedder.pt` 已上传。
- `No wav files were found in noise root`：确认环境音直接放在 `data/noise/*.wav`。
- 某个 manifest 是 0 行：不要继续训练，先检查 `metadata/subjects.json`、`splits/*.txt` 和 `metadata/preprocess_snr_stats.csv`。
- 显存不足：先用 `DEVICE=cuda:1` 或 `CUDA_VISIBLE_DEVICES=1 DEVICE=cuda:0` 换卡；仍不足时把 `config/platform_gpu.yaml` 里的 `batch_size` 改小。
