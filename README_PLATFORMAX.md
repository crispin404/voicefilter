# SnoreFilter Platformax GPU 运行说明

本说明用于在 Platformax JupyterLab 终端运行 `SnoreFilter` 的全量 GPU 流程。项目对外名称已经统一为 `SnoreFilter`，但仓库当前真实目录名仍是 `voicefilter`，因此下面的路径示例继续保留真实可运行写法。

推荐项目目录示例：

```bash
/opt/data/private/mel-data/毕设/voicefilter
```

如果你的项目实际放在其他位置，也没有关系，关键是后续命令都在“项目根目录”下执行，并保持目录结构一致。

## 环境与数据准备

### 目录结构

在项目下准备数据和权重：

```text
voicefilter/
  data/
    raw/        # 完整原始被试数据，每个被试一个文件夹
    noise/      # 环境音 wav，例如 jb.wav / km.wav / qm.wav / ye.wav
  pretrained/
    embedder.pt # 预训练 speaker embedder
```

`pretrained/` 和 `data/` 默认不会进 git，需要单独上传到平台。

### 进入项目目录

```bash
cd /opt/data/private/mel-data/毕设/voicefilter
```

### 安装平台依赖

不要在 Platformax 上运行 `pip install -r requirements.txt`。平台镜像通常已经带 CUDA 版 PyTorch，本项目在平台上只需要补齐非 torch 依赖。

推荐使用带 `--system-site-packages` 的虚拟环境，让项目环境直接继承平台镜像中的 CUDA 版 torch：

```bash
deactivate 2>/dev/null || true
rm -rf .venv
python -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-platform.txt
```

不要在这个环境里单独 `pip install torch`。平台镜像里的 torch 往往是和 CUDA 版本绑定好的，直接从 PyPI 重新装很容易不匹配。

安装后务必确认当前环境拿到的是 CUDA 版 torch：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

最后一行必须是 `True`，否则先不要继续训练。

如果遇到 `scipy 1.6.3 requires numpy<1.23` 这类提示，不要随意把 `numpy` 改高。当前 `requirements-platform.txt` 使用 `numpy==1.22.4`，这是和平台旧版 SciPy 兼容的稳定组合。

如果 `python -m pip` 不可用，可退回：

```bash
pip3 install -r requirements-platform.txt
```

### 检查 CUDA、数据和 embedder

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

如果这一步失败，先不要往后跑。

### 可选：先做 clean snore 审计

```bash
python scripts/audit_clean_snores.py --data-root data/raw
```

也可以基于 `subjects.json` 运行：

```bash
python scripts/audit_clean_snores.py --subjects metadata/subjects.json
```

这个脚本只生成 `metadata/snore_audit.csv` 和 `metadata/snore_audit.json` 两份报告，不会自动删除音频，也不接入一键全流程。当前主要基于 `active_ratio`、`rms`、`clipping_ratio` 做质量筛查。

## 一键全流程

如果环境、数据和权重都确认正常，可以直接运行一键脚本。

默认使用 `cuda:1`：

```bash
bash scripts/platformax_run_all.sh
```

改用 GPU0：

```bash
DEVICE=cuda:0 bash scripts/platformax_run_all.sh
```

如果 GPU0 被占用，最稳妥的方式是只暴露物理 GPU1：

```bash
CUDA_VISIBLE_DEVICES=1 DEVICE=cuda:0 bash scripts/platformax_run_all.sh
```

一键脚本会依次完成：

```text
check -> scan -> split -> synthesize -> preprocess -> manifest -> embedding(conditional) -> train -> evaluate
```

一键脚本当前默认：

- 评估权重：`outputs/platformax/checkpoints/best_metric.pt`
- 评估结果：`outputs/platformax/eval/metrics.csv`
- 默认不导出增强 wav
- 划分和混合音裁剪随机种子：`SEED=42`
- 训练随机种子：来自 `config/platform_gpu.yaml` 中的 `train.seed`
- d-vector 开关：来自 `config/platform_gpu.yaml` 中的 `model.use_d_vector`
- 元音 embedding 模式：来自 `config/platform_gpu.yaml` 中的 `data.vowel_embedding_mode`

默认配置是：

```yaml
model:
  use_d_vector: true
data:
  vowel_embedding_mode: avg
```

- `use_d_vector: true`：正常使用真实元音 embedding
- `use_d_vector: false`：进入零向量占位消融，一键脚本会跳过 embedder 检查和 embedding 预计算
- `avg`：对 `a/e/i/o/u` 分别编码后取均值，输出到 `processed/embeddings/`
- `a/e/i/o/u`：只使用对应元音，输出到 `processed/embeddings_a/`、`processed/embeddings_o/` 等目录

如果你想做 `d-vector` 或单元音对比实验，直接修改 `config/platform_gpu.yaml`，然后重新跑一键脚本即可。

- 只改 `model.use_d_vector`：不需要重建 manifest；`true -> false` 时可直接跳过 embedding 预计算
- 改 `data.vowel_embedding_mode`：至少要从 `build_manifests.py` 和 `precompute_vowel_embeddings.py` 开始往后跑

如果你想同时导出增强 wav：

```bash
EVAL_SAVE_WAVS=1 bash scripts/platformax_run_all.sh
```

如果你想手动指定评估 checkpoint：

```bash
EVAL_CHECKPOINT_PATH=outputs/platformax/checkpoints/best_loss.pt bash scripts/platformax_run_all.sh
```

## 分步全流程

如果你想一步一步排查问题，建议按下面顺序执行。前一步失败时先停，不要继续往后跑。

### 1. 扫描 `data/raw`

```bash
python scripts/scan_dataset.py \
  --data-root data/raw \
  --output metadata/subjects.json
```

作用：生成被试索引 `metadata/subjects.json`。

### 2. 生成被试级划分

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

如果你手动修改了 `splits/*.txt`，不要直接重新跑一键脚本；至少要从 `build_manifests.py` 开始往后执行，因为训练和评估真正读取的是 `manifests/*.jsonl`。

注意：再次运行 `scripts/build_subject_splits.py` 或 `bash scripts/platformax_run_all.sh`，都会覆盖你手改的 `splits/*.txt`。

### 3. 增量生成混合音

```bash
python scripts/synthesize_mixed_snore.py \
  --subjects metadata/subjects.json \
  --noise-root data/noise \
  --output-subdir 合成声 \
  --target-snr-db 5.0 \
  --seed 42 \
  --metadata-path metadata/synthesized_mix_metadata.jsonl \
  --metadata-csv metadata/synthesized_mix_metadata.csv
```

作用：把 clean snore 和环境音按目标 SNR 合成为混合音，并写出合成元数据。

### 4. 预处理音频

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

作用：统一采样率、整理 `processed/vowel`、`processed/clean`、`processed/mix`，并记录预处理 SNR 统计。

### 5. 生成 manifest

```bash
python scripts/build_manifests.py \
  -c config/platform_gpu.yaml \
  --subjects metadata/subjects.json \
  --splits-dir splits \
  --output-dir manifests \
  --processed-root processed \
  --mix-dir-name 合成声 \
  --snr-stats-csv metadata/preprocess_snr_stats.csv
```

建议确认每个 split 都不是 0：

```bash
wc -l manifests/*.jsonl
```

### 6. 预计算元音 embedding

使用 GPU0：

```bash
python scripts/precompute_vowel_embeddings.py \
  -c config/platform_gpu.yaml \
  --subjects metadata/subjects.json \
  --processed-root processed \
  --device cuda:0
```

使用 GPU1：

```bash
python scripts/precompute_vowel_embeddings.py \
  -c config/platform_gpu.yaml \
  --subjects metadata/subjects.json \
  --processed-root processed \
  --device cuda:1
```

这个脚本会自动读取 `data.vowel_embedding_mode`，并把结果写到匹配的目录：

- `avg` -> `processed/embeddings/`
- `a/e/i/o/u` -> `processed/embeddings_<vowel>/`

如果 `model.use_d_vector=false`，这一整步可以跳过；一键脚本也会自动跳过。

### 7. GPU 训练

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

当前平台训练配置的要点是：

- 最多训练 `30` 轮，`early_stop_patience=10`
- 优化器为 `AdamW`
- 学习率调度器为 `ReduceLROnPlateau`
- 损失函数为 `mag_l1 + 0.1 * mask_l1`
- 启用了 active crop，优先抽取有鼾声活动的 5 秒片段
- 输出目录为 `outputs/platformax`

做 `use_d_vector=true/false` 对比实验时，脚本不会自动把结果拆分到不同目录。请手动修改 `train.save_dir`，或者在开始下一组实验前先备份 `outputs/platformax`。

如果你更关心训练速度，可以在 [config/platform_gpu.yaml](config/platform_gpu.yaml) 里设置：

```yaml
train:
  enable_best_metric_eval: false
```

这样训练仍会保存 `latest.pt` 和 `best_loss.pt`，但不会再跑整段 `best_metric` 验证，也不会生成新的 `best_metric.pt`。

如果这是从更旧的训练配置迁移过来的实验，不建议直接拿旧 `latest.pt` 续训，因为损失函数、优化器和训练裁剪策略都已经变过。

### 8. 断点续训

如果训练中断且 `latest.pt` 已存在：

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

若 `model.use_d_vector=false`，评估会自动改用零向量条件，不再依赖 `processed/embeddings/*.npy` 或 `pretrained/embedder.pt`。

## 评估与结果解读

### 默认测试集评估

使用 GPU0：

```bash
python scripts/evaluate_enhancement.py \
  -c config/platform_gpu.yaml \
  --checkpoint-path outputs/platformax/checkpoints/best_loss.pt \
  --output-csv outputs/platformax/eval/metrics_a_1.csv \
  --no-save-wavs \
  --device cuda:0
```

如果使用 `CUDA_VISIBLE_DEVICES=1`：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_enhancement.py \
  -c config/platform_gpu.yaml \
  --checkpoint-path outputs/platformax/checkpoints/best_metric.pt \
  --output-csv outputs/platformax/eval/metrics.csv \
  --no-save-wavs \
  --device cuda:0
```

### `best_metric.pt` 和 `best_loss.pt` 怎么选

- `best_metric.pt`：默认首选，用于更贴近最终增强效果的评估和试听
- `best_loss.pt`：当你关闭了 `enable_best_metric_eval`，或者只想按验证损失选模型时使用
- `latest.pt`：只用于断点续训，不建议直接拿来做正式结果汇报

如果你把 `train.enable_best_metric_eval` 设成了 `false`，后续评估请改用：

```bash
--checkpoint-path outputs/platformax/checkpoints/best_loss.pt
```

### `--no-save-wavs` 的含义

- 加 `--no-save-wavs`：只输出 `metrics.csv`，评估更快，适合批量对比
- 去掉 `--no-save-wavs`：除了 `metrics.csv`，还会把增强音频写到 `outputs/platformax/eval/enhanced_wavs/`

如果你想导出增强 wav，可以这样运行：

```bash
python scripts/evaluate_enhancement.py \
  -c config/platform_gpu.yaml \
  --checkpoint-path outputs/platformax/checkpoints/best_metric.pt \
  --output-csv outputs/platformax/eval/metrics.csv \
  --save-wavs-dir outputs/platformax/eval/enhanced_wavs \
  --device cuda:1
```

### 自定义多人评估：`splits/metrics_test.txt`

如果你只想评估指定的 1 个或多个被试，不需要改 `splits/test_subjects.txt`，也不需要先重建 manifest。只要新建：

```text
splits/metrics_test.txt
```

文件格式是一行一个 `subject_id`，例如：

```text
subject_a
subject_b
subject_c
```

运行命令：

```bash
python scripts/evaluate_enhancement.py \
  -c config/platform_gpu.yaml \
  --checkpoint-path outputs/platformax/checkpoints/best_metric.pt \
  --subject-ids-file splits/metrics_test.txt \
  --subjects metadata/subjects.json \
  --output-csv outputs/platformax/eval/metrics_test.csv \
  --no-save-wavs \
  --device cuda:1
```

这条路径会直接从 `metadata/subjects.json` 和现有 `processed/` 数据动态构建评估样本，不依赖 `splits/test_subjects.txt`，也不需要重建 manifest。

### 主要评估指标

`metrics.csv` 每条样本包含：

- `input_snr`：增强前混合音相对 clean 的 SNR
- `snr_improvement`：增强前后 SNR 提升量
- `enhanced_snr`：增强后的绝对 SNR
- `input_si_sdr`：增强前输入的 SI-SDR
- `si_sdr`：增强后的 SI-SDR
- `si_sdr_improvement`：增强前后 SI-SDR 提升量
- `mag_l1`：增强频谱与 clean 频谱的 L1 误差
- `clean_active_ratio`：clean snore 有效活动窗口占比

终端还会打印 `all_count`、`active_count`、`zero_active_count`、`negative_improvement_rate` 和 `negative_count`，用于辅助判断模型在哪些样本上可能出现负提升。

## 环境音或 clean snore 变更后怎么更新

以后如果你新增 `data/noise/*.wav`、删除效果不好的环境音，或者手动增删某个被试目录里的原始 clean snore，不需要从头全量重跑。

如果你修改的是 `config/platform_gpu.yaml` 里的 `data.vowel_embedding_mode`，至少要重新执行 `build_manifests.py` 和 `precompute_vowel_embeddings.py`，因为 manifest 里的 `embedding_path` 会跟着模式切换。

如果你修改的是 `model.use_d_vector`：

- 切到 `false`：不需要重建 manifest，也不需要预计算 embedding
- 切回 `true`：不需要重建 manifest，但训练前要先补跑 `precompute_vowel_embeddings.py`

对于 clean snore 变更，先重新扫描一次：

```bash
python scripts/scan_dataset.py \
  --data-root data/raw \
  --output metadata/subjects.json
```

如果被试集合本身没有变化，`splits/` 不需要重建；只需要继续执行：

```bash
python scripts/synthesize_mixed_snore.py \
  --subjects metadata/subjects.json \
  --noise-root data/noise \
  --output-subdir 合成声 \
  --target-snr-db 5.0 \
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
  -c config/platform_gpu.yaml \
  --subjects metadata/subjects.json \
  --splits-dir splits \
  --output-dir manifests \
  --processed-root processed \
  --mix-dir-name 合成声 \
  --snr-stats-csv metadata/preprocess_snr_stats.csv
```

脚本会自动补新增文件、删除已剔除环境音对应的旧文件，并跳过没有变化的样本。需要彻底重跑时，在合成和预处理命令里加 `--force`。

## 最后检查

确认模型和评估结果存在：

```bash
ls -lh outputs/platformax/checkpoints/best_metric.pt
ls -lh outputs/platformax/checkpoints/best_loss.pt
ls -lh outputs/platformax/eval/metrics.csv
```

确认训练日志里确实在用 GPU：

```bash
grep -E "Using device|CUDA device name|epoch=" outputs/platformax/logs/train.log
```

你应该看到类似：

```text
Using device: cuda:1
CUDA device name: NVIDIA GeForce RTX 4090
epoch=1 train_loss=... val_loss=... train_mag_l1=... val_mag_l1=... train_mask_l1=... val_mask_l1=... lr=... stale_epochs=...
epoch=1 train_loss=... val_loss=... val_avg_si_sdr_improvement=... val_avg_si_sdr=... val_negative_count=... saved_best_loss=... saved_best_metric=... stale_epochs=...
```

## 常见问题

- `CUDA available: False`：当前 Python 没拿到 CUDA 版 torch，先不要继续训练
- `Embedder checkpoint not found`：确认 `pretrained/embedder.pt` 已上传
- `No wav files were found in noise root`：确认环境音直接放在 `data/noise/*.wav`
- 某个 manifest 是 0 行：先检查 `metadata/subjects.json`、`splits/*.txt` 和 `metadata/preprocess_snr_stats.csv`
- 显存不足：先尝试 `DEVICE=cuda:1` 或 `CUDA_VISIBLE_DEVICES=1 DEVICE=cuda:0`，仍不足时再调小 `config/platform_gpu.yaml` 里的 `batch_size`
