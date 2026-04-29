# SnoreFilter 个体化鼾声增强

`SnoreFilter` 是一个面向毕业设计场景的个体化鼾声增强项目。它利用同一被试的 5 段元音 `a/e/i/o/u` 作为身份条件，从“鼾声 + 环境音”的混合音中恢复目标被试的增强鼾声音频。默认会对 5 个元音分别编码后取均值；也可以在配置中切换为只使用单个元音 `a/e/i/o/u` 做对比实验。

仓库当前真实目录名仍然是 `voicefilter`，所以下文中的示例路径继续写成 `F:\voicefilter` 或 Platformax 上的 `.../voicefilter`；这只是为了和你现在的实际运行环境保持一致，项目对外名称统一使用 `SnoreFilter`。

## 项目定位

- 任务目标：做“面向个体”的鼾声增强，而不是通用降噪。
- 条件信息：每个被试提供 5 段元音，默认做均值聚合，也支持切换为单元音条件输入。
- 训练输入：目标被试的元音 embedding + 该被试的鼾声与环境音混合后的频谱。
- 模型输出：与目标被试 clean snore 更接近的增强频谱，再还原为增强音频。

当前保留两条可复现主流程：

- 本地 CPU 小规模跑通：使用 [config/enhancement.yaml](config/enhancement.yaml)
- Platformax GPU 全量训练：使用 [README_PLATFORMAX.md](README_PLATFORMAX.md) 和 [scripts/platformax_run_all.sh](scripts/platformax_run_all.sh)

## 方法概览

SnoreFilter 的核心思路可以概括为：

```text
5 段元音
  -> 提取被试身份 embedding
  -> 与混合鼾声音频的频谱特征拼接
  -> 条件增强网络预测时频 mask
  -> 得到增强频谱
  -> 重建目标被试的增强鼾声音频
```

这里的“身份条件”不是额外标签，而是由元音音频直接提取出来的说话人式 embedding。这样模型学习的不只是“去噪”，而是“尽量保留目标被试的鼾声特征，同时压制环境音干扰”。

## 数据组织

原始被试数据按“每个被试一个文件夹”组织，至少包含 5 段元音、若干段 clean snore，以及 `info.txt`：

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

## 多噪声对比实验

现在支持按 `noise_count=1/2/3` 切换单噪声、双噪声、三噪声数据流。

- `noise_count=1`：保持原行为，直接遍历 `data/noise/*.wav`
- `noise_count=2/3`：读取 `data/noise_splice.txt`
- `noise_splice.txt` 每行一个配方，例如 `jb+nz`、`dpt+km`、`xcq+jb+ye`
- 同一个 `noise_splice.txt` 可以同时放 2 段和 3 段配方；脚本会按当前 `noise_count` 自动筛选
- 原始合成声目录会切到 `合成声_2 / 合成声_3`
- 预处理混合声目录会切到 `processed/mix_2 / processed/mix_3`
- manifest 目录会切到 `manifests_2 / manifests_3`
- 元数据会切到 `metadata/synthesized_mix_metadata_2.jsonl`、`metadata/preprocess_snr_stats_3.csv` 等

配置默认值：

```yaml
data:
  noise_count: 1
```

手动分步命令示例：

```powershell
python scripts/synthesize_mixed_snore.py -c config/enhancement.yaml --noise-root data/noise --noise-count 2 --metadata-path metadata/synthesized_mix_metadata_2.jsonl --metadata-csv metadata/synthesized_mix_metadata_2.csv
python scripts/preprocess_audio.py -c config/enhancement.yaml --subjects metadata/subjects.json --processed-root processed --noise-count 2 --mix-dir-name 合成声_2 --synthesis-metadata metadata/synthesized_mix_metadata_2.jsonl --snr-stats-csv metadata/preprocess_snr_stats_2.csv
python scripts/build_manifests.py -c config/enhancement.yaml --subjects metadata/subjects.json --splits-dir splits --output-dir manifests_2 --processed-root processed --noise-count 2 --mix-dir-name 合成声_2 --snr-stats-csv metadata/preprocess_snr_stats_2.csv
python scripts/train_enhancement.py -c config/enhancement.yaml --noise-count 2 --device cpu
python scripts/evaluate_enhancement.py -c config/enhancement.yaml --noise-count 2 --checkpoint-path outputs/checkpoints/best_metric.pt --output-csv outputs/eval/metrics_2.csv --device cpu
```

`evaluate_enhancement.py --subject-ids-file ...` 现在也会按 `noise_count` 自动切到对应的 `processed/mix[_N]` 数据，不会再误用单噪声目录。

## 全流程逻辑图

项目主流程按下面顺序组织，建议先理解每一步“在做什么”，再执行命令：

```text
scan
  扫描 data/raw，生成被试索引 metadata/subjects.json

split
  生成被试级 train/val/test 划分，避免同一被试同时出现在多个集合

synthesize
  用 clean snore + 环境音按目标 SNR 合成混合音，并记录合成元数据

preprocess
  统一采样率、裁剪长度、整理 vowel / clean / mix 目录，并生成 SNR 统计

manifest
  把训练和评估真正要读取的样本路径整理成 jsonl manifest

embedding
  为每个被试预计算元音 embedding，供训练和评估直接读取

train
  训练 SnoreFilter 模型，同时保存 latest / best_loss / best_metric 三类 checkpoint

evaluate
  在测试集或自定义被试集合上评估增强结果，输出 metrics.csv
```

## 本地 CPU 跑通

这一条主流程适合做功能确认、小样本联调和答辩前的流程自检，不适合替代 Platformax 的全量 GPU 训练。

### 1. 安装依赖

```powershell
cd F:\voicefilter
pip install -r requirements.txt
```

### 2. 生成被试索引

```powershell
python scripts/scan_dataset.py --data-root data/raw --output metadata/subjects.json
```

作用：扫描 `data/raw`，确认有哪些被试、每个被试有哪些元音和鼾声音频。

### 3. 生成被试级划分

```powershell
python scripts/build_subject_splits.py --subjects metadata/subjects.json --output-dir splits --train-count 3 --val-count 1 --test-count 1 --seed 42
```

作用：生成一个小规模可跑通的 `train/val/test` 划分，方便本地 CPU 做流程验证。

如果你手动修改了 `splits/train_subjects.txt`、`splits/val_subjects.txt`、`splits/test_subjects.txt`，必须重新执行 `build_manifests.py`，因为训练和评估真正读取的是 `manifests/*.jsonl`，不是 `splits/*.txt` 本身。

注意：再次运行 `scripts/build_subject_splits.py` 会覆盖现有 `splits/*.txt`。

### 4. 增量合成混合音

```powershell
python scripts/synthesize_mixed_snore.py --subjects metadata/subjects.json --noise-root data/noise --output-subdir 合成声 --target-snr-db 8.0 --seed 42 --metadata-path metadata/synthesized_mix_metadata.jsonl --metadata-csv metadata/synthesized_mix_metadata.csv
```

作用：把 clean snore 与环境音按目标 SNR 合成为混合音，并记录合成元数据。

### 5. 预处理音频

```powershell
python scripts/preprocess_audio.py --subjects metadata/subjects.json --processed-root processed --sample-rate 16000 --vowel-seconds 1.0 --mix-dir-name 合成声 --synthesis-metadata metadata/synthesized_mix_metadata.jsonl --snr-stats-csv metadata/preprocess_snr_stats.csv
```

作用：统一采样率和长度，生成训练/评估直接读取的 `processed/vowel`、`processed/clean`、`processed/mix`。

### 6. 生成 manifest

```powershell
python scripts/build_manifests.py -c config/enhancement.yaml --subjects metadata/subjects.json --splits-dir splits --output-dir manifests --processed-root processed --mix-dir-name 合成声 --snr-stats-csv metadata/preprocess_snr_stats.csv
```

作用：把训练和评估真正需要的样本列表整理成：

```text
manifests/enhancement_manifest_train.jsonl
manifests/enhancement_manifest_val.jsonl
manifests/enhancement_manifest_test.jsonl
```

### 7. 预计算元音 embedding

```powershell
python scripts/precompute_vowel_embeddings.py -c config/enhancement.yaml --subjects metadata/subjects.json --processed-root processed --device cpu
```

作用：先把身份条件提取好，后续训练和评估就不需要每次重复计算元音 embedding。

默认配置：

```yaml
model:
  use_d_vector: true
data:
  vowel_embedding_mode: avg
```

- `true`：使用真实元音 embedding 作为条件输入
- `false`：进入零向量占位消融，不再依赖 `processed/embeddings/*.npy`
- `avg`：分别计算 `a/e/i/o/u` 的 embedding 再取均值，输出到 `processed/embeddings/`
- `a/e/i/o/u`：只使用对应元音，输出到 `processed/embeddings_a/`、`processed/embeddings_o/` 等目录

例如要改成只用 `a`：

```yaml
data:
  vowel_embedding_mode: a
```

切换 `data.vowel_embedding_mode` 后，至少要重新执行 `build_manifests.py` 和 `precompute_vowel_embeddings.py`，这样 manifest 中的 `embedding_path` 和磁盘上的 embedding 目录才会一致。

如果你切换的是 `model.use_d_vector`：

- `true -> false`：不需要重建 manifest，也不需要预计算 embedding，可以直接从训练开始
- `false -> true`：不需要重建 manifest，但训练前要先补跑 `precompute_vowel_embeddings.py`

### 8. CPU 训练

```powershell
python scripts/train_enhancement.py -c config/enhancement.yaml --device cpu
```

默认本地配置是小规模调试配置：`batch_size=2`、`num_epochs=3`、输出目录为 `outputs/`。这条流程的主要目的，是确认数据链路、模型前向和评估脚本都能顺利跑通。

如果你要做 `use_d_vector=true/false` 对比实验，当前不会自动把结果拆分到不同目录。请手动修改 `train.save_dir`，或者在下一组实验开始前先备份 `outputs/` 下的 checkpoint、日志和评估结果。

### 9. 单条推理试听

```powershell
$testSubject = Get-Content splits/test_subjects.txt -Encoding UTF8 | Select-Object -First 1
$mixedFile = Get-ChildItem "processed/mix/$testSubject" -Filter *.wav | Select-Object -First 1 -ExpandProperty FullName
python scripts/infer_enhancement_long.py -c config/enhancement.yaml --checkpoint-path outputs/checkpoints/best_metric.pt --mixed-file "$mixedFile" --vowel-dir "processed/vowel/$testSubject" --output-path "outputs/enhanced_wavs/${testSubject}_demo.wav" --device cpu
```

作用：对一条混合音做滑窗增强，便于快速听感确认。它会自动读取配置里的 `data.vowel_embedding_mode`；默认用 5 个元音均值，单元音模式下只会使用对应的那个元音文件。若 `model.use_d_vector=false`，脚本会自动改用零向量条件，`--vowel-dir` 只保留命令行兼容形状，不再参与编码。

### 10. 测试集评估

```powershell
python scripts/evaluate_enhancement.py -c config/enhancement.yaml --checkpoint-path outputs/checkpoints/best_metric.pt --output-csv outputs/eval/metrics.csv --device cpu
```

作用：在测试集上输出逐条样本指标和终端汇总，便于比较不同 checkpoint 或配置。

## 什么时候需要重跑哪些步骤

- 只新增或删除 `data/noise/*.wav`：重跑 `synthesize -> preprocess -> manifest`
- 新增或删除原始 clean snore，但被试集合没变：先重跑 `scan`，再重跑 `synthesize -> preprocess -> manifest`
- 被试集合本身发生变化：重跑 `scan -> split -> synthesize -> preprocess -> manifest`
- 修改 `data.vowel_embedding_mode`：至少重跑 `manifest -> embedding`
- 修改 `model.use_d_vector` 为 `false`：不需要重跑 `manifest`，可以跳过 `embedding`
- 修改 `model.use_d_vector` 为 `true`：不需要重跑 `manifest`，但要先补跑 `embedding`
- 想彻底重建混合音和预处理数据：在 `synthesize_mixed_snore.py` 和 `preprocess_audio.py` 里加 `--force`
- 只手动修改了 `splits/*.txt`：至少重跑 `build_manifests.py`

对应的环境音增量同步命令如下：

```powershell
python scripts/synthesize_mixed_snore.py --subjects metadata/subjects.json --noise-root data/noise --output-subdir 合成声 --target-snr-db 8.0 --seed 42 --metadata-path metadata/synthesized_mix_metadata.jsonl --metadata-csv metadata/synthesized_mix_metadata.csv
python scripts/preprocess_audio.py --subjects metadata/subjects.json --processed-root processed --sample-rate 16000 --vowel-seconds 1.0 --mix-dir-name 合成声 --synthesis-metadata metadata/synthesized_mix_metadata.jsonl --snr-stats-csv metadata/preprocess_snr_stats.csv
python scripts/build_manifests.py -c config/enhancement.yaml --subjects metadata/subjects.json --splits-dir splits --output-dir manifests --processed-root processed --mix-dir-name 合成声 --snr-stats-csv metadata/preprocess_snr_stats.csv
```

## Checkpoint 与评估说明

训练会同时维护三类 checkpoint：

- `outputs/checkpoints/latest.pt`：用于断点续训
- `outputs/checkpoints/best_loss.pt`：按 `val_loss` 最优保存
- `outputs/checkpoints/best_metric.pt`：按验证集整段增强指标最优保存，推理和评估默认优先使用它

如果你更关注训练速度，可以在 [config/enhancement.yaml](config/enhancement.yaml) 中设置：

```yaml
train:
  enable_best_metric_eval: false
```

这样训练仍会保存 `latest.pt` 和 `best_loss.pt`，但不会再生成新的 `best_metric.pt`。此时后续推理和评估请改用 `outputs/checkpoints/best_loss.pt`。

`metrics.csv` 核心指标包括：

- `input_snr`：增强前输入的 SNR
- `snr_improvement`：增强前后 SNR 提升量
- `enhanced_snr`：增强后的绝对 SNR
- `input_si_sdr`：增强前输入的 SI-SDR
- `si_sdr`：增强后的 SI-SDR
- `si_sdr_improvement`：增强前后 SI-SDR 提升量
- `mag_l1`：增强频谱与 clean 频谱的 L1 误差
- `clean_active_ratio`：clean snore 在评估窗口中的活动占比

终端汇总还会打印 `all_count`、`active_count`、`zero_active_count`、`negative_improvement_rate` 和 `negative_count`，用于辅助判断模型是否在一部分样本上“越增强越差”。

## 可选的数据审计工具

如果你想先筛查原始 clean snore 的质量，可以运行：

```powershell
python scripts/audit_clean_snores.py --data-root data/raw
```

也可以基于 `subjects.json` 运行：

```powershell
python scripts/audit_clean_snores.py --subjects metadata/subjects.json
```

这个脚本只生成报告，不会自动删除音频，也不接入一键全流程。输出为 `metadata/snore_audit.csv` 和 `metadata/snore_audit.json`，当前只基于 `active_ratio`、`rms`、`clipping_ratio` 评估，并重点给出 `quality_score`、`risk_flags` 和 `decision` 供人工筛样。

## 关键产物

本地 CPU 流程跑完后，建议优先检查这些产物：

```text
metadata/subjects.json
splits/train_subjects.txt
splits/val_subjects.txt
splits/test_subjects.txt
processed/vowel/
processed/clean/
processed/mix/
processed/embeddings/
processed/embeddings_<vowel>/
manifests/enhancement_manifest_train.jsonl
manifests/enhancement_manifest_val.jsonl
manifests/enhancement_manifest_test.jsonl
outputs/checkpoints/latest.pt
outputs/checkpoints/best_loss.pt
outputs/checkpoints/best_metric.pt
outputs/logs/train.log
outputs/eval/metrics.csv
```

## 当前主流程依赖的脚本

- `scripts/scan_dataset.py`：扫描被试数据并生成 `subjects.json`
- `scripts/build_subject_splits.py`：生成被试级 train/val/test 划分
- `scripts/synthesize_mixed_snore.py`：按目标 SNR 合成混合音
- `scripts/preprocess_audio.py`：整理 vowel / clean / mix 预处理数据
- `scripts/build_manifests.py`：生成训练和评估实际读取的 manifest
- `scripts/precompute_vowel_embeddings.py`：预计算元音 embedding
- `scripts/train_enhancement.py`：训练 SnoreFilter 模型
- `scripts/infer_enhancement_long.py`：对单条混合音做滑窗增强
- `scripts/evaluate_enhancement.py`：输出测试集或自定义集合的评估指标
- `scripts/audit_clean_snores.py`：对原始 clean snore 做质量审计
- `scripts/check_platform_env.py`：检查 Platformax 环境、CUDA 和数据可用性
- `scripts/platformax_run_all.sh`：Platformax 一键全流程脚本

Platformax GPU 全量训练说明见 [README_PLATFORMAX.md](README_PLATFORMAX.md)。
