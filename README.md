# VoiceFilter 跨域元音条件鼾声增强

这个仓库保留了原始 `VoiceFilter` 的主体代码，并补充了一套面向当前任务的数据流水线：

- 条件输入：同一被试的 5 段清醒短元音 `a/e/i/o/u`
- 主输入：30 秒“鼾声 + 环境音”混合音
- 目标输出：对应目标人的干净鼾声音频

当前推荐流程如下：

1. 扫描被试目录并生成 `subjects.json`
2. 按被试级生成 `train / val / test` split
3. 用固定 SNR 规则重新合成 `合成声_2`
4. 对元音、clean、mix 做统一预处理
5. 导出并按需过滤 SNR 统计
6. 预计算 5 段元音 embedding
7. 基于 manifest 进行条件增强训练
8. 执行长音频推理与基础评估

## 依赖安装

```bash
pip install -r requirements.txt
```

## 数据目录要求

原始数据根目录下应为被试文件夹。每个被试至少应包含：

```text
data_root/
  2022_09_06_张三/
    元音/
      a1_1.wav
      e1_1.wav
      i1_1.wav
      o1_1.wav
      u1_1.wav
    鼾声/
      hs_01_1.wav
      ...
      hs_01_10.wav
    info.txt
```

如果已经存在旧版混合音，也可以保留：

```text
  2022_09_06_张三/
    合成声_1/
      hs_01_jb_01.wav
      ...
```

使用新脚本后，会在每个被试目录下额外生成：

```text
  2022_09_06_张三/
    合成声_2/
      hs_01_jb_01.wav
      hs_01_km_01.wav
      hs_01_qm_01.wav
      hs_01_ye_01.wav
      ...
```

说明：

- `scan_dataset.py` 现在不再把 `合成声_1` 视为必需目录。
- 新流程默认优先使用 `合成声_2/`，不会覆盖已有的 `合成声_1/`。
- 环境音目录采用平铺结构，文件名去掉扩展名后即为 `noise_type`，例如 `jb.wav -> jb`。

命名规则默认按以下方式匹配：

- 混合音：`hs_{inner_id}_{noise_type}_{snore_index}.wav`
- 干净鼾声：`hs_{inner_id}_{snore_index}.wav`
- 例如：`hs_01_jb_03.wav -> hs_01_3.wav`

## 配置文件

主配置文件为 [config/enhancement.yaml](./config/enhancement.yaml)。

关键字段如下：

- `data.manifest_train / manifest_val / manifest_test`：训练、验证、测试 manifest
- `audio.sample_rate`：统一采样率，默认 `16000`
- `data.segment_seconds`：训练随机裁切长度，默认 `3.0`
- `data.inference_window_seconds`：长音频推理窗口，默认 `3.0`
- `data.inference_hop_seconds`：长音频推理 hop，默认 `1.5`
- `embedder.emb_dim`：元音 embedding 维度，默认 `256`
- `model.use_embedding_adapter`：是否启用轻量条件适配器
- `train.batch_size / learning_rate / num_epochs / save_dir`：训练超参数和输出目录

## 八步工作流

### 第一步：扫描数据并生成 split

扫描所有被试目录并生成 `metadata/subjects.json`：

```bash
python scripts/scan_dataset.py --data-root F:/your_raw_dataset --output metadata/subjects.json
```

按被试级生成 `train / val / test`：

```bash
python scripts/build_subject_splits.py --subjects metadata/subjects.json --output-dir splits --seed 42
```

可选解析 `info.txt` 到 CSV：

```bash
python scripts/parse_info_txt.py --subjects metadata/subjects.json --output metadata/subject_info.csv
```

输出位置：

- `metadata/subjects.json`
- `metadata/subject_info.csv`
- `splits/train_subjects.txt`
- `splits/val_subjects.txt`
- `splits/test_subjects.txt`

### 第二步：重新合成 `合成声_2`

用固定 SNR 重新为每个被试生成新的“鼾声 + 环境音”混合音。默认目标 SNR 为 `8.0 dB`，表示鼾声比环境音高 `8 dB`。

```bash
python scripts/synthesize_mixed_snore.py --subjects metadata/subjects.json --noise-root F:/环境音 --output-subdir 合成声_2 --target-snr-db 8.0 --seed 42 --metadata-path metadata/synthesized_mix_metadata.jsonl --metadata-csv metadata/synthesized_mix_metadata.csv
```

脚本行为：

- 读取 clean snore 和环境音
- 重采样到统一采样率并转为单声道
- 如果 noise 更长，则随机裁切到与 clean 等长
- 如果 noise 更短，则 repeat 到足够长度后再裁切
- 用 RMS 缩放 noise 到目标 SNR
- `mix = clean + scaled_noise`
- 如有必要，整体缩放混合结果以避免削波
- 输出合成元数据，便于后续预处理和质量追踪

输出位置：

- `{subject_dir}/合成声_2/*.wav`
- `metadata/synthesized_mix_metadata.jsonl`
- `metadata/synthesized_mix_metadata.csv`

### 第三步：预处理音频

把元音、clean、mix 统一处理到 `16k / mono / float32`。

元音处理规则：

- 单独做 peak normalize
- repeat-pad 到固定时长，默认 `1.0` 秒

clean 和 mix 处理规则：

- 不再分别独立 peak normalize
- 对每一对严格对齐的 clean + mix 使用同一个缩放系数做成对归一化
- 为每条 mix 额外生成对应的 pair-clean 文件
- 导出每对样本的 SNR 统计 CSV
- 对长度异常、SNR 异常等情况输出 warning

如果你要处理新生成的 `合成声_2`，推荐这样运行：

```bash
python scripts/preprocess_audio.py --subjects metadata/subjects.json --processed-root processed --sample-rate 16000 --vowel-seconds 1.0 --mix-dir-name 合成声_2 --synthesis-metadata metadata/synthesized_mix_metadata.jsonl --snr-stats-csv metadata/preprocess_snr_stats.csv
```

输出位置：

- `processed/vowel/{subject_id}/*.wav`
- `processed/clean/{subject_id}/{mix_stem}_clean.wav`
- `processed/mix/{subject_id}/{mix_filename}.wav`
- `metadata/preprocess_snr_stats.csv`

说明：

- `processed/clean/` 中的 clean 文件现在是“按 mix 配对”的 clean，而不是简单复制原始 clean。
- 这样可以确保同一对 clean / mix 在预处理后仍保持严格一致的幅度关系。

### 第四步：生成 manifest 并按 SNR 过滤

预处理完成后，再生成训练所需的 manifest。

如果使用 `合成声_2` 和新的 SNR 统计，可以这样运行：

```bash
python scripts/build_manifests.py --subjects metadata/subjects.json --splits-dir splits --output-dir manifests --processed-root processed --mix-dir-name 合成声_2 --snr-stats-csv metadata/preprocess_snr_stats.csv --min-snr-db 7.0 --max-snr-db 9.0
```

如果不想做 SNR 过滤，可以去掉 `--min-snr-db` 和 `--max-snr-db`。

输出位置：

- `manifests/enhancement_manifest_train.jsonl`
- `manifests/enhancement_manifest_val.jsonl`
- `manifests/enhancement_manifest_test.jsonl`

### 第五步：预计算元音 embedding

若你有原始 speaker embedder 权重，可通过 `--embedder-path` 指定；没有也可以直接运行，脚本会使用随机初始化接口完成流程。

```bash
python scripts/precompute_vowel_embeddings.py -c config/enhancement.yaml --subjects metadata/subjects.json --processed-root processed --output-dir processed/embeddings
```

带预训练 embedder 的示例：

```bash
python scripts/precompute_vowel_embeddings.py -c config/enhancement.yaml --subjects metadata/subjects.json --processed-root processed --output-dir processed/embeddings --embedder-path path/to/embedder.pt
```

输出位置：

- `processed/embeddings/{subject_id}.npy`

### 第六步：训练条件增强模型

训练脚本会：

- 从 manifest 读取 `mix_path / clean_path / embedding_path`
- 从 30 秒配对音频中随机裁切对齐的 `3` 秒片段
- 用原始 `VoiceFilter` 主干预测 mask
- 默认使用频谱 `L1` 损失

训练命令：

```bash
python scripts/train_enhancement.py -c config/enhancement.yaml
```

断点续训：

```bash
python scripts/train_enhancement.py -c config/enhancement.yaml --checkpoint-path outputs/checkpoints/latest.pt
```

输出位置：

- `outputs/checkpoints/latest.pt`
- `outputs/checkpoints/best.pt`
- `outputs/logs/train.log`
- `outputs/logs/` 下的 TensorBoard 标量

### 第七步：30 秒整段推理

输入某个被试的 5 段元音和一条 30 秒混合音，脚本会：

- 先在线计算 5 段元音的聚合 embedding
- 再对混合音按 `3s window / 1.5s hop` 滑窗推理
- 最后 overlap-add 重建整段增强音频

```bash
python scripts/infer_enhancement_long.py -c config/enhancement.yaml --checkpoint-path outputs/checkpoints/best.pt --mixed-file processed/mix/{subject_id}/hs_01_jb_03.wav --vowel-dir processed/vowel/{subject_id} --output-path outputs/enhanced_wavs/{subject_id}_hs_01_jb_03.wav
```

如果需要与训练时一致的 embedder 权重：

```bash
python scripts/infer_enhancement_long.py -c config/enhancement.yaml --checkpoint-path outputs/checkpoints/best.pt --mixed-file processed/mix/{subject_id}/hs_01_jb_03.wav --vowel-dir processed/vowel/{subject_id} --embedder-path path/to/embedder.pt --output-path outputs/enhanced_wavs/{subject_id}_hs_01_jb_03.wav
```

输出位置：

- `outputs/enhanced_wavs/*.wav`

### 第八步：基础评估与复核

在 test manifest 上批量推理并输出 CSV：

```bash
python scripts/evaluate_enhancement.py -c config/enhancement.yaml --checkpoint-path outputs/checkpoints/best.pt --output-csv outputs/eval/metrics.csv
```

如果 test manifest 没有预计算 embedding，也可以加上 `--embedder-path` 让脚本回退到在线计算。

输出位置：

- `outputs/eval/metrics.csv`
- `outputs/eval/enhanced_wavs/*.wav`

默认统计：

- `SDR`
- `SI-SDR`
- `SNR improvement`
- `mag_l1`

建议最终至少检查：

- `subjects.json` 是否完整覆盖所有被试
- split 是否是被试级划分，而不是音频级
- `合成声_2/` 是否为每个被试生成了 `10 x 噪声数` 条混合音
- `preprocess_snr_stats.csv` 中是否存在异常 warning
- manifest 是否都指向 `processed/` 下的音频
- `processed/embeddings/` 是否为每个被试都有一个 `.npy`
- 训练是否至少能完整跑通一个 epoch
- 推理是否能稳定输出 30 秒 wav

## 新增脚本一览

- `scripts/scan_dataset.py`
- `scripts/build_subject_splits.py`
- `scripts/parse_info_txt.py`
- `scripts/synthesize_mixed_snore.py`
- `scripts/preprocess_audio.py`
- `scripts/build_manifests.py`
- `scripts/precompute_vowel_embeddings.py`
- `scripts/train_enhancement.py`
- `scripts/infer_enhancement_long.py`
- `scripts/evaluate_enhancement.py`

## 与原仓库的关系

本次改造没有重写整个项目，仍然复用了原仓库中的：

- `utils/audio.py` 的频谱特征逻辑
- `model/model.py` 的 `VoiceFilter` 主干
- `model/embedder.py` 的 speaker embedder 结构

原始 `generator.py`、`trainer.py`、`inference.py` 仍然保留，便于回看或兼容旧流程；新的主任务建议优先使用 `scripts/` 下的脚本。

## License

Apache License 2.0
