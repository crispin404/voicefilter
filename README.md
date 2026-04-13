# VoiceFilter 跨域元音条件鼾声增强

这个仓库仍然保留原始 `VoiceFilter` 相关代码，但已经补充了一套新的主任务流程：

- 条件输入：同一被试的 5 段清醒短元音 `a/e/i/o/u`
- 主输入：30 秒“鼾声 + 环境音”混合音
- 目标输出：对应的干净鼾声音频

当前改造优先跑通以下链路：

1. 数据扫描与被试级 split
2. 音频预处理到 `16k / mono / float32`
3. 5 段元音 embedding 预计算与均值聚合
4. 基于 manifest 的条件增强训练
5. 30 秒整段滑窗推理
6. test 集基础评估

## 依赖安装

```bash
pip install -r requirements.txt
```

## 数据目录要求

原始数据根目录下应为被试文件夹，每个被试至少包含：

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
    合成声_1/
      hs_01_jb_01.wav
      hs_01_km_10.wav
      ...
    info.txt
```

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

## 七步工作流

### 第一步：扫描数据、生成 split、生成 manifest

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

生成增强任务 manifest：

```bash
python scripts/build_manifests.py --subjects metadata/subjects.json --splits-dir splits --output-dir manifests
```

输出位置：

- `metadata/subjects.json`
- `metadata/subject_info.csv`
- `splits/train_subjects.txt`
- `splits/val_subjects.txt`
- `splits/test_subjects.txt`
- `manifests/enhancement_manifest_train.jsonl`
- `manifests/enhancement_manifest_val.jsonl`
- `manifests/enhancement_manifest_test.jsonl`

### 第二步：预处理音频

把所有元音、干净鼾声、混合音统一处理到 `16k / mono`，并把元音 repeat-padding 到 1 秒：

```bash
python scripts/preprocess_audio.py --subjects metadata/subjects.json --processed-root processed --sample-rate 16000 --vowel-seconds 1.0
```

输出位置：

- `processed/vowel/{subject_id}/*.wav`
- `processed/clean/{subject_id}/*.wav`
- `processed/mix/{subject_id}/*.wav`

建议在预处理完成后重新执行一次 `build_manifests.py`，确保 manifest 指向处理后的音频。

```bash
python scripts/build_manifests.py --subjects metadata/subjects.json --splits-dir splits --output-dir manifests --processed-root processed
```

### 第三步：预计算元音 embedding

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

### 第四步：训练条件增强模型

训练脚本会：

- 从 manifest 读取 `mix_path / clean_path / embedding_path`
- 从 30 秒配对音频中随机裁切对齐的 3 秒片段
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

### 第五步：30 秒整段推理

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

### 第六步：基础评估

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

### 第七步：查看结果与复核

建议最终至少检查：

- `subjects.json` 是否完整覆盖所有被试
- split 是否是被试级划分，而不是音频级
- manifest 是否都指向 `processed/` 下的音频
- `processed/embeddings/` 是否为每个被试都有一个 `.npy`
- 训练是否至少能完整跑通一个 epoch
- 推理是否能稳定输出 30 秒 wav

## 新增脚本一览

- `scripts/scan_dataset.py`
- `scripts/build_subject_splits.py`
- `scripts/parse_info_txt.py`
- `scripts/build_manifests.py`
- `scripts/preprocess_audio.py`
- `scripts/precompute_vowel_embeddings.py`
- `scripts/train_enhancement.py`
- `scripts/infer_enhancement_long.py`
- `scripts/evaluate_enhancement.py`

## 与原仓库的关系

本次改造没有重写整个项目，仍然复用了原仓库中的：

- `utils/audio.py` 频谱特征逻辑
- `model/model.py` 的 `VoiceFilter` 主干
- `model/embedder.py` 的 speaker embedder 结构

原始 `generator.py`、`trainer.py`、`inference.py` 仍保留，便于回看或兼容旧流程；新的主任务请优先使用 `scripts/` 下的新脚本。

## License

Apache License 2.0
