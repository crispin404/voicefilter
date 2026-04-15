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

## 五人数据 CPU 跑通

这一节专门对应你当前这次实验：

- 数据根目录固定为 `F:\五人跑通`
- 环境音目录固定为 `F:\环境音`
- 需要重新生成 `合成声_2`
- 数据划分固定为 `3 / 1 / 1`
- 按 CPU 路线跑通

为了少改配置，仓库已经把 [config/enhancement.yaml](./config/enhancement.yaml) 的默认训练参数调整为：

- `train.batch_size: 2`
- `train.num_epochs: 3`

下面的步骤按“小白照抄”来写。每一步都包含：

- 要执行的命令
- 你应该看到什么
- 失败时怎么判断

### 第 0 步：激活你自己的兼容 Python 虚拟环境

先打开 PowerShell，再进入项目目录：

```powershell
cd F:\voicefilter
```

然后执行你自己的 venv 激活命令。这里不写死，是因为每个人路径不一样。激活后立刻检查 Python 版本：

```powershell
python --version
```

你应该看到：

- `Python 3.9.x`
- 或 `Python 3.10.x`
- 或 `Python 3.11.x`

失败时怎么判断：

- 如果还是 `Python 3.12.x`，先不要继续，说明你还没切到兼容环境。
- 如果 `python` 命令都找不到，也先不要继续。

接着安装依赖：

```powershell
pip install -r requirements.txt
```

你应该看到：

- `Successfully installed ...`
- 或大部分依赖已经存在，只补装少量包

失败时怎么判断：

- 如果报 `No matching distribution found`，通常是 Python 版本不兼容。
- 如果报 `torch` 安装失败，也先不要继续，直接把原始报错发出来。

### 第 1 步：扫描 5 个人的数据

```powershell
python scripts/scan_dataset.py --data-root "F:\五人跑通" --output metadata/subjects.json
```

你应该看到：

- `Scanned 5 subjects`
- `Saved to ...\metadata\subjects.json`

产物位置：

- `metadata/subjects.json`

失败时怎么判断：

- 如果不是 `5 subjects`，说明目录结构有缺失，先检查每个被试下面是否都有 `元音/鼾声/info.txt`。

### 第 2 步：生成 3/1/1 的 train / val / test 划分

```powershell
python scripts/build_subject_splits.py --subjects metadata/subjects.json --output-dir splits --train-count 3 --val-count 1 --test-count 1 --seed 42
```

你应该看到三行输出，分别对应：

- `train: 3 subjects`
- `val: 1 subjects`
- `test: 1 subjects`

产物位置：

- `splits/train_subjects.txt`
- `splits/val_subjects.txt`
- `splits/test_subjects.txt`

失败时怎么判断：

- 如果报 `Not enough subjects`，通常是上一步扫描出的有效被试数不够 5。

### 第 3 步：用 `F:\环境音` 重新生成 `合成声_2`

```powershell
python scripts/synthesize_mixed_snore.py --subjects metadata/subjects.json --noise-root "F:\环境音" --output-subdir 合成声_2 --target-snr-db 8.0 --seed 42 --metadata-path metadata/synthesized_mix_metadata.jsonl --metadata-csv metadata/synthesized_mix_metadata.csv
```

你应该看到：

- 进度条正常走完
- 最后输出 `Synthesized ... mixtures for 5 subjects`

产物位置：

- 每个被试目录下的 `合成声_2\*.wav`
- `metadata/synthesized_mix_metadata.jsonl`
- `metadata/synthesized_mix_metadata.csv`

失败时怎么判断：

- 如果报 `No wav files were found in noise root`，先检查 `F:\环境音` 下是否还是 `jb.wav/km.wav/qm.wav/ye.wav`
- 如果报某个被试的 `Clean snore filename does not match ...`，说明该被试的 `鼾声` 文件名不符合仓库约定

### 第 4 步：预处理元音、clean 和 mix

```powershell
python scripts/preprocess_audio.py --subjects metadata/subjects.json --processed-root processed --sample-rate 16000 --vowel-seconds 1.0 --mix-dir-name 合成声_2 --synthesis-metadata metadata/synthesized_mix_metadata.jsonl --snr-stats-csv metadata/preprocess_snr_stats.csv
```

你应该看到：

- 进度条正常走完
- 最后输出 `Processed 5 subjects into ...\processed`

产物位置：

- `processed/vowel/`
- `processed/clean/`
- `processed/mix/`
- `metadata/preprocess_snr_stats.csv`

失败时怎么判断：

- 如果出现少量 `WARNING:`，先不要慌，先看命令有没有完整跑完。
- 如果命令中途直接报错退出，再把整段报错发出来。

### 第 5 步：生成训练用 manifest

这一步先不做 SNR 过滤，避免 5 人数据被筛空。

```powershell
python scripts/build_manifests.py --subjects metadata/subjects.json --splits-dir splits --output-dir manifests --processed-root processed --mix-dir-name 合成声_2 --snr-stats-csv metadata/preprocess_snr_stats.csv
```

你应该看到三行输出，分别对应：

- `train: ... samples`
- `val: ... samples`
- `test: ... samples`

产物位置：

- `manifests/enhancement_manifest_train.jsonl`
- `manifests/enhancement_manifest_val.jsonl`
- `manifests/enhancement_manifest_test.jsonl`

失败时怎么判断：

- 如果某个 split 显示 `0 samples`，先不要继续训练，先把输出发出来。

### 第 6 步：预计算 5 个被试的元音 embedding

```powershell
python scripts/precompute_vowel_embeddings.py -c config/enhancement.yaml --subjects metadata/subjects.json --processed-root processed --output-dir processed/embeddings --device cpu
```

你应该看到：

- 进度条正常走完
- 最后输出 `Saved 5 embeddings`

产物位置：

- `processed/embeddings/*.npy`

失败时怎么判断：

- 如果报找不到某个 `processed/vowel/...wav`，说明上一步预处理没有完整成功。

### 第 7 步：训练模型

当前仓库默认已经适配这次 5 人 CPU 跑通：

- `batch_size = 2`
- `num_epochs = 3`

直接运行：

```powershell
python scripts/train_enhancement.py -c config/enhancement.yaml --device cpu
```

你应该看到：

- 每个 epoch 的 `train_loss` 和 `val_loss`
- 训练结束后生成 checkpoint

产物位置：

- `outputs/checkpoints/latest.pt`
- `outputs/checkpoints/best.pt`
- `outputs/logs/train.log`

失败时怎么判断：

- 如果报内存不足，可以先把 [config/enhancement.yaml](./config/enhancement.yaml) 中的 `batch_size` 再改成 `1`
- 如果 `best.pt` 没有生成，不要继续推理和评估

### 第 8 步：做 1 条测试集推理

先自动读取测试被试和该被试的第一条混合音，再直接推理：

```powershell
$testSubject = Get-Content splits/test_subjects.txt -Encoding UTF8 | Select-Object -First 1
Write-Output $testSubject
$mixedFile = Get-ChildItem "processed/mix/$testSubject" -Filter *.wav | Select-Object -First 1 -ExpandProperty FullName
Write-Output $mixedFile
python scripts/infer_enhancement_long.py -c config/enhancement.yaml --checkpoint-path outputs/checkpoints/best.pt --mixed-file "$mixedFile" --vowel-dir "processed/vowel/$testSubject" --output-path "outputs/enhanced_wavs/${testSubject}_demo.wav" --device cpu
```

你应该看到：

- 最后输出 `Saved enhanced wav to ...`

产物位置：

- `outputs/enhanced_wavs/*.wav`

失败时怎么判断：

- 如果报 `outputs/checkpoints/best.pt` 不存在，说明训练还没有成功完成
- 如果报 `processed/mix/$testSubject` 下没有 wav，说明前面的预处理或 manifest 环节有问题
- 如果报 `argument --mixed-file: expected one argument`，通常就是 `$mixedFile` 为空，先看上面两行 `Write-Output` 的结果是不是正常

### 第 9 步：跑完整个测试集评估

```powershell
python scripts/evaluate_enhancement.py -c config/enhancement.yaml --checkpoint-path outputs/checkpoints/best.pt --output-csv outputs/eval/metrics.csv --device cpu
```

你应该看到：

- 评估进度条正常走完
- 最后输出 `Saved ... evaluation rows`

产物位置：

- `outputs/eval/metrics.csv`
- `outputs/eval/enhanced_wavs/*.wav`

失败时怎么判断：

- 如果 `metrics.csv` 没生成，就把整段报错直接发出来

### 最后检查清单

至少确认这些文件或目录存在：

- `metadata/subjects.json`
- `splits/train_subjects.txt`
- `splits/val_subjects.txt`
- `splits/test_subjects.txt`
- 每个被试目录下的 `合成声_2`
- `processed/vowel`
- `processed/clean`
- `processed/mix`
- `processed/embeddings`
- `manifests/enhancement_manifest_train.jsonl`
- `manifests/enhancement_manifest_val.jsonl`
- `manifests/enhancement_manifest_test.jsonl`
- `outputs/checkpoints/best.pt`
- `outputs/enhanced_wavs`
- `outputs/eval/metrics.csv`

遇到报错时，直接把原始报错贴出来，不需要你先自己分析。

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
      e1_3.wav
      i1_2.wav
      o1_3.wav
      u1_2.wav
    鼾声/
      hs_01_1.wav
      ...
      hs_01_10.wav
    info.txt
```

说明：

- `元音/` 目录必须能覆盖 `a/e/i/o/u` 这 5 个元音类别。
- 文件名不要求必须固定为 `a1_1.wav/e1_1.wav/...`。
- 只要文件名首字母能唯一表示元音即可，例如 `e1_3.wav`、`u1_2.wav` 也可以。
- 如果同一元音出现多个候选文件，当前脚本会按文件名字典序选择第一个，并打印 warning。

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
