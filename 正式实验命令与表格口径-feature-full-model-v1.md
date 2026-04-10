# feature/full-model-v1 正式实验命令与表格口径

## 1. 文档目的

本文件用于固定两类内容：

- `feature/full-model-v1` 分支在正式实验阶段应如何运行
- 后续 baseline 到 full-model 的正式实验顺序与论文表格口径如何统一

这份文档的定位是**正式实验执行说明**，不是分支修改记录。

---

## 2. 当前分支定位

当前 [feature/full-model-v1](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode) 已经整合了：

- `ReliabilityGate`
- `Cross-State Modulation`
- `Robust Training`

因此，这个分支对应的是当前毕设中的**完整模型第一版**，适合承担：

- 主模型正式训练
- 主模型普通评估
- 主模型退化评估
- 主模型分类图输出

建议论文或实验记录中的口径统一写为：

- `RCSF-MSFMamba (Full Model v1)`

如果只是内部实验目录命名，可以继续用：

- `full_model_h2013`
- `full_model_h2018`

---

## 3. feature/full-model-v1 正式实验命令

### 3.1 Houston2013 正式训练

建议正式 run name：

- `full_model_h2013`

建议命令：

```bash
python train.py --dataset Houston2013 --batchsize 8 --epoch 40 --num_work 2 --print_freq 20 --run_name full_model_h2013 --robust_train 1
```

说明：

- 当前阶段建议保留 `robust_train=1`
- `batchsize=8` 是相对稳妥的起点
- `epoch=40` 先与当前代码默认值一致，后续如有必要再扩到更长轮次

---

### 3.2 Houston2013 普通评估

```bash
python eval.py --dataset Houston2013 --run_name full_model_h2013
```

该命令用于获得：

- OA
- AA
- Kappa
- Macro-F1
- per-class accuracy
- classification report
- confusion matrix

输出目录默认位于：

- [checkpoints/Houston2013/full_model_h2013/metrics](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/checkpoints/Houston2013/full_model_h2013/metrics)

---

### 3.3 Houston2013 退化评估

建议至少跑以下四组：

```bash
python eval.py --dataset Houston2013 --run_name full_model_h2013 --robust_eval_mode drop_hsi
python eval.py --dataset Houston2013 --run_name full_model_h2013 --robust_eval_mode drop_aux
python eval.py --dataset Houston2013 --run_name full_model_h2013 --robust_eval_mode noise
python eval.py --dataset Houston2013 --run_name full_model_h2013 --robust_eval_mode combined
```

说明：

- `drop_hsi`：测试缺失 HSI 时性能
- `drop_aux`：测试缺失 LiDAR/辅助模态时性能
- `noise`：测试轻噪声条件
- `combined`：测试综合退化条件

这些结果将直接用于后续“鲁棒性实验表”。

---

### 3.4 Houston2013 分类图输出

```bash
python visualize.py --dataset Houston2013 --run_name full_model_h2013 --split all --save_input_views 1
```

该命令会输出：

- `prediction_map.png`
- `gt_map.png`
- `hsi_rgb.png`
- `aux_view.png`
- `four_panel.png`

这些图可直接用于：

- 论文分类图
- PPT 展示图
- 与基线分支做视觉对比

---

### 3.5 Houston2018 同步实验命令

当 `Houston2013` 跑完后，再用同一套命令切换到 `Houston2018`。

建议 run name：

- `full_model_h2018`

训练：

```bash
python train.py --dataset Houston2018 --batchsize 8 --epoch 40 --num_work 2 --print_freq 20 --run_name full_model_h2018 --robust_train 1
```

普通评估：

```bash
python eval.py --dataset Houston2018 --run_name full_model_h2018
```

退化评估：

```bash
python eval.py --dataset Houston2018 --run_name full_model_h2018 --robust_eval_mode drop_hsi
python eval.py --dataset Houston2018 --run_name full_model_h2018 --robust_eval_mode drop_aux
python eval.py --dataset Houston2018 --run_name full_model_h2018 --robust_eval_mode noise
python eval.py --dataset Houston2018 --run_name full_model_h2018 --robust_eval_mode combined
```

可视化：

```bash
python visualize.py --dataset Houston2018 --run_name full_model_h2018 --split all --save_input_views 1
```

---

## 4. 正式实验顺序建议

### 4.1 推荐总顺序

后续正式实验建议严格按以下顺序进行：

1. `dev` 基线
2. `feature/reliability-gate`
3. `feature/cross-state`
4. `feature/robust-training`
5. `feature/full-model-v1`

这样安排的原因是：

- 先固定 baseline
- 再逐个验证单模块贡献
- 最后验证完整整合模型是否优于单模块

这才适合形成论文中的：

- 基线对比
- 单模块消融
- 完整模型实验

---

### 4.2 每个分支建议的正式 run_name

为了后续整理结果，建议统一使用如下命名：

#### Houston2013

- `baseline_h2013`
- `gate_h2013`
- `cross_h2013`
- `robust_h2013`
- `full_model_h2013`

#### Houston2018

- `baseline_h2018`
- `gate_h2018`
- `cross_h2018`
- `robust_h2018`
- `full_model_h2018`

这样目录结构最清楚，也方便你后续直接回看。

---

### 4.3 每个分支建议最少产物

每个正式实验都建议至少保留：

- `weights/best.pth`
- `weights/last.pth`
- `metrics/test_metrics_summary.json`
- `metrics/classification_report.txt`
- `metrics/confusion_matrix.csv` 或 `.npy`
- `metrics/per_class_accuracy.csv`
- `figures/prediction_map.png`
- `figures/gt_map.png`
- `figures/four_panel.png`

对于 `robust-training` 和 `full-model-v1`，还建议额外保留：

- `metrics/test_drop_hsi_metrics_summary.json`
- `metrics/test_drop_aux_metrics_summary.json`
- `metrics/test_noise_metrics_summary.json`
- `metrics/test_combined_metrics_summary.json`

---

## 5. 论文表格口径建议

### 5.1 主结果表

建议论文主结果表用于回答：

**你的完整模型相对基线和单模块改进是否整体更优。**

建议列：

- Method
- Dataset
- OA
- AA
- Kappa
- Macro-F1

建议行：

- MSFMamba Baseline
- MSFMamba + ReliabilityGate
- MSFMamba + Cross-State Modulation
- MSFMamba + Robust Training
- RCSF-MSFMamba (Full Model v1)

建议按数据集分成两张表，或同一张表按 `Houston2013 / Houston2018` 分块。

---

### 5.2 消融实验表

建议消融表重点强调：

**每个模块单独引入时的效果，以及完整模型的综合效果。**

建议列：

- Baseline
- ReliabilityGate
- Cross-State
- Robust Training
- OA
- AA
- Kappa

建议行写成二进制开关形式：

- `0 0 0`
- `1 0 0`
- `0 1 0`
- `0 0 1`
- `1 1 1`

如果你后面想更细，也可以补：

- `1 1 0`
- `1 0 1`
- `0 1 1`

但当前阶段不强制。

---

### 5.3 鲁棒性实验表

建议鲁棒性实验表只对以下模型展开：

- Baseline
- Robust Training
- Full Model v1

这样最有解释力。

建议列：

- Method
- Normal
- Drop HSI
- Drop Aux
- Noise
- Combined

每列指标建议优先填：

- OA

如果版面允许，再补：

- AA
- Kappa

这样最容易直观看出：

- 完整模型是否在退化条件下更稳
- robust-training 是否真的带来鲁棒性提升

---

### 5.4 分类图展示口径

建议最终论文分类图至少展示：

- Baseline
- ReliabilityGate
- Cross-State
- Full Model v1
- Ground Truth

如果篇幅有限，最少保留：

- Baseline
- Full Model v1
- Ground Truth

这样最容易体现整合模型的最终效果。

---

## 6. 当前推荐执行节奏

### 6.1 建议先做 Houston2013

推荐顺序：

1. 先把 `Houston2013` 上五个分支全部正式跑完
2. 先整理出：
   - 主结果表
   - 消融表
   - 鲁棒性实验表
3. 再把同样流程迁移到 `Houston2018`

原因：

- `Houston2013` 当前更适合作为主验收集
- 代码、数据、图像口径已经更成熟
- 更适合先形成完整论文骨架

---

### 6.2 建议每跑完一个分支立即整理

不要等全部跑完再整理。

建议流程固定为：

1. 训练
2. 普通评估
3. 退化评估
4. 可视化
5. 把核心指标填入统一表格

这样最不容易丢结果，也方便后续比较。

---

## 7. 当前已知注意事项

### 7.1 计时输出偶发负值

在 `eval.py` 和 `visualize.py` 的日志中，`Elapsed` 偶发会出现负值。

这不影响实验结果本身，但后续建议统一修成：

- `time.perf_counter()`

---

### 7.2 smoke test 结果不能直接当正式结果

当前你已经跑通的 `full_smoke_h2013` 只能说明：

- 代码链路通
- 整合成功

不能直接写进论文正式结果表。

正式结果应统一来自：

- `baseline_h2013`
- `gate_h2013`
- `cross_h2013`
- `robust_h2013`
- `full_model_h2013`

以及对应的 `h2018` 实验。

---

## 8. 建议结论

当前阶段最推荐的正式实验推进路线是：

1. 先在 `Houston2013` 上完成五组正式实验
2. 先形成：
   - 主结果表
   - 消融实验表
   - 鲁棒性实验表
3. 再切换 `Houston2018`
4. 最后整理分类图与论文表格

对于 `feature/full-model-v1` 本身，当前正式命令已经可以固定为：

```bash
python train.py --dataset Houston2013 --batchsize 8 --epoch 40 --num_work 2 --print_freq 20 --run_name full_model_h2013 --robust_train 1
python eval.py --dataset Houston2013 --run_name full_model_h2013
python eval.py --dataset Houston2013 --run_name full_model_h2013 --robust_eval_mode drop_hsi
python eval.py --dataset Houston2013 --run_name full_model_h2013 --robust_eval_mode drop_aux
python eval.py --dataset Houston2013 --run_name full_model_h2013 --robust_eval_mode noise
python eval.py --dataset Houston2013 --run_name full_model_h2013 --robust_eval_mode combined
python visualize.py --dataset Houston2013 --run_name full_model_h2013 --split all --save_input_views 1
```

这套命令已经足够支撑你当前完整模型的正式实验流程。
