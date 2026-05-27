# feature/robust-training 分支实施方案

## 1. 文档目的

本文件用于在 `feature/robust-training` 分支正式写代码前，先把“这一分支到底改什么、改到哪、第一版做到什么程度”固定下来，避免后续把鲁棒训练和结构创新混在一起。

本分支的定位是：

- **只做训练策略增强**
- **不改 `MSFMamba` 网络结构**
- **不引入 `ReliabilityGate`**
- **不引入 `Cross-State Modulation`**
- 作为后续 `full-model` 整合分支的一个独立可消融模块

这份文档面向当前代码实现，不是泛泛的论文综述。

---

## 2. 设计出发点

### 2.1 原始 MSFMamba 与 CSFMamba 的公开代码都更偏“完整双模态条件”

从当前 [1-paper-MSFMambaRS-match-CodeA](/G:/temp/A-BiYe/BiShe3/4-ReferencePapersAndCode/1-paper-MSFMambaRS-match-CodeA) 与 [2-paper-CSFMambaRS](/G:/temp/A-BiYe/BiShe3/4-ReferencePapersAndCode/2-paper-CSFMambaRS) 所对应的方法背景来看，它们更强调：

- 双模态协同建模
- 融合效率与分类精度
- 多模态交互机制设计

但并没有把“模态缺失、模态弱化、噪声退化条件下仍能稳定工作”作为代码层面重点公开出来。

因此，你的毕设里“鲁棒训练”这一分支是合理且必要的补充，其价值不在于再发明一个新 backbone，而在于：

**让当前多模态融合模型在弱模态、缺失模态或局部噪声条件下更稳。**

---

### 2.2 本分支的创新边界应该保持清晰

本分支不应该去动：

- [model/MSFMamba.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/model/MSFMamba.py) 的主干定义
- [modules/Mamba_v3.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/modules/Mamba_v3.py) 的融合结构

因为这两部分分别已经由：

- `feature/reliability-gate`
- `feature/cross-state`

承担结构创新。

`feature/robust-training` 的职责应固定为：

- 构造退化输入
- 控制退化强度
- 增加鲁棒损失
- 支持退化条件评估

也就是说，这一分支是**训练策略创新**，不是**网络结构创新**。

---

## 3. 当前代码中最合适的改造入口

结合 [train.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/train.py)、[setting/dataLoader.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/setting/dataLoader.py)、[setting/options.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/setting/options.py)、[eval.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/eval.py) 的当前结构，最合理的落点是：

### 3.1 数据退化构造不直接塞进 `HXDataset.__getitem__`

当前 `HXDataset` 已经承担了：

- patch 切块
- pad
- ToTensor
- 基础翻转增强

如果继续把：

- modality dropout
- HSI 波段 dropout
- 高斯噪声
- LiDAR 噪声

全部直接塞进 `__getitem__`，会导致：

- 数据读取逻辑越来越臃肿
- 无法方便地同时得到“干净输入”和“退化输入”
- 一致性损失实现困难

因此，本分支第一原则是：

**`dataLoader.py` 保持样本读取职责；退化逻辑抽到独立增强模块，在训练阶段对 batch 进行处理。**

---

### 3.2 最适合新增一个独立鲁棒训练工具模块

建议在 [setting](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/setting) 下新增一个独立文件，例如：

- `setting/robustness.py`

该文件统一负责：

- 退化配置
- batch 级输入退化
- consistency loss
- 退化信息日志统计

这样可以避免继续把复杂逻辑堆进 `train.py` 或 `dataLoader.py`。

---

### 3.3 `train.py` 负责训练主循环与双分支损失

鲁棒训练的核心实现放在 [train.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/train.py) 中：

- 先跑 clean branch
- 再构造 degraded branch
- 再计算 clean CE / degraded CE / consistency loss
- 汇总成总损失

这才符合当前项目已经形成的“训练入口集中在 `train.py`”的现实结构。

---

### 3.4 `eval.py` 建议支持可选退化评估

如果只训练不支持退化条件评估，这一分支后续很难做论文表格。

因此建议本分支同时补一个轻量能力：

- [eval.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/eval.py) 增加**可选退化评估参数**

默认仍是普通评估，不影响现有基线流程。

---

## 4. 本分支固定实现口径

### 4.1 总体原则

本分支固定遵循以下原则：

- 只改训练策略，不改模型结构
- 第一版先追求“稳定可跑通”，不追求复杂退化模拟
- 训练与评估都要能结构化保存结果
- 退化策略必须可关闭，默认不影响普通训练命令
- 所有退化强度都通过 CLI 显式控制，不写死在代码深处

---

### 4.2 第一版退化类型固定为三类

第一版只实现以下三类退化，不再额外扩展：

1. **模态 dropout**
- 以一定概率将 HSI 分支整体置零
- 或将 LiDAR / 辅助模态整体置零
- 每个样本最多只 drop 一支模态
- 不允许双模态同时全零

2. **HSI 轻量光谱退化**
- 对 `hsi_pca` 加轻度高斯噪声
- 对 `hsi` 原始 patch 做 band-wise dropout

3. **辅助模态轻量退化**
- 对 LiDAR / 辅助模态加入轻度高斯噪声
- 第一版不做复杂遮挡块，不做形态学扰动

第一版明确**不做**：

- 大面积 Cutout
- 几何错位模拟
- `.roi` 层面的特殊增强
- GAN 类生成扰动

这样能保证实现简单、训练可控、结果容易解释。

---

### 4.3 退化应用位置固定在 batch tensor 上

退化逻辑统一对已经从 DataLoader 取出的 tensor 生效。

具体口径：

- clean 分支输入：
  - `hsi_pca.unsqueeze(1)`
  - `xdata`
- degraded 分支输入：
  - 基于当前 batch clone 出来的退化版本

这样训练时同一个 batch 可以同时得到：

- `clean logits`
- `degraded logits`

从而自然支持一致性损失。

---

### 4.4 第一版损失函数固定写法

本分支最终损失固定为：

```text
L_total = L_ce_clean + lambda_deg * L_ce_deg + lambda_cons * L_cons
```

各项定义固定为：

- `L_ce_clean`
  - 干净输入上的交叉熵
- `L_ce_deg`
  - 退化输入上的交叉熵
- `L_cons`
  - 退化预测与干净预测之间的一致性约束

一致性项固定采用：

```text
KL( log_softmax(degraded_logits), softmax(clean_logits).detach() )
```

这里固定使用：

- clean 分支作为 teacher
- degraded 分支作为 student
- `clean_logits` 做 `detach`

这样更稳定，也更符合“让退化输入尽量接近完整输入预测”的直觉。

---

### 4.5 默认超参数口径固定

为了避免实现时再临时做产品级决策，本分支第一版默认参数先固定如下：

- `robust_train = 0`
- `lambda_deg = 1.0`
- `lambda_cons = 0.5`
- `modality_dropout_prob = 0.2`
- `hsi_dropout_prob = 0.15`
- `hsi_noise_std = 0.05`
- `aux_noise_std = 0.03`

具体解释：

- 普通训练默认不启用鲁棒训练，避免破坏现有命令
- 一旦启用 `robust_train=1`，就同时启用退化分支和一致性损失
- 第一版退化强度保持轻量，避免直接把训练打崩

---

### 4.6 退化概率控制规则固定

为保证行为明确，退化概率规则写死如下：

- 每个训练样本独立决定是否进入模态 dropout
- 若进入模态 dropout：
  - 50% 概率丢 HSI
  - 50% 概率丢辅助模态
- 无论是否进入模态 dropout，都可叠加轻量噪声
- band dropout 只作用于 HSI patch，不作用于 LiDAR

这样第一版就同时具备：

- 缺失模态训练
- 弱模态训练
- 轻噪声训练

---

## 5. 建议代码改造结构

### 5.1 新增 `setting/robustness.py`

建议新增独立文件 [setting/robustness.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/setting/robustness.py)，统一放置以下内容：

- `degrade_hsi_batch(...)`
- `degrade_aux_batch(...)`
- `apply_modality_dropout(...)`
- `build_degraded_batch(...)`
- `consistency_kl_loss(...)`
- `summarize_degradation_stats(...)`

这样可以把“训练策略逻辑”和“模型逻辑”分离。

---

### 5.2 `setting/options.py` 增加鲁棒训练参数

建议新增以下 CLI 参数：

- `--robust_train`
- `--lambda_deg`
- `--lambda_cons`
- `--modality_dropout_prob`
- `--hsi_dropout_prob`
- `--hsi_noise_std`
- `--aux_noise_std`
- `--robust_eval_mode`

其中：

- `robust_eval_mode` 默认设为 `none`
- 可选值固定为：
  - `none`
  - `drop_hsi`
  - `drop_aux`
  - `noise`
  - `combined`

这样后续 `eval.py` 就能直接复用一套退化接口。

---

### 5.3 `train.py` 的第一版改法

`train.py` 中固定按以下流程改：

1. 保留原有 clean forward
2. 若 `robust_train == 0`
   - 完全走现有训练逻辑
3. 若 `robust_train == 1`
   - 基于当前 batch 构造 degraded batch
   - 跑第二次 forward
   - 计算：
     - `loss_clean`
     - `loss_deg`
     - `loss_cons`
   - 汇总为总损失

同时在训练日志中新增：

- `loss_clean`
- `loss_deg`
- `loss_cons`
- `drop_hsi_ratio`
- `drop_aux_ratio`

并写入 `epoch_history.csv`。

---

### 5.4 `eval.py` 的第一版改法

`eval.py` 不新增新脚本，直接扩展现有脚本。

固定行为：

- 默认 `robust_eval_mode=none` 时，与当前行为完全一致
- 当指定退化模式时，在评估 batch 上应用相同类型的退化
- 输出仍保存到当前 `metrics/` 目录
- 文件名前缀带上退化模式，例如：
  - `test_drop_hsi_metrics_summary.json`
  - `test_combined_classification_report.txt`

这样后续可以直接整理鲁棒性实验表。

---

### 5.5 `visualize.py` 暂不做鲁棒分支改造

第一版不建议把退化输入可视化也混进这个分支。

原因是：

- 本分支核心是训练与评估鲁棒性
- 退化可视化不是当前必需项
- 会拖慢实施速度

因此本分支固定为：

- `visualize.py` 不改，继续复用现有输出链路

如果后续需要展示“退化条件下分类图”，再单独扩展。

---

## 6. 推荐实施顺序

本分支建议按以下顺序编码，而不是一次全堆进去：

### 第一步：先做退化构造与 degraded CE

先完成：

- `setting/robustness.py`
- `options.py` 参数
- `train.py` 中 clean + degraded 双分支
- 总损失先跑通

第一轮先验证：

- 退化 batch 构造无 shape 错误
- 训练链路能正常反向传播

---

### 第二步：再加 consistency loss

在第一步稳定后，再加入：

- `KL consistency`
- 日志记录 `loss_cons`

这样一旦训练不稳，问题更容易定位。

---

### 第三步：补退化评估

最后再扩展：

- `eval.py` 的 `robust_eval_mode`
- 退化条件下指标保存

这样能快速形成论文中的鲁棒性实验表。

---

## 7. 推荐提交节奏

建议本分支按以下节奏提交：

1. `feat: add robustness utilities for degraded multimodal batches`
2. `feat: add robust training losses for clean and degraded inputs`
3. `feat: add optional consistency loss and degradation logging`
4. `feat: support degraded evaluation modes for robustness experiments`

---

## 8. 最小验证方案

### 8.1 训练 smoke test

建议最小验证命令：

```bash
python train.py --dataset Houston2013 --batchsize 4 --epoch 1 --num_work 2 --max_train_batches 10 --max_test_batches 5 --print_freq 5 --run_name robust_smoke_h2013 --robust_train 1
```

验收目标：

- 能正常进入训练
- 不出现 shape mismatch
- 不出现 loss 为 nan
- 日志中能看到 clean / degraded / consistency 的分项

---

### 8.2 普通评估兼容性

```bash
python eval.py --dataset Houston2013 --run_name robust_smoke_h2013
```

验收目标：

- 普通评估仍可运行
- checkpoint 读取逻辑不坏

---

### 8.3 退化评估

```bash
python eval.py --dataset Houston2013 --run_name robust_smoke_h2013 --robust_eval_mode drop_hsi
python eval.py --dataset Houston2013 --run_name robust_smoke_h2013 --robust_eval_mode drop_aux
```

验收目标：

- 退化条件下可正常输出指标
- 输出文件命名可区分不同退化模式

---

## 9. 本分支成功标准

当前阶段不要求立刻得到最优精度，本分支的成功标准固定为：

- 不改模型结构的前提下，鲁棒训练策略能稳定接入
- `train.py` 可同时支持普通训练和鲁棒训练
- `eval.py` 可支持退化条件评估
- 结果文件可结构化保存，便于后续论文表格整理
- 后续能够与 `gate` / `cross-state` 分支自然整合

---

## 10. 最终结论

`feature/robust-training` 分支最合理、最清晰、最适合你当前毕设推进方式的路线是：

**不改网络结构，只新增一个独立鲁棒训练模块，对当前 batch 构造轻量退化输入，并以“clean CE + degraded CE + consistency loss”的方式训练，同时补充可选退化评估接口。**

这一路线的优点是：

- 与 `feature/reliability-gate`、`feature/cross-state` 边界清楚
- 与当前代码结构兼容
- 实现成本可控
- 后续容易合并成完整模型
