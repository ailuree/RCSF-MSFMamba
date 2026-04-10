# feature/full-model-v1 分支修改记录

## 1. 分支定位

`feature/full-model-v1` 是当前毕设代码的**第一版完整整合分支**。

它不是新的单独创新分支，而是把此前已经独立实现并验证过的三个功能分支合并到一个统一可运行版本中：

- `feature/reliability-gate`
- `feature/cross-state`
- `feature/robust-training`

本分支的目标不是继续引入新想法，而是：

- 在同一份代码中同时保留三类改进
- 建立完整的“训练 + 评估 + 可视化 + 鲁棒性评估”闭环
- 形成后续正式实验和论文主模型实验的基础版本

也就是说，本分支对应的是你当前毕设中的**完整模型第一版工程实现**。

---

## 2. 整合来源与提交来源

本分支是从 [dev](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode) 基点整合出来的，基线起点为：

- `1c4e8fb`：完善基线，增加验证、计算、可视化

整合过程中实际引入的功能提交包括：

### 2.1 来自 `feature/reliability-gate`

- `efb8c2a`：添加 `ReliabilityGate` 初版
- `aba81d9`：增加 gate 均值统计
- `6b128bb`：增加该分支记录文档

### 2.2 来自 `feature/cross-state`

- `a7df4e1`：第一版纯 `Cross-State Modulation`
- `13c76ce`：增加该分支思路文档

### 2.3 来自 `feature/robust-training`

- `91df16c`：新增鲁棒训练工具层和思路文档
- `5d85413`：接入 clean/degraded 双分支训练
- `dd47b38`：增加退化评估与记录文档

---

## 3. 本分支最终保留的功能结构

### 3.1 结构改进一：ReliabilityGate

保留来源：

- `feature/reliability-gate`

当前保留在：

- [modules/Mamba_v3.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/modules/Mamba_v3.py)

最终口径：

- `ReliabilityGate` 位于 `FSSBlock` 内部
- 输入是 `x_1 / y_1` 的双模态摘要
- 输出是 `gate_x / gate_y`
- 作用位置固定为：
  - `x_out = x_out * F.silu(x_2)` 之后
  - `y_out = y_out * F.silu(y_2)` 之后
  - `out_proj1 / out_proj2` 之前

本分支中仍保留：

- `self.reliability_gate`
- `self.last_gate_x`
- `self.last_gate_y`

因此 gate 在整合分支中是**真实生效**的，不是只保留了日志壳子。

---

### 3.2 结构改进二：Cross-State Modulation

保留来源：

- `feature/cross-state`

当前保留在：

- [modules/Mamba_v3.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/modules/Mamba_v3.py)

最终口径：

- `CrossStateModulator` 位于 `Fuse_SS2D` 内
- 用参数生成模态 `x` 的全局摘要显式调制：
  - `dts`
  - `Bs`
  - `Cs`
- 不修改 `A`
- 不修改 `selective_scan`
- 不做 `beta`
- 缩放方式固定为：

```python
alpha = 1.0 + 0.5 * torch.tanh(raw_alpha)
```

本分支中仍保留：

- `self.cross_state_modulator`
- `self.last_alpha_dts_mean`
- `self.last_alpha_bs_mean`
- `self.last_alpha_cs_mean`

因此 cross-state 在整合分支中也是**真实生效**的。

---

### 3.3 训练策略改进：Robust Training

保留来源：

- `feature/robust-training`

当前保留在：

- [setting/robustness.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/setting/robustness.py)
- [setting/options.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/setting/options.py)
- [train.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/train.py)
- [eval.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/eval.py)

最终口径：

- `robust_train=0` 时，走普通训练
- `robust_train=1` 时，走 clean/degraded 双分支训练

当前保留的退化构造包括：

- `modality dropout`
- HSI PCA 高斯噪声
- HSI band-wise dropout
- 辅助模态高斯噪声

当前保留的总损失为：

```text
L_total = L_ce_clean + lambda_deg * L_ce_deg + lambda_cons * L_cons
```

其中：

- `L_ce_clean`：完整输入交叉熵
- `L_ce_deg`：退化输入交叉熵
- `L_cons`：退化预测与完整预测之间的 KL 一致性

---

### 3.4 统一训练日志与内部可观测性

本分支中的 [train.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/train.py) 已经不只是原始基线训练脚本，而是整合版训练入口。

现在它会同时记录三类内部信息：

#### 1. Gate 统计

- `GateX`
- `GateY`
- `GateX_avg`
- `GateY_avg`

#### 2. Cross-State 统计

- `AlphaDt`
- `AlphaB`
- `AlphaC`
- `AlphaDt_avg`
- `AlphaB_avg`
- `AlphaC_avg`

#### 3. Robust Training 统计

- `CleanCE`
- `DegCE`
- `Cons`
- `DropHSI`
- `DropAux`
- `HSINoise`
- `AuxNoise`
- `HSIBandDrop`

这些信息会进入：

- 终端输出
- `logs/train.log`
- `metrics/epoch_history.csv`
- `metrics/run_summary.json`
- `metrics/best_metrics.json`
- `metrics/last_metrics.json`

因此，本分支已经具备后续实验分析所需的内部可解释性基础。

---

## 4. 整合过程中实际发生的冲突与处理原则

### 4.1 为什么不直接 merge，而是按提交 cherry-pick

这几个功能分支都是从同一个 `dev` 基点独立拉出的，因此它们的改动彼此并不知道对方的存在。

如果直接 merge，最容易出现的问题是：

- `train.py` 逻辑被互相覆盖
- `modules/Mamba_v3.py` 的结构改动被部分丢失
- 日志与指标字段口径不一致

因此本次整合采用的是：

- 以 `dev` 为底
- 逐个 `cherry-pick`
- 冲突点人工整合

---

### 4.2 核心冲突文件一：`modules/Mamba_v3.py`

冲突来源：

- `feature/reliability-gate` 改了 `FSSBlock`
- `feature/cross-state` 改了 `Fuse_SS2D`

处理原则固定为：

- `FSSBlock` 保留 `ReliabilityGate`
- `Fuse_SS2D` 保留 `CrossStateModulator`
- 二者职责分开，不互相覆盖

最终结果是：

- `Fuse_SS2D` 负责状态参数调制
- `FSSBlock` 负责融合输出门控

这与原毕设方案完全一致。

---

### 4.3 核心冲突文件二：`train.py`

这是整合过程中最主要的冲突文件，因为：

- gate 分支修改了训练日志统计
- cross-state 分支也修改了训练日志统计
- robust-training 分支重构了训练主循环

最终处理原则是：

- 以 `robust-training` 的训练主循环为骨架
- 同时补回 gate 统计
- 同时补回 cross-state 统计
- 最终形成统一训练入口

也就是说：

- 训练主逻辑由 robust-training 提供
- 内部状态统计由 gate/cross-state 分支补充

这样才不会丢失任何一方能力。

---

## 5. 当前代码状态总览

截至本记录编写时，本分支已经形成如下统一状态：

### 5.1 模型结构

- `FSSBlock`：含 `ReliabilityGate`
- `Fuse_SS2D`：含 `CrossStateModulator`

### 5.2 训练入口

- [train.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/train.py)
  - 支持普通训练
  - 支持鲁棒双分支训练
  - 支持 gate/cross-state/退化统计统一输出

### 5.3 评估入口

- [eval.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/eval.py)
  - 支持普通评估
  - 支持 `drop_hsi`
  - 支持 `drop_aux`
  - 支持 `noise`
  - 支持 `combined`

### 5.4 可视化入口

- [visualize.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/visualize.py)
  - 可正常输出预测图
  - 可正常输出 GT 图
  - 可输出 HSI RGB 与辅助模态视图
  - 可输出四联图

---

## 6. 当前已完成验证

### 6.1 代码静态检查

已完成：

- `modules/Mamba_v3.py`
- `train.py`
- `eval.py`
- `setting/options.py`
- `setting/robustness.py`

的源码语法检查，结果通过。

---

### 6.2 训练 smoke test

在 `feature/full-model-v1` 下，已完成以下最小训练验证：

```bash
python train.py --dataset Houston2013 --batchsize 4 --epoch 1 --num_work 2 --max_train_batches 10 --max_test_batches 5 --print_freq 5 --run_name full_smoke_h2013 --robust_train 1
```

已观察到：

- `GateX / GateY` 正常输出
- `AlphaDt / AlphaB / AlphaC` 正常输出
- `CleanCE / DegCE / Cons` 正常输出
- `DropHSI / DropAux / HSINoise / AuxNoise / HSIBandDrop` 正常输出

示例结论：

- gate 不恒等于 1
- cross-state 调制不恒等于 1
- robust 双分支训练链路稳定

---

### 6.3 普通评估 smoke test

已完成：

```bash
python eval.py --dataset Houston2013 --run_name full_smoke_h2013
```

结果说明：

- 普通评估链路未被整合破坏
- checkpoint 读取正常
- 指标输出正常

---

### 6.4 退化评估 smoke test

已完成：

```bash
python eval.py --dataset Houston2013 --run_name full_smoke_h2013 --robust_eval_mode combined
```

说明：

- 退化评估模式可正常工作
- 退化统计会在评估阶段实时打印
- 退化条件下的指标可正常输出并保存

此前在 `feature/robust-training` 分支中还完成了：

- `none`
- `drop_hsi`
- `drop_aux`
- `noise`
- `combined`

五种模式的验证，因此鲁棒评估接口本身已具备可用性。

---

### 6.5 可视化 smoke test

已完成：

```bash
python visualize.py --dataset Houston2013 --run_name full_smoke_h2013 --split all --save_input_views 1
```

结果说明：

- 整合分支没有破坏可视化链路
- `prediction_map`
- `gt_map`
- `HSI RGB`
- `auxiliary view`
- `four-panel figure`

都能正常输出。

---

## 7. 当前观察到的现象与解释

### 7.1 Gate 与 Cross-State 都在工作

在整合分支 smoke test 中，已观察到例如：

- `GateX ≈ 0.50`
- `GateY ≈ 0.50`
- `AlphaDt ≈ 1.006`
- `AlphaB ≈ 0.999`
- `AlphaC ≈ 1.014`

这说明：

- 门控不是恒等映射
- cross-state 不是无效模块
- 目前是“轻量生效”，而不是过强干预

这与当前第一版设计目标一致。

---

### 7.2 模型当前更依赖 HSI 主模态

在此前鲁棒评估中可观察到：

- `drop_hsi` 时指标明显下降
- `drop_aux` 时下降较小

这说明当前模型在短训练条件下更依赖 HSI 主信息。

这并不奇怪，原因可能包括：

- HSI 本身信息量更强
- smoke test 训练轮数太少
- 辅助模态作用尚未充分学习出来

因此现阶段不能据此下结论说辅助模态“没用”，只能说明：

**在当前最小验证条件下，模型主导信息来源更偏向 HSI。**

---

## 8. 当前已知小问题

### 8.1 `Elapsed` 偶尔出现负数

在 `eval.py` / `visualize.py` 的运行过程中，偶发出现：

- `Elapsed: -0.45s`
- 或其他负数时间

这不是功能错误，更像是：

- WSL 与系统时钟源之间的小幅抖动
- `time.time()` 在当前环境下的偶发不稳定

后续建议修复方式：

- 将 [utility.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/utility.py)、[eval.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/eval.py)、[visualize.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/visualize.py) 中的计时统一改为 `time.perf_counter()`

该问题不影响训练、评估、可视化功能本身。

---

### 8.2 当前 smoke test 结果不能代表最终性能

本分支的所有当前结论都建立在：

- `epoch = 1`
- `max_train_batches = 10`
- `max_test_batches = 5`

的 smoke test 条件之上。

因此：

- 可以证明代码链路稳定
- 不能用于比较正式性能优劣
- 不能替代最终论文实验结果

---

## 9. 当前阶段结论

截至本记录编写时，`feature/full-model-v1` 已经达到以下状态：

1. `ReliabilityGate` 已整合
2. `Cross-State Modulation` 已整合
3. `Robust Training` 已整合
4. 训练链路可运行
5. 普通评估链路可运行
6. 退化评估链路可运行
7. 可视化链路可运行

这意味着：

**当前代码已经从“单项改动分支”进入“完整模型整合版”阶段。**

---

## 10. 下一步建议

本分支后续最合理的工作顺序是：

### 10.1 先提交并保存当前整合结果

建议先将当前整合分支提交并推送，作为稳定检查点。

---

### 10.2 再开始正式实验

建议后续正式实验顺序为：

1. `dev` 基线
2. `feature/reliability-gate`
3. `feature/cross-state`
4. `feature/robust-training`
5. `feature/full-model-v1`

这样才能形成完整消融与主模型对比表。

---

### 10.3 正式实验重点

后续真正需要关注的，不再是“代码能不能跑”，而是：

- `full-model-v1` 是否相对基线有稳定提升
- 在退化模式下是否比基线更稳
- gate / cross-state / robust-training 各自贡献大小如何

这将直接对应你论文中的：

- 主结果表
- 消融实验表
- 鲁棒性实验表

---

## 11. 总结

`feature/full-model-v1` 是你当前毕设代码中的**第一版完整整合实现**。

它已经把：

- 结构层面的 `ReliabilityGate`
- 状态层面的 `Cross-State Modulation`
- 训练层面的 `Robust Training`

三者统一到同一个工程版本中，并完成了最小训练、评估、退化评估、可视化验证。

因此，从工程角度看，这个分支已经具备作为后续正式实验主分支的基础条件。
