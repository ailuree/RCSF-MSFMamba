# feature/reliability-gate 分支修改记录

## 1. 分支目标

本分支用于实现毕设方案第一阶段的最小改法：

- 不改主干网络
- 不改 `Fuse_SS2D`
- 不改层数和宽度
- 只在 `FSSBlock` 中加入第一版 `ReliabilityGate`

当前阶段目标是完成一个**可训练、可验证、接口不变**的纯通道门控版本。

---

## 2. 本分支实际修改内容

### 2.1 新增 `ReliabilityGate`

修改文件：

- [Mamba_v3.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/modules/Mamba_v3.py)

新增模块：

- `class ReliabilityGate(nn.Module)`

实现方式：

- 输入：
  - `x_feat`，对应 `x_1`
  - `y_feat`，对应 `y_1`
- 先分别做 `GAP`
- 将双模态摘要拼接
- 通过两个独立 `MLP` 分支输出：
  - `gate_x`
  - `gate_y`
- 经 `sigmoid` 后作为纯通道门控权重

输出形状：

- `gate_x`: `(B, 1, 1, Cx)`
- `gate_y`: `(B, 1, 1, Cy)`

---

### 2.2 在 `FSSBlock` 中挂载门控

修改文件：

- [Mamba_v3.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/modules/Mamba_v3.py)

修改位置：

- `FSSBlock.__init__`

新增内容：

- `self.reliability_gate = ReliabilityGate(self.d_inner1, self.d_inner2)`

说明：

- `self.d_inner1 / self.d_inner2` 与 `x_1 / y_1`、`x_out / y_out` 的通道维一致
- 因此适合直接生成门控权重

---

### 2.3 在 `FSSBlock.forward` 中接入门控

修改文件：

- [Mamba_v3.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/modules/Mamba_v3.py)

固定接入位置：

- `x_out = x_out * F.silu(x_2)` 之后
- `y_out = y_out * F.silu(y_2)` 之后
- `out_proj1 / out_proj2` 之前

当前逻辑：

```text
gate_x, gate_y = self.reliability_gate(x_1, y_1)
x_out = gate_x * x_out
y_out = gate_y * y_out
```

说明：

- 只控制融合输出强度
- 不提前改动 `x_1 / y_1`
- 不改变残差结构
- 不改变原有外部接口

---

### 2.4 增加门控均值统计

修改文件：

- [train.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/train.py)

新增内容：

- 训练过程中读取每层 `FSSBlock.last_gate_x / last_gate_y`
- 统计：
  - `GateX`
  - `GateY`
  - `GateX_avg`
  - `GateY_avg`

当前输出位置：

- 终端日志
- `logs/train.log`
- `metrics/epoch_history.csv`
- `metrics/last_metrics.json`
- `metrics/best_metrics.json`
- `metrics/run_summary.json`

作用：

- 验证门控不是恒等于 `1`
- 便于后续比较不同实验中门控的工作状态

---

## 3. 修改索引

### 3.1 代码文件索引

- [Mamba_v3.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/modules/Mamba_v3.py)
  - 新增 `ReliabilityGate`
  - 修改 `FSSBlock.__init__`
  - 修改 `FSSBlock.forward`
- [train.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/train.py)
  - 新增 gate 均值收集函数
  - 新增 gate 日志输出
  - 新增 gate 指标结构化保存

### 3.2 不变部分

以下内容在本分支未改动：

- `Fuse_SS2D`
- `MSpa-Mamba`
- `Spe-Mamba`
- `Net.forward`
- `train.py` 命令行接口
- `eval.py`
- `visualize.py`

---

## 4. 当前验证情况

### 4.1 已完成验证

已完成最小 smoke test 验证：

- 训练命令可正常运行
- `eval.py` 可加载当前分支权重
- `visualize.py` 可正常生成预测图、GT 图、输入对比图和四联图

验证命令：

```bash
python train.py --dataset Houston2013 --batchsize 4 --epoch 1 --num_work 2 --max_train_batches 10 --max_test_batches 5 --print_freq 5 --run_name gate_smoke_h2013_v2
python eval.py --dataset Houston2013 --run_name gate_smoke_h2013
python visualize.py --dataset Houston2013 --run_name gate_smoke_h2013 --split all --save_input_views 1
```

### 4.2 当前门控统计现象

示例输出中门控均值约为：

- `GateX_avg ≈ 0.5039`
- `GateY_avg ≈ 0.4967`

说明：

- 门控不是恒等于 `1`
- 门控也没有塌缩到 `0`
- 在少量 batch 训练下仍接近初始化中性状态，属于合理现象

---

## 5. 当前结论

本分支当前已经完成：

- 第一版纯通道 `ReliabilityGate` 结构接入
- 训练链路兼容
- 评估链路兼容
- 可视化链路兼容
- 门控均值监控

当前阶段尚未完成：

- 正式长训练
- 与基线的完整对比实验
- 门控可视化分析

因此，本分支当前状态可定义为：

**第一版可靠性门控结构已实现并通过最小验证，可进入后续正式实验或作为下一阶段创新的基础。**

---

## 6. 后续建议

本分支后续可做两类事情：

1. 继续正式训练并与基线对比  
2. 保持当前状态，转入下一分支实现 `Cross-State Modulation`

当前更推荐：

- 保持本分支为“第一版门控已完成”状态
- 进入下一创新分支继续开发
