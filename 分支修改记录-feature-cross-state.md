# feature/cross-state 分支修改记录

## 1. 分支目标

本分支实现第一版纯 `Cross-State Modulation`，目标是：

- 只修改 `Fuse_SS2D`
- 不叠加 `ReliabilityGate`
- 不改 `FSSBlock`
- 不改 `A`
- 不重写 `selective_scan`
- 在现有 `MSFMamba` 融合主干中引入一个轻量、显式、可训练的跨状态调制机制

本分支对应的思想来源是 `CSFMambaRS` 中“跨模态参与状态参数生成”的核心思路，但不照搬其 `token + CLS` 结构。

## 2. 修改原则

本次实现固定遵循以下口径：

- 只做 `cross-state-only`
- 只调 `dts / Bs / Cs`
- 第一版只做乘法缩放，不做 `beta`
- 缩放系数采用 `1 + 0.5 * tanh(raw_alpha)`，使调制范围围绕 `1`
- 不改变 `train.py`、`eval.py`、`visualize.py` 的命令行接口

## 3. 实际改动位置

主要改动文件：

- [Mamba_v3.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/modules/Mamba_v3.py)

本次实际修改点：

1. 新增 `CrossStateModulator`
2. 在 `Fuse_SS2D.__init__` 中挂载 `self.cross_state_modulator`
3. 在 `Fuse_SS2D` 中增加最近一次调制均值缓存
4. 在 `Fuse_SS2D.forward_core` 中对 `dts / Bs / Cs` 进行显式调制

## 4. CrossStateModulator 结构说明

`CrossStateModulator` 的输入是参数生成模态 `x` 的中间特征，形状为 `(B, C, H, W)`。

处理流程为：

1. 对 `x` 做 `GAP`
2. 通过一个共享轻量 MLP 干路
3. 通过三个独立 head 输出：
   - `alpha_dts`
   - `alpha_bs`
   - `alpha_cs`

输出形状固定为：

- `alpha_dts`: `(B, 1, d_inner2, 1)`
- `alpha_bs`: `(B, 1, d_state, 1)`
- `alpha_cs`: `(B, 1, d_state, 1)`

最终调制公式固定为：

```python
alpha = 1.0 + 0.5 * torch.tanh(raw_alpha)
```

## 5. forward_core 插入点说明

当前 `Fuse_SS2D.forward_core` 的主链路是：

```text
x -> xs -> x_dbl -> dts / Bs / Cs
y -> ys
selective_scan(ys, dts, As, Bs, Cs, Ds, ...)
```

本分支的插入点固定为：

- `dts / Bs / Cs` 已生成
- `dts` 仍处于 `(B, K, d_inner2, L)`
- `Bs / Cs` 仍处于 `(B, K, d_state, L)`
- 且在 reshape 成 `selective_scan` 最终输入之前

接入逻辑为：

```python
alpha_dts, alpha_bs, alpha_cs = self.cross_state_modulator(x)
dts = dts * alpha_dts
Bs = Bs * alpha_bs
Cs = Cs * alpha_cs
```

这使当前分支实现的是一个显式的、轻量的参数调制版 cross-state，而不是完全重写状态扫描模块。

## 6. 可观测性保留

为便于后续检查调制是否真正生效，`Fuse_SS2D` 额外缓存：

- `last_alpha_dts_mean`
- `last_alpha_bs_mean`
- `last_alpha_cs_mean`

这些值记录最近一次前向中三个调制系数的均值，当前阶段不强制写进训练日志，但后续如需观测或画图可直接复用。

## 7. 与其他分支的边界

与 `feature/reliability-gate` 的边界固定为：

- `feature/reliability-gate` 只改 `FSSBlock`，做纯通道可靠性门控
- `feature/cross-state` 只改 `Fuse_SS2D`，做状态参数轻量调制
- 两个分支当前保持独立，便于后续消融：
  - baseline
  - gate-only
  - cross-state-only
  - gate + cross-state

## 8. 最小验证命令

建议使用以下最小 smoke test：

```bash
python train.py --dataset Houston2013 --batchsize 4 --epoch 1 --num_work 2 --max_train_batches 10 --max_test_batches 5 --print_freq 5 --run_name cross_smoke_h2013
python eval.py --dataset Houston2013 --run_name cross_smoke_h2013
python visualize.py --dataset Houston2013 --run_name cross_smoke_h2013 --split all --save_input_views 1
```

## 9. 当前阶段成功标准

当前阶段不要求性能立刻优于基线，成功标准是：

- 结构已稳定接入
- 训练可进入前向与反向
- `eval.py` 与 `visualize.py` 不被破坏
- 调制均值缓存可读
- 为后续正式实验和消融对比打好基础
