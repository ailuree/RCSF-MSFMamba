# feature/cross-state-v3 分支修改记录

## 1. 分支目标

本分支在 `feature/cross-state` 第一版实现的基础上，继续做两轮小范围结构优化，目标不是重写 `Cross-State Modulation`，而是：

- 保留 `cross-state-only` 的边界
- 不混入 `ReliabilityGate`
- 不改训练脚本外部接口
- 在尽量小的代码改动下，观察 `cross-state` 是否还能进一步提升

当前优化重点固定为两类：

1. 调制强度
2. 调制对象与调制输入

也就是说，本分支的核心目标不是发明新模块，而是把第一版 `cross-state` 的调制设计从“能跑通”推进到“更接近可用”。

---

## 2. 优化背景

第一版 `feature/cross-state` 的正式结果表明：

- 相比 baseline，`cross-state` 有一定正向作用
- 但提升较温和
- 说明“显式跨状态调制”这个方向并非无效
- 更可能的问题在于：第一版调制策略还偏保守

因此本分支的优化思路不是推翻 `cross-state`，而是围绕以下问题做最小改动：

- `alpha = 1 + 0.5 * tanh(raw)` 是否过于保守
- 是否有必要继续调 `dts`
- 是否只用参数生成模态 `x` 的摘要，跨模态信息不够充分

---

## 3. v2 改动记录

### 3.1 目标

`v2` 的目标是先做最小风险优化：

- 提高调制强度
- 同时避免继续对更敏感的 `dts` 做调制

### 3.2 实际改动

在 [Mamba_v3.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/modules/Mamba_v3.py) 中：

1. 将 `CrossStateModulator` 的缩放强度从：

```python
scale = 0.5
```

调整为：

```python
scale = 0.75
```

2. 在 `Fuse_SS2D` 中新增：

```python
self.modulate_dts = False
```

3. 在 `forward_core` 中把：

```python
dts = dts * alpha_dts
Bs = Bs * alpha_bs
Cs = Cs * alpha_cs
```

改成：

```python
if self.modulate_dts:
    dts = dts * alpha_dts
Bs = Bs * alpha_bs
Cs = Cs * alpha_cs
```

即默认只调 `Bs / Cs`，先不调 `dts`。

### 3.3 v2 结果结论

`cross_h2013_v2` 相比 `cross_h2013`：

- 没有变差
- 但提升非常有限

因此 `v2` 的结论是：

- 方向安全
- 但收益还不足以说明优化成功

---

## 4. v3 改动记录

### 4.1 目标

`v3` 在 `v2` 基础上继续推进，但仍然坚持“只做小改动”的原则。

这次固定尝试：

1. 进一步提高调制强度
2. 将调制输入从单边 `x` 摘要升级为 `x / y` 联合摘要
3. 继续只调 `Bs / Cs`，不恢复 `dts`

### 4.2 实际改动

仍然只改 [Mamba_v3.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/modules/Mamba_v3.py)。

#### 改动 1：调制强度继续提高

将：

```python
scale = 0.75
```

改为：

```python
scale = 1.0
```

即：

```python
alpha = 1.0 + 1.0 * torch.tanh(raw_alpha)
```

使调制范围进一步扩大。

#### 改动 2：CrossStateModulator 支持联合摘要

将原先只接收 `x` 的：

```python
def forward(self, x):
```

改为：

```python
def forward(self, x, y=None):
```

并在内部使用：

```python
summary_x = GAP(x)
summary_y = GAP(y)
summary = concat(summary_x, summary_y)
```

也就是说，`alpha` 不再只由单边模态决定，而是由双模态联合决定。

#### 改动 3：调制器输入维度同步调整

在 `Fuse_SS2D.__init__` 中，将调制器输入维度从：

```python
CrossStateModulator(self.d_inner1, self.d_inner2, self.d_state)
```

改为：

```python
CrossStateModulator(self.d_inner1 + self.d_inner2, self.d_inner2, self.d_state)
```

因为 `v3` 的共享 MLP 不再只吃 `x`，而是吃拼接后的 `x+y` 联合摘要。

#### 改动 4：forward_core 调用改为双输入

将：

```python
alpha_dts, alpha_bs, alpha_cs = self.cross_state_modulator(x)
```

改为：

```python
alpha_dts, alpha_bs, alpha_cs = self.cross_state_modulator(x, y)
```

但仍固定：

- `self.modulate_dts = False`
- 即只调 `Bs / Cs`

### 4.3 v3 结果结论

`cross_h2013_v3` 相比此前结果：

| 配置             |      OA |      AA |   Kappa | Macro-F1 |
| ---------------- | ------: | ------: | ------: | -------: |
| `baseline_h2013` | 90.5960 | 91.7780 | 89.7897 |  91.3968 |
| `cross_h2013`    | 90.7108 | 92.0792 | 89.9114 |  92.2261 |
| `cross_h2013_v2` | 90.7190 | 92.0856 | 89.9203 |  92.2312 |
| `cross_h2013_v3` | 90.9404 | 91.9541 | 90.1612 |  92.0565 |

可以看出：

- `v3` 相比 `v1/v2`，在 `OA / Kappa` 上有更明显提升
- 但 `AA / Macro-F1` 反而略低于 `v1/v2`

因此 `v3` 的改进结论不是“全面变强”，而是：

- 更偏向提高整体分类正确率
- 对类别均衡性的帮助没有第一版明显

换句话说，`v3` 更像是把 `cross-state` 的收益方向从“偏均衡”拉向了“偏整体精度”。

---

## 5. 当前阶段判断

### 5.1 是否值得保留

值得保留。

原因是：

- `v3` 已经不再是 `v2` 那种几乎无变化的小抖动
- 它至少证明：
  - `cross-state` 不是完全不可优化
  - 调制输入和调制对象的设计，确实会显著影响结果

### 5.2 是否继续深挖到 v4

当前不建议继续深挖。

原因是：

- 你已经拿到一个比 `v1/v2` 更有说服力的版本
- 再继续调，很可能进入低收益反复微调
- 当前毕设阶段更重要的是：
  - 固定主线
  - 写论文
  - 准备答辩展示

因此，更合理的策略是：

- 将 `v3` 作为 `cross-state` 的改进版保留
- 在论文中解释优化过程
- 而不是继续无限迭代

---

## 6. 论文中如何描述 cross-state 从 v1 到 v3 的优化过程

### 6.1 推荐的论文叙事方式

不要把 `cross-state` 写成“一次设计就显著成功”的模块。  
更适合的叙事是：

1. 借鉴 `CSFMambaRS` 的跨模态状态交互思想，首先设计了一个第一版显式 `Cross-State Modulation`
2. 第一版验证了方向可行，但性能提升较温和
3. 为进一步分析影响因素，又围绕调制强度、调制对象、调制输入做了小范围优化
4. 最终得到 `v3`，说明：
   - 联合模态摘要更适合做跨状态调制
   - 对 `B/C` 状态项的调制比同时调 `dts` 更稳健

这样写的好处是：

- 更真实
- 更符合研究过程
- 更容易体现“你不是机械照搬论文，而是在做设计迭代”

### 6.2 论文里可直接使用的描述口径

可以写成类似下面的意思：

> 在初版 cross-state 模块中，本文采用单边模态全局摘要生成对 `dts/Bs/Cs` 的统一缩放系数。实验表明，该设计能够带来一定增益，但提升较为有限。进一步分析后，本文认为 cross-state 模块的效果与调制对象选择和跨模态信息注入方式密切相关。因此，在后续优化中，本文一方面提高调制强度，另一方面将调制输入从单边模态摘要扩展为双模态联合摘要，并优先保留对 `Bs/Cs` 状态项的调制而不直接作用于更敏感的 `dts`。实验结果表明，优化后的 `v3` 版本在整体精度和 Kappa 系数上进一步提升，说明联合摘要驱动的状态调制方式能够更有效地发挥 cross-state 机制的作用。

### 6.3 在论文中应如何定位 v1 / v2 / v3

建议写法：

- `v1`：第一版可运行结构
- `v2`：中间过渡版，用于验证“增强强度 + 去掉 dts 调制”的安全性
- `v3`：最终保留版

也就是说，论文主文不需要详细展开 `v2` 全部内容，但可以在正文或补充实验中简要提到：

- `v2` 主要用于验证方向安全
- `v3` 才是最终写入论文的优化版结果

### 6.4 在论文里不要怎么写

不建议写成：

- “cross-state 模块显著优于所有其他单模块”
- “cross-state 是本论文最主要的性能来源”

因为从现有结果看，这两句话都不够稳。

更合理的写法是：

- cross-state 是本文在结构层面的一个探索性增强模块
- 其最终优化版能够稳定改善整体指标
- 但最主要、最稳定的性能来源仍然是 `Robust Training`

---

## 7. 当前最适合写入论文的综合结论

基于 `Houston2013` 与 `Houston2018 small` 当前结果，`cross-state` 更适合被写成：

- 一个具有研究意义的结构探索点
- 经调制设计优化后，能够对整体指标带来正向作用
- 但它不是当前最强单模块
- 它的价值更多体现在：
  - 证明跨模态状态调制方向可行
  - 说明结构设计细节会显著影响该模块收益

这样写，既保住了你参考 `CSFMambaRS` 的学术来源，也不会因为结果不够强而显得牵强。
