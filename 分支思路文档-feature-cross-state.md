# feature/cross-state 分支思路文档

## 1. 文档目的

本文件用于在正式修改 `feature/cross-state` 分支代码前，明确以下问题：

- `CSFMambaRS` 真正值得借鉴的核心思想是什么
- 这些思想如何映射到当前 `MSFMamba` 代码骨架
- 哪些内容可以借鉴，哪些内容不应直接照搬
- 本分支后续改造应如何保持合理、自洽、可实现

本文件的定位不是“论文综述”，而是**面向当前代码实现的设计说明**。

---

## 2. 重新阅读 CSFMambaRS 后的核心结论

### 2.1 CSFMambaRS 的核心创新不在于 token 形式本身

根据 [IGARSS2025LaTeXTemplate.tex](/G:/temp/A-BiYe/BiShe3/4-ReferencePapersAndCode/2-paper-CSFMambaRS/IGARSS2025LaTeXTemplate.tex) 中 `Cross State Fusion Mamba block` 一节，`CSFMambaRS` 的关键点并不是“用了 CLS token”这件事本身，而是：

- 将一种模态的全局抽象信息显式注入另一种模态
- 让另一模态参与当前模态的 Mamba 状态参数生成
- 也就是让 Mamba 的 time-varying parameter generation 不再只由本模态决定，而带有跨模态信息

论文中对应的叙事是：

- 先用 `CLS Token` 作为模态整体信息代理
- 再将一支模态的 `CLS` 拼接到另一支模态特征中
- 最终在 `Fusion-SSM` 中实现“cross input informs state generation”

因此，`CSFMambaRS` 真正的可借鉴点应概括为：

**跨模态信息不仅参与特征拼接，还参与状态参数生成或状态更新过程。**

---

### 2.2 CSFMambaRS 的 cross-state 可概括为“跨模态决定状态参数”

论文中的公式写法为：

$$
h_t=\overline{\mathbf{A}}\cdot h_{t-1}+\overline{\mathbf{B}}(X^{cross}_{L,t})\cdot X^{cross}_{H,t}
$$

$$
Y_{H,t}=\mathbf{C}(X^{cross}_{L,t})\cdot h_t+\mathbf{D}\cdot X^{cross}_{H,t}
$$

其含义可以直接归纳为：

- 当前模态的输入不再独立决定状态演化
- 另一模态的信息进入了 `B/C/Δ` 一类动态参数的生成过程
- 这就是论文命名为 `Cross State Fusion` 的根本原因

也就是说，`cross-state` 的本质不是“简单双分支交互”，而是：

**一支模态不仅影响另一支模态的输出，还影响其内部状态建模参数。**

---

### 2.3 但 CSFMambaRS 的具体实现不适合直接搬到当前代码

虽然思想是可借鉴的，但 `CSFMambaRS` 的完整实现路径并不适合直接照搬到你当前的 `MSFMamba` 工程里，原因有三点：

1. `CSFMambaRS` 采用的是 token 化 + `CLS Token` 路线  
   而你当前的 `MSFMamba` 开源代码是卷积特征 + 2D 扫描状态建模路线。

2. `CSFMambaRS` 的 cross-state 叙事建立在“token 序列 + class token + fusion-SSM”的框架上  
   而你现在的 `MSFMamba` 融合代码主要集中在 [Mamba_v3.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/modules/Mamba_v3.py) 的 `Fuse_SS2D` 与 `FSSBlock` 中，没有 `CLS token` 这一层结构。

3. 如果直接照搬 token/CLS 结构，会导致：
   - 改动范围太大
   - 与现有 `MSFMamba` 代码组织不兼容
   - 风险明显高于本科毕设当前阶段可接受范围

所以本分支必须坚持的原则是：

**借鉴其“跨模态决定状态参数”的思想，不照搬其 token/CLS 实现。**

---

## 3. 当前 MSFMamba 代码中最适合承载 cross-state 的位置

### 3.1 当前 `MSFMamba` 的融合核心已经天然带有“跨模态决定参数”的雏形

阅读当前 [Mamba_v3.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/modules/Mamba_v3.py) 中的 `Fuse_SS2D.forward_core` 可以看到：

- `x` 分支先生成 `xs`
- 再用 `x_proj_weight` 和 `dt_projs_weight` 生成：
  - `dts`
  - `Bs`
  - `Cs`
- 同时 `y` 分支形成 `ys`
- 最终执行：

```text
out_y = selective_scan(
    ys, dts, As, Bs, Cs, Ds, ...
)
```

这意味着当前 `Fuse_SS2D` 已经不是“两个模态完全独立再相加”，而是：

- 一个模态负责生成动态状态参数
- 另一个模态负责提供被扫描输入

从这个角度看，当前 `MSFMamba` 的 `Fuse_SS2D` 已经包含了一种“弱 cross-state”雏形。

---

### 3.2 本分支真正要做的，不是从零发明 cross-state，而是把它变得更显式

因此，本分支最合理的目标不是：

- 重写 `selective_scan`
- 重写底层 `A/B/C/Δ` 公式
- 重构为 token + CLS 架构

而是：

- 在当前 `Fuse_SS2D` 的参数生成路径周围
- 增加一个轻量、显式、可控的跨模态状态调制分支

换句话说，本分支不是替换原有融合逻辑，而是做：

**从“隐式跨模态参数生成”升级到“显式跨模态状态调制”。**

---

## 4. 本分支建议采用的 cross-state 口径

### 4.1 核心口径

本分支的 `cross-state` 统一定义为：

**使用模态 A 的全局摘要或压缩状态，对模态 B 的中间状态参数生成过程进行轻量调制。**

这一定义与 `CSFMambaRS` 的精神一致，但不要求复刻其 token 路径。

---

### 4.2 当前最推荐的实现方式

结合你当前代码和毕设文档，本分支最推荐的实现方式是：

#### 方案：Cross-State Modulation

在 `Fuse_SS2D.forward_core(x, y)` 中加入以下逻辑：

1. 先从参数生成模态 `x` 中提取全局摘要  
   例如对 `x` 或 `xs` 做 `GAP`

2. 用该摘要生成一个轻量调制向量  
   例如：

```text
state_mod = MLP(GAP(x))
```

3. 用这个调制向量去调节另一条路径中与状态生成密切相关的中间表示  
   推荐优先调节以下对象之一：

- `x_dbl`
- `dts`
- `Bs`
- `Cs`

其中最稳的第一版建议优先调节：

- `dts`
- `Bs`
- `Cs`

即：

```text
dts = dts * alpha_dt
Bs  = Bs  * alpha_b
Cs  = Cs  * alpha_c
```

或者采用 affine 版本：

```text
dts = dts * alpha_dt + beta_dt
Bs  = Bs  * alpha_b  + beta_b
Cs  = Cs  * alpha_c  + beta_c
```

---

### 4.3 为什么不建议第一版直接改 `A`

尽管理论上 `cross-state` 也可以体现在 `A` 上，但第一版不建议动 `A_logs`，原因是：

- `A` 在状态转移中更敏感
- 改它更容易破坏训练稳定性
- 当前 `MSFMamba` 中 `A_logs` 是较稳定的核心参数
- 本科阶段更稳妥的方式是先调节输入相关的动态参数：
  - `dts`
  - `Bs`
  - `Cs`

所以本分支应固定遵循：

**先调节 input-dependent parameters，不直接碰状态转移主参数 `A`。**

---

## 5. 与 reliability-gate 分支的关系

### 5.1 两个创新点的职责应该严格分开

为了让后续消融实验清晰，`feature/cross-state` 分支的实现职责应该与 `feature/reliability-gate` 分支分开：

- `ReliabilityGate`
  - 解决“模态质量差异”和“错误传播控制”问题
  - 属于输出侧的融合强度调节

- `Cross-State Modulation`
  - 解决“跨模态交互仍然偏浅层”和“状态生成未显式共享”问题
  - 属于状态参数生成侧的深层调节

因此，本分支不应再把门控逻辑揉进核心实现里，而应保持：

- 该分支只做 cross-state
- 后续再考虑与 gate 版整合

---

### 5.2 后续完整模型的组合关系

后续完整模型的推荐叙事顺序应为：

1. 原始 `MSFMamba`
2. `MSFMamba + ReliabilityGate`
3. `MSFMamba + Cross-State Modulation`
4. `MSFMamba + ReliabilityGate + Cross-State Modulation`
5. 再加鲁棒训练

这样的好处是：

- 每个创新点的贡献可以单独验证
- 最终组合模型更容易写成论文中的模块叠加关系

---

## 6. 本分支后续代码修改建议

### 6.1 首要修改入口

本分支后续正式改代码时，优先修改：

- [Mamba_v3.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/modules/Mamba_v3.py)

重点位置：

- `Fuse_SS2D.__init__`
- `Fuse_SS2D.forward_core`

不建议优先改：

- `train.py`
- `Net.forward`
- `dataLoader.py`

---

### 6.2 推荐新增模块形式

最适合本分支的最小新增模块可命名为：

- `CrossStateModulator`

建议结构：

- 输入：某一模态的全局摘要
- 输出：对 `dts / Bs / Cs` 的缩放或仿射参数

推荐第一版只做：

- `sigmoid` 或 `tanh` 范围内的轻量缩放

避免一开始输出过强扰动。

---

### 6.3 推荐第一版口径

为了与 `ReliabilityGate` 分支保持同样的“最小改法”风格，本分支第一版应固定为：

- 不做双向复杂交叉
- 先做单位置、单层次、轻量调制
- 先只影响 `dts / Bs / Cs`
- 不改外部接口
- 不增新命令行参数

一句话概括为：

**在当前 `Fuse_SS2D` 内部增加一个轻量版 cross-state parameter modulation，而不是重写整个 fusion-SSM。**

---

## 7. 当前结论

重新阅读 `CSFMambaRS` 后，可以将本分支的设计原则明确为：

1. 借鉴的是其“另一模态参与状态参数生成”的思想  
2. 不照搬其 `CLS token + tokenization + fusion-SSM` 完整结构  
3. 当前 `MSFMamba` 的 `Fuse_SS2D` 已经有隐式 cross-state 雏形  
4. 本分支应做的是“显式 cross-state modulation”  
5. 最合理的改造位置是 `Fuse_SS2D.forward_core`  
6. 最合理的第一版对象是 `dts / Bs / Cs`，而不是 `A`

因此，`feature/cross-state` 分支最自洽、最可实现、最符合毕设当前阶段的路线是：

**在 `Fuse_SS2D` 中引入一个轻量跨状态调制模块，用一支模态的全局摘要显式调制另一支模态相关的动态状态参数生成过程，以实现对 `CSFMambaRS` cross-state 思想的本科可落地改写。**

---

## 8. 参考索引

### 8.1 CSFMambaRS 关键参考文件

- [IGARSS2025LaTeXTemplate.tex](/G:/temp/A-BiYe/BiShe3/4-ReferencePapersAndCode/2-paper-CSFMambaRS/IGARSS2025LaTeXTemplate.tex)
- [NetworkStructure.png](/G:/temp/A-BiYe/BiShe3/4-ReferencePapersAndCode/2-paper-CSFMambaRS/Figures/PNG/NetworkStructure.png)
- [MambaBlock.png](/G:/temp/A-BiYe/BiShe3/4-ReferencePapersAndCode/2-paper-CSFMambaRS/Figures/PNG/MambaBlock.png)

### 8.2 当前 MSFMamba 对应代码

- [Mamba_v3.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/modules/Mamba_v3.py)
- [MSFMamba.py](/G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/model/MSFMamba.py)

### 8.3 当前毕设方案文档

- [1-精简技术方案-项目执行版.md](/G:/temp/A-BiYe/BiShe3/5-MyProject/B-Docs/1-精简技术方案-项目执行版.md)
- [4-本科毕业设计技术方案.md](/G:/temp/A-BiYe/BiShe3/5-MyProject/B-Docs/4-本科毕业设计技术方案.md)
