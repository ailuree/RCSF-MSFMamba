---
title: "3"
output:
    word_document:
        path: G:\temp\A-BiYe\BiShe3\5-MyProject\C-MyPaper\BC.docx
        toc: true
        toc_depth: 6
        number_sections: True
---

传统状态空间模型中，$\mathbf{B}$、$\mathbf{C}$ 和 $\Delta$ 在训练完成后对全体序列位置保持不变，
模型对所有输入位置采用统一的状态更新规则，无法根据当前内容自适应地选择信息。
Mamba 通过将上述参数设计为当前输入的函数，克服了这一局限。
具体而言，对序列中每个位置 $t$，模型通过独立的线性投影实时生成对应的动态参数：

$$\Delta_t = \mathrm{Softplus}(W_\Delta x_t)$$

$$\mathbf{B}_t = W_B x_t$$

$$\mathbf{C}_t = W_C x_t$$

其中 $W_\Delta$、$W_B$、$W_C$ 为可学习的投影矩阵，上述投影在选择性扫描执行之前完成，
对应代码实现中的 `dts`、`Bs` 和 `Cs` 参数。

将动态参数代入离散化过程，得到时变的离散状态方程：

$$\overline{\mathbf{A}}_t = \exp(\Delta_t \mathbf{A})$$

$$\overline{\mathbf{B}}_t \approx \Delta_t \mathbf{B}_t$$

$$h_t = \overline{\mathbf{A}}_t h_{t-1} + \overline{\mathbf{B}}_t x_t$$

$$y_t = \mathbf{C}_t h_t$$

注意，状态转移矩阵 $\mathbf{A}$ 仍保持全局共享，而 $\overline{\mathbf{A}}_t$、$\overline{\mathbf{B}}_t$、$\mathbf{C}_t$
均随位置 $t$ 变化，赋予模型逐位置自适应调整状态更新的能力。

其中，时间尺度参数 $\Delta_t$ 起到核心的门控作用：当 $\Delta_t$ 较大时，
$\overline{\mathbf{A}}_t = \exp(\Delta_t \mathbf{A})$ 快速衰减，同时
$\overline{\mathbf{B}}_t \approx \Delta_t \mathbf{B}_t$ 增大，模型以较大权重将当前输入写入状态，
对历史信息进行遗忘；当 $\Delta_t$ 较小时，$\overline{\mathbf{A}}_t \to \mathbf{I}$，
历史状态得以保留而当前输入影响减弱。这一机制在功能上类似于循环神经网络中的遗忘门，
但其系数直接从连续动力学的离散化过程中自然导出，而非人工设计的门控结构。
在 MSFMamba 的框架下，由于 `dts`、`Bs`、`Cs` 在扫描前独立生成，
对其施加轻量的跨模态调制即可在不修改选择性扫描公式的前提下改变各位置的状态建模条件，
为跨模态信息融合提供了自然的介入点。