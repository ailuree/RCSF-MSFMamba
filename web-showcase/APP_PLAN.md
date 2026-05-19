# 毕设答辩 Web 应用简化构思文档

## 1. 应用定位

这个 Web 应用不是论文全文展示页，也不是训练平台，而是一个面向答辩最后 1 到 3 分钟的项目成果展示页。

目标只有两个：

1. 让老师快速看懂你做了哪些工作。
2. 让老师直观看到项目结果和改进效果。

因此，应用应当尽量简洁，不展示研究背景、问题由来、长篇理论铺垫，而是直接进入：

- 我做了什么
- 我怎么实现
- 结果怎么样

## 2. 展示核心

整个应用只围绕两条主线展开：

### 2.1 主线一：工作量

要让老师看到这不是“只调了几个参数”，而是完成了较完整的工程与实验实现。

重点体现：

- 基于 `MSFMamba` 基线完成了二次改进
- 设计并实现了 3 类增强
  - `Reliability Gate`
  - `Cross-State Modulation`
  - `Robust Training`
- 完成了 `Full Model v1` 的统一整合
- 建立了训练、评估、可视化、鲁棒性测试的完整闭环
- 在两个数据集上完成了正式实验

### 2.2 主线二：项目效果

要让老师看到这个项目不是只有代码量，而是确实有结果支撑。

重点体现：

- `Full Model v1` 综合指标最优
- `Robust Training` 是最稳定的核心增益来源
- 局部区域可视化能看出误差收缩
- 类别级和鲁棒性结果能解释模型特点

## 3. 当前项目中可直接复用的资产

仓库里已经有很多适合直接接入 Web 的资源，不需要重新造素材。

### 3.1 图像结果

来自 `checkpoints/paper_figures/chapter5/`：

- `fig5_1_houston2013_inputs_vertical.png`
- `fig5_3_houston2013_improvement_focus.png`
- `fig5_4_houston2013_method_advantage.png`
- `fig5_extra_houston2013_metrics_dashboard.png`
- `fig5_extra_houston2013_per_class_gain.png`
- `fig5_extra_houston2013_robustness.png`

来自各实验目录 `figures/`：

- `hsi_rgb.png`
- `aux_view.png`
- `gt_map.png`
- `prediction_map.png`

### 3.2 指标结果

来自各实验目录 `metrics/`：

- `test_metrics_summary.json`
- `test_per_class_accuracy.csv`
- `test_confusion_matrix.csv`
- `test_drop_hsi_metrics_summary.json`
- `test_drop_aux_metrics_summary.json`
- `test_noise_metrics_summary.json`
- `test_combined_metrics_summary.json`

### 3.3 工作量证明材料

这些文档非常适合提炼成“项目实现内容”：

- `分支修改记录-feature-reliability-gate.md`
- `分支修改记录-feature-cross-state.md`
- `分支修改记录-feature-robust-training.md`
- `分支修改记录-feature-full-model-v1.md`
- `正式实验命令与表格口径-feature-full-model-v1.md`

## 4. 页面结构建议

第一版应用建议只做 5 个区块，足够支撑 1 到 3 分钟展示。

### 4.1 首屏总览

作用：

- 一眼说明课题
- 一眼给出结果

建议内容：

- 课题题目
- 一句话说明：
  - `面向高光谱与辅助模态遥感数据，完成融合建模、地物分类与鲁棒增强的一体化实现`
- 3 到 4 个结果卡片：
  - `Houston2013 OA 91.87%`
  - `Houston2018 small OA 91.14%`
  - `3 类核心改进`
  - `5 个主实验分支`

### 4.2 方法改进区

作用：

- 这是最重要的“工作量展示区”

建议内容：

- Baseline
- Reliability Gate
- Cross-State Modulation
- Robust Training
- Full Model v1

展示形式建议：

- 中间一条主流程线
- 旁边 4 个改进模块卡片

每张卡片只写 3 件事：

- 改了哪里
- 解决什么问题
- 在完整模型中的作用

这里的目标不是细讲论文原理，而是让老师看到：

- 你做了结构改进
- 你做了训练策略改进
- 你做了整合实现

### 4.3 工程与实验闭环区

作用：

- 直接体现“设计与实现”

建议内容：

- `数据准备`
- `train.py`
- `eval.py`
- `visualize.py`
- `checkpoints 结果整理`

建议配一个简洁流程图：

`数据 -> 训练 -> 评估 -> 可视化 -> 结果分析`

同时配一个工作量面板：

- 改进模块：3 个
- 完整整合模型：1 个
- 主结果分支：5 个
- 正式数据集：2 个
- 随机种子统计：3 组
- 退化评估模式：4 种

这一块非常重要，因为它能比单纯放结果图更能说明“你确实做了很多事”。

### 4.4 项目效果展示区

作用：

- 用最直观的图说明结果确实变好

建议保留 3 类展示：

1. 综合指标
   - `fig5_extra_houston2013_metrics_dashboard.png`

2. 局部改进效果
   - `fig5_3_houston2013_improvement_focus.png`

3. 鲁棒性或类别级特点
   - `fig5_extra_houston2013_robustness.png`
   - `fig5_extra_houston2013_per_class_gain.png`

如果展示时间特别紧，优先保留：

- 综合指标图
- 局部改进图
- 鲁棒性图

类别级图可以作为补充展示。

### 4.5 结论区

作用：

- 1 屏内收束全部信息

建议内容：

- `Full Model v1 综合性能最好`
- `Robust Training 是最稳定的核心增益来源`
- `结构改进 + 训练改进 + 工程整合共同完成了本课题实现`

这一块要短，不要再展开。

## 5. 1 到 3 分钟的推荐讲解顺序

建议现场讲解顺序如下：

1. 首屏
   - 讲题目、任务、最终结果
2. 方法改进区
   - 讲你做了哪 3 类增强，以及如何整合成完整模型
3. 工程与实验闭环区
   - 讲你不仅改了模型，还完成了训练、评估、可视化和鲁棒性测试
4. 项目效果展示区
   - 讲综合指标、局部改进、鲁棒性结果
5. 结论区
   - 用 2 到 3 句话结束

这个顺序最适合短时间展示，也最能突出工作量。

## 6. 第一版功能边界

第一版只做展示，不做复杂系统功能。

建议做：

- 单页滚动展示
- 图表与图片切换
- 数据集/方法切换
- 简单动效

不建议做：

- 在线推理
- 后端接口
- 用户系统
- 可编辑训练配置
- 复杂三维遥感交互

原因很明确：这些对答辩帮助有限，但会显著增加开发成本和风险。

## 7. 技术方案建议

建议单独建立前端目录：

`A-MyCode/web-showcase-app/`

建议技术栈：

- React + Vite
- Tailwind CSS
- ECharts 或 Recharts

建议做成纯前端静态应用，原因是：

- 本地运行稳定
- 演示风险低
- 开发和维护简单
- 后续也方便单独打包

## 8. 当前推荐的最终方向

这次 Web 应用不追求“大而全”，而是追求：

- 结构清楚
- 结果直观
- 工作量可见
- 1 到 3 分钟内能讲完

所以我们下一步正式构建时，应优先围绕下面 5 个关键词：

- `简洁`
- `工程感`
- `结果导向`
- `工作量可见`
- `答辩友好`

## 9. 下一步

如果这份简化版大纲没问题，下一步就直接开始建立：

`G:/temp/A-BiYe/BiShe3/5-MyProject/A-MyCode/web-showcase-app`

然后先做第一版页面骨架，把 5 个核心区块搭起来，再接入图片和指标数据。
