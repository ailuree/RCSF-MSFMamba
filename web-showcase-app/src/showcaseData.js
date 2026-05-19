export const datasets = {
  houston2013: {
    key: 'houston2013',
    label: 'Houston2013',
    highlight: '主分析数据集',
    heroStats: {
      oa: 91.87,
      aa: 93.03,
      kappa: 91.17,
      macroF1: 92.74,
    },
    methods: [
      { name: 'Baseline', oa: 90.596, aa: 91.778, kappa: 89.79, macroF1: 91.397 },
      { name: 'Reliability Gate', oa: 91.186, aa: 92.403, kappa: 90.433, macroF1: 92.002 },
      { name: 'Cross-State', oa: 90.711, aa: 92.079, kappa: 89.911, macroF1: 92.226 },
      { name: 'Robust Training', oa: 91.777, aa: 92.878, kappa: 91.075, macroF1: 91.66 },
      { name: 'Full Model v1', oa: 91.867, aa: 93.029, kappa: 91.17, macroF1: 92.743 },
    ],
    talkingPoints: [
      'Full Model v1 在 4 项综合指标上全部最优。',
      'Robust Training 是单模块中最稳定、最强的性能来源。',
      'Cross-State 更偏向改善类别均衡性，而不是直接拉高 OA。',
    ],
    previewMaps: {
      gt: '/showcase/assets/houston2013_gt_map.png',
      baseline: '/showcase/assets/baseline_h2013_prediction_map.png',
      robust: '/showcase/assets/robust_h2013_prediction_map.png',
      full: '/showcase/assets/full_model_h2013_prediction_map.png',
    },
  },
  houston2018: {
    key: 'houston2018',
    label: 'Houston2018 small',
    highlight: '跨场景验证集',
    heroStats: {
      oa: 91.14,
      aa: 94.6,
      kappa: 88.6,
      macroF1: 89.7,
    },
    methods: [
      { name: 'Baseline', oa: 90.492, aa: 93.934, kappa: 87.788, macroF1: 88.197 },
      { name: 'Reliability Gate', oa: 90.745, aa: 94.617, kappa: 88.113, macroF1: 89.99 },
      { name: 'Cross-State', oa: 90.308, aa: 94.105, kappa: 87.522, macroF1: 88.464 },
      { name: 'Robust Training', oa: 91.124, aa: 94.591, kappa: 88.581, macroF1: 89.135 },
      { name: 'Full Model v1', oa: 91.137, aa: 94.599, kappa: 88.601, macroF1: 89.704 },
    ],
    talkingPoints: [
      'Full Model v1 在第二个数据集上仍保持综合最优。',
      'Robust Training 的收益具有跨数据集一致性。',
      'Cross-State 在该数据集上收益较弱，说明其更具探索性。',
    ],
    previewMaps: {
      gt: '/showcase/assets/houston2018_gt_map.png',
      full: '/showcase/assets/full_model_h2018_small_prediction_map.png',
    },
  },
}

export const workStats = [
  { value: '3', label: '核心改进模块', detail: 'Gate / Cross-State / Robust Training' },
  { value: '5', label: '主实验分支', detail: 'Baseline 到 Full Model v1' },
  { value: '2', label: '正式数据集', detail: 'Houston2013 与 Houston2018 small' },
  { value: '4', label: '退化评估模式', detail: 'Drop HSI / Drop Aux / Noise / Combined' },
  { value: '3', label: '随机种子统计', detail: 'Houston2013 三组 mean ± std' },
  { value: '1', label: '完整工程闭环', detail: '训练、评估、可视化、结果整理' },
]

export const moduleCards = [
  {
    title: 'Baseline: MSFMamba',
    location: '多模态融合主干',
    summary: '以 MSFMamba 作为统一基线，承担高光谱与辅助模态的融合分类任务。',
  },
  {
    title: 'Reliability Gate',
    location: 'FSSBlock 输出侧',
    summary: '对双模态融合结果做显式可靠性重标定，抑制不稳定通道响应。',
  },
  {
    title: 'Cross-State Modulation',
    location: 'Fuse_SS2D 状态参数层',
    summary: '利用模态摘要调制 dts / Bs / Cs，增强跨模态状态交互。',
  },
  {
    title: 'Robust Training',
    location: '训练与评估流程',
    summary: '引入 clean/degraded 双分支、一致性约束与退化评估，强化鲁棒性。',
  },
  {
    title: 'Full Model v1',
    location: '最终整合方案',
    summary: '将三类增强统一整合为完整模型，形成可训练、可评估、可视化的闭环。',
  },
]

export const pipelineSteps = [
  '数据准备与划分',
  '统一训练入口 train.py',
  '统一评估入口 eval.py',
  '可视化生成 visualize.py',
  '结果汇总与论文图表整理',
]

export const featureGallery = [
  {
    key: 'inputs',
    title: '多模态输入与标注',
    image: '/showcase/assets/fig5_1_houston2013_inputs_vertical.png',
    description: '展示高光谱、辅助模态与标签分布，直观说明任务输入和输出。',
  },
  {
    key: 'improvement',
    title: '局部区域改进效果',
    image: '/showcase/assets/fig5_3_houston2013_improvement_focus.png',
    description: '通过高净增益 ROI 说明 Full Model v1 对基线误差的直接修正效果。',
  },
  {
    key: 'advantage',
    title: '方法优势分布',
    image: '/showcase/assets/fig5_4_houston2013_method_advantage.png',
    description: '整体展示 Robust Training 与 Full Model v1 相对基线的改善区域分布。',
  },
]

export const supportingCharts = [
  {
    key: 'metrics',
    title: '综合指标对比',
    image: '/showcase/assets/fig5_extra_houston2013_metrics_dashboard.png',
    note: '适合快速说明完整模型在多项指标上的综合领先。',
  },
  {
    key: 'robustness',
    title: '退化鲁棒性对比',
    image: '/showcase/assets/fig5_extra_houston2013_robustness.png',
    note: '说明正常精度与退化条件之间的平衡关系。',
  },
  {
    key: 'perClass',
    title: '类别级变化',
    image: '/showcase/assets/fig5_extra_houston2013_per_class_gain.png',
    note: '说明综合指标提升主要来自关键类别识别能力的增强。',
  },
]
