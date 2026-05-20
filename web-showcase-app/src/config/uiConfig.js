import { BarChart3, Binary, Compass, Map, MonitorPlay } from 'lucide-react'

export const views = [
  { key: 'overview', label: '项目总览', icon: Compass },
  { key: 'methods', label: '方法结构', icon: Binary },
  { key: 'results', label: '结果对比', icon: BarChart3 },
  { key: 'maps', label: '分类效果', icon: Map },
  { key: 'demo', label: '样例推理', icon: MonitorPlay },
]

export const datasetOptions = [
  { key: 'houston2013', label: 'Houston2013' },
  { key: 'houston2018', label: 'Houston2018 small' },
]

export const methodSequence = ['Baseline', 'Reliability Gate', 'Cross-State', 'Robust Training', 'Full Model v1']

export const methodShort = {
  Baseline: 'Baseline',
  'Reliability Gate': 'Gate',
  'Cross-State': 'Cross',
  'Robust Training': 'Robust',
  'Full Model v1': 'Full',
}

export const mapOptions = [
  { key: 'gt', label: 'Ground Truth' },
  { key: 'baseline', label: 'Baseline' },
  { key: 'robust', label: 'Robust Training' },
  { key: 'full', label: 'Full Model v1' },
]

export const inferenceNarrative = [
  '输入数据不是普通照片，而是 HSI 伪 RGB 与 LiDAR 高程信息。',
  '模型对 ROI 内每个像素做地物分类，输出完整分类图而不是零散测试点。',
  '点击预测图任意位置，可以直接读出该像素的地物类别和置信度。',
]

export const overviewTags = ['多模态融合', '结果对比', 'ROI 推理', '答辩展示']

export function getModulePurpose(title) {
  if (title === 'Reliability Gate') {
    return '控制不可靠融合通道，稳定输出。'
  }
  if (title === 'Cross-State Modulation') {
    return '让辅助模态参与状态调制。'
  }
  if (title === 'Robust Training') {
    return '提升模态退化场景下的鲁棒性。'
  }
  if (title === 'Baseline: MSFMamba') {
    return '统一多模态融合基线。'
  }
  if (title === 'Full Model v1') {
    return '整合结构与训练策略的最终方案。'
  }
  return ''
}
