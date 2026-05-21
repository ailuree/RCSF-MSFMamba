import { Boxes, Layers3, Radar, Sparkles, Waypoints } from 'lucide-react'
import { moduleCards, pipelineSteps } from '../showcaseData'

function MethodsView({ selectedModule, onSelectModule, selectedModuleInfo }) {
  const moduleSignals = [
    { label: 'Module Count', value: '5 nodes' },
    { label: 'Current Focus', value: selectedModuleInfo.title.replace('Baseline: ', '') },
    { label: 'Talk Track', value: 'Position -> Impact -> Result' },
  ]

  return (
    <div className="view-grid methods-grid">
      <section className="panel panel-hero-strip">
        <div className="panel-header">
          <span>Method Signals</span>
          <Sparkles size={16} />
        </div>
        <div className="signal-grid">
          {moduleSignals.map((item) => (
            <article key={item.label} className="signal-card">
              <span>{item.label}</span>
              <strong>{item.value}</strong>
            </article>
          ))}
        </div>
      </section>

      <section className="panel module-menu-panel">
        <div className="panel-header">
          <span>Method Navigator</span>
          <Layers3 size={16} />
        </div>
        <div className="module-menu">
          {moduleCards.map((card) => (
            <button
              key={card.title}
              type="button"
              className={selectedModule === card.title ? 'active' : ''}
              onClick={() => onSelectModule(card.title)}
            >
              <span>{card.title}</span>
              <small>{card.location}</small>
            </button>
          ))}
        </div>
      </section>

      <section className="panel module-detail-panel">
        <div className="panel-header">
          <span>Selected Module</span>
          <Radar size={16} />
        </div>
        <div className="module-detail">
          <h2>{selectedModuleInfo.title}</h2>
          <p className="module-location">{selectedModuleInfo.location}</p>
          <p>{selectedModuleInfo.summary}</p>
          <div className="module-detail-boxes">
            <article>
              <strong>作用</strong>
              <p>
                {selectedModuleInfo.title === 'Reliability Gate' &&
                  '减少融合后不可靠通道对最终分类的干扰，提高输出可信度。'}
                {selectedModuleInfo.title === 'Cross-State Modulation' &&
                  '让辅助模态摘要显式调制状态参数，强化跨模态状态交互。'}
                {selectedModuleInfo.title === 'Robust Training' &&
                  '在退化场景下维持性能，降低模型对单一模态完整性的依赖。'}
                {selectedModuleInfo.title === 'Baseline: MSFMamba' &&
                  '提供统一可对比的多模态融合主干，作为所有改进的起点。'}
                {selectedModuleInfo.title === 'Full Model v1' &&
                  '将结构与训练策略统一为最终方案，形成综合表现最优的完整模型。'}
              </p>
            </article>
            <article>
              <strong>讲解顺序</strong>
              <p>位置、作用、结果。</p>
            </article>
          </div>
          <div className="preview-callout">
            <div className="preview-callout-head">
              <span>Current Reading</span>
              <Waypoints size={15} />
            </div>
            <strong>{selectedModuleInfo.location}</strong>
            <p>这个模块在答辩里更适合和结果页联动讲，不单独停留太久。</p>
          </div>
        </div>
      </section>

      <section className="panel pipeline-panel">
        <div className="panel-header">
          <span>Engineering Pipeline</span>
          <Boxes size={16} />
        </div>
        <div className="pipeline-list">
          {pipelineSteps.map((step, index) => (
            <article key={step} className="pipeline-item">
              <div className="pipeline-index">{`0${index + 1}`}</div>
              <div>
                <strong>{step}</strong>
                <p>
                  {index === 0 && '数据与基线。'}
                  {index === 1 && '训练分支切换。'}
                  {index === 2 && '统一评估。'}
                  {index === 3 && '图表与预测图导出。'}
                  {index === 4 && '论文与答辩整理。'}
                </p>
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  )
}

export default MethodsView
