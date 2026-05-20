import { Boxes, Layers3, Radar } from 'lucide-react'
import { moduleCards, pipelineSteps } from '../showcaseData'

function MethodsView({ selectedModule, onSelectModule, selectedModuleInfo }) {
  return (
    <div className="view-grid methods-grid">
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
              <strong>解决的问题</strong>
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
              <strong>答辩讲法</strong>
              <p>先说模块插入位置，再说模块为什么有必要，最后切到结果页证明它是否带来真实增益。</p>
            </article>
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
                  {index === 0 && '统一数据准备、划分口径和实验基线。'}
                  {index === 1 && '不同分支共享训练入口，只在配置上切换。'}
                  {index === 2 && '正常测试与退化测试走统一评估框架。'}
                  {index === 3 && '输出预测图、对比图和论文图表素材。'}
                  {index === 4 && '将结果整理成论文可写、答辩可讲的成品。'}
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
