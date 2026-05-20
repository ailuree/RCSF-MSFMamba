import { FolderKanban, ShieldCheck, Sparkles } from 'lucide-react'
import { moduleCards, workStats } from '../showcaseData'
import { getModulePurpose, overviewTags } from '../config/uiConfig'

function OverviewView({ dataset, onOpenResults, onOpenMaps }) {
  return (
    <div className="view-grid overview-grid">
      <section className="panel hero-panel">
        <div className="panel-header">
          <span>Overview</span>
          <strong className="accent-badge">{dataset.highlight}</strong>
        </div>
        <div className="hero-content">
          <div className="hero-copy">
            <div className="hero-copy-stack">
              <h2>多模态遥感分类展示台</h2>
              <p>整合方法结构、核心指标、分类效果与样例推理，方便直接进行演示与讲解。</p>
            </div>
            <div className="hero-chip-row">
              {overviewTags.map((tag) => (
                <span key={tag} className="hero-chip">
                  {tag}
                </span>
              ))}
            </div>
          </div>
          <div className="hero-summary-card">
            <span>Quick Brief</span>
            <strong>{dataset.label}</strong>
            <p>主看 Full Model v1 与鲁棒训练分支的性能差异，再用 ROI 推理做视觉收束。</p>
            <div className="hero-summary-meta">
              <article>
                <strong>5</strong>
                <span>方法版本</span>
              </article>
              <article>
                <strong>4</strong>
                <span>展示页面</span>
              </article>
              <article>
                <strong>1</strong>
                <span>推理样例</span>
              </article>
            </div>
          </div>
          <div className="hero-action-row">
            <button type="button" className="primary-button" onClick={onOpenResults}>
              查看实验结果
            </button>
            <button type="button" className="ghost-button" onClick={onOpenMaps}>
              进入分类效果页
            </button>
          </div>
        </div>
        <div className="metric-row">
          {Object.entries(dataset.heroStats).map(([key, value]) => (
            <article key={key} className="metric-card">
              <span>{key.toUpperCase()}</span>
              <strong>{value.toFixed(2)}%</strong>
            </article>
          ))}
        </div>
      </section>

      <section className="panel quick-panel">
        <div className="panel-header">
          <span>核心改进</span>
          <ShieldCheck size={16} />
        </div>
        <div className="quick-list">
          {moduleCards.slice(1).map((card) => (
            <article key={card.title} className="quick-item">
              <div className="quick-item-head">
                <strong>{card.title}</strong>
                <span>{card.location}</span>
              </div>
              <p>{getModulePurpose(card.title)}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="panel overview-work-panel">
        <div className="panel-header">
          <span>项目工作量</span>
          <FolderKanban size={16} />
        </div>
        <div className="stats-grid">
          {workStats.map((item) => (
            <article key={item.label} className="stat-card">
              <strong>{item.value}</strong>
              <span>{item.label}</span>
              <p>{item.detail}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="panel image-panel">
        <div className="panel-header">
          <span>结果快照</span>
          <Sparkles size={16} />
        </div>
        <div className="image-frame">
          <img src="/showcase/assets/fig5_extra_houston2013_metrics_dashboard.png" alt="Overview metric dashboard" />
        </div>
        <p className="viewer-note">论文图表、指标看板和预测图导出都已经整合进当前工作流。</p>
      </section>
    </div>
  )
}

export default OverviewView
