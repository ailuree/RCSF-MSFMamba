import { Activity, ArrowUpRight, FolderKanban, PlayCircle, ShieldCheck, Sparkles, Waves } from 'lucide-react'
import { moduleCards, workStats } from '../showcaseData'
import { getModulePurpose, overviewTags } from '../config/uiConfig'

function OverviewView({ dataset, onOpenResults, onOpenMaps }) {
  const signalItems = [
    { label: 'Primary Lead', value: 'Full Model v1', icon: Activity },
    { label: 'Best Gain', value: 'Robust Training', icon: Waves },
    { label: 'Final Demo', value: 'ROI Explain', icon: PlayCircle },
  ]

  return (
    <div className="view-grid overview-grid">
      <section className="panel hero-panel hero-panel-focus">
        <div className="panel-header">
          <span>Overview</span>
          <strong className="accent-badge">{dataset.highlight}</strong>
        </div>
        <div className="hero-content">
          <div className="hero-copy hero-main-story">
            <div className="hero-stage-visual" aria-hidden="true">
              <div className="hero-stage-ring hero-stage-ring-a"></div>
              <div className="hero-stage-ring hero-stage-ring-b"></div>
              <div className="hero-stage-grid">
                <span></span>
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
            <div className="hero-copy-stack">
              <h2>多模态遥感分类展示台</h2>
              <p>把方法、结果、分类图和 ROI 推理收成一条清晰展示路径。</p>
            </div>
            <div className="hero-chip-row">
              {overviewTags.map((tag) => (
                <span key={tag} className="hero-chip">
                  {tag}
                </span>
              ))}
            </div>
            <div className="hero-signal-strip">
              {signalItems.map((item) => {
                const Icon = item.icon
                return (
                  <article key={item.label} className="hero-signal-card">
                    <div className="hero-signal-icon">
                      <Icon size={15} />
                    </div>
                    <div>
                      <span>{item.label}</span>
                      <strong>{item.value}</strong>
                    </div>
                  </article>
                )
              })}
            </div>
            <div className="hero-action-row">
              <button type="button" className="primary-button" onClick={onOpenResults}>
                查看结果
              </button>
              <button type="button" className="ghost-button" onClick={onOpenMaps}>
                查看分类图
              </button>
            </div>
          </div>
          <div className="hero-aside">
            <div className="hero-summary-card hero-story-card">
              <span>Main Thread</span>
              <strong>{dataset.label}</strong>
              <p>从 Full Model v1 切入，用 Robust Training 和 ROI 推理收束。</p>
              <div className="hero-summary-meta">
                <article>
                  <strong>1</strong>
                  <span>主结论</span>
                </article>
                <article>
                  <strong>2</strong>
                  <span>关键改进</span>
                </article>
                <article>
                  <strong>3</strong>
                  <span>演示入口</span>
                </article>
              </div>
            </div>

            <article className="hero-preview-card">
              <div className="hero-preview-head">
                <span>Visual Focus</span>
                <button type="button" className="hero-preview-link" onClick={onOpenMaps}>
                  <ArrowUpRight size={14} />
                </button>
              </div>
              <div className="hero-preview-image">
                <img src={dataset.previewMaps.full ?? dataset.previewMaps.gt} alt={`${dataset.label} preview`} />
              </div>
              <div className="hero-preview-foot">
                <strong>{dataset.talkingPoints[0]}</strong>
                <p>{dataset.talkingPoints[1]}</p>
              </div>
            </article>
          </div>
        </div>
      </section>

      <section className="panel overview-metrics-panel">
        <div className="panel-header">
          <span>Core Metrics</span>
          <strong>{dataset.label}</strong>
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
        <p className="viewer-note">指标图、预测图、论文图表已统一整理。</p>
      </section>
    </div>
  )
}

export default OverviewView
