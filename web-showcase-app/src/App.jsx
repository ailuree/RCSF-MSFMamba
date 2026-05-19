import { useState } from 'react'
import {
  ArrowRight,
  Binary,
  Boxes,
  BrainCircuit,
  ChevronRight,
  CircuitBoard,
  FolderKanban,
  Radar,
  ScanSearch,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'
import './App.css'
import {
  datasets,
  featureGallery,
  moduleCards,
  pipelineSteps,
  supportingCharts,
  workStats,
} from './showcaseData'

const datasetTabs = [
  { key: 'houston2013', label: 'Houston2013' },
  { key: 'houston2018', label: 'Houston2018 small' },
]

const mapTabs = [
  { key: 'gt', label: 'Ground Truth' },
  { key: 'baseline', label: 'Baseline' },
  { key: 'robust', label: 'Robust Training' },
  { key: 'full', label: 'Full Model v1' },
]

const iconMap = [Radar, ShieldCheck, BrainCircuit, CircuitBoard, Boxes]

const chartPalette = ['#7d8cff', '#72d6b4', '#f8a54b', '#62b7ff', '#ff7f50']

function App() {
  const [selectedDataset, setSelectedDataset] = useState('houston2013')
  const [selectedFeature, setSelectedFeature] = useState('improvement')
  const [selectedChart, setSelectedChart] = useState('metrics')
  const [selectedMap, setSelectedMap] = useState('full')

  const activeDataset = datasets[selectedDataset]
  const activeFeature = featureGallery.find((item) => item.key === selectedFeature) ?? featureGallery[0]
  const activeChart = supportingCharts.find((item) => item.key === selectedChart) ?? supportingCharts[0]

  const availableMapTabs = mapTabs.filter((tab) => activeDataset.previewMaps[tab.key])
  const activeMap = activeDataset.previewMaps[selectedMap] ?? activeDataset.previewMaps[availableMapTabs[0].key]

  return (
    <div className="app-shell">
      <div className="ambient ambient-one"></div>
      <div className="ambient ambient-two"></div>

      <header className="topbar">
        <div className="brand-mark">
          <div className="brand-icon">RS</div>
          <div>
            <p>本科毕业设计展示</p>
            <strong>Multimodal Land Cover Classification</strong>
          </div>
        </div>
        <nav className="topnav">
          <a href="#methods">方法</a>
          <a href="#pipeline">实现</a>
          <a href="#results">结果</a>
          <a href="#visuals">效果</a>
        </nav>
      </header>

      <main>
        <section className="hero-section">
          <div className="hero-copy">
            <div className="eyebrow-line">
              <Sparkles size={16} />
              <span>答辩成果展示页</span>
            </div>
            <h1>基于深度学习的多模态遥感数据融合与地物分类一体化方法设计与实现</h1>
            <p className="hero-summary">
              以 MSFMamba 为基线，围绕融合可靠性、跨状态调制与鲁棒训练完成模型改进，并在
              Houston2013 与 Houston2018 small 上进行了系统验证。
            </p>

            <div className="hero-actions">
              <a className="primary-action" href="#results">
                查看项目结果
                <ArrowRight size={18} />
              </a>
              <a className="secondary-action" href="#pipeline">
                查看工程闭环
              </a>
            </div>

            <div className="hero-stat-grid">
              <article>
                <span>Houston2013 OA</span>
                <strong>{datasets.houston2013.heroStats.oa.toFixed(2)}%</strong>
              </article>
              <article>
                <span>Houston2018 OA</span>
                <strong>{datasets.houston2018.heroStats.oa.toFixed(2)}%</strong>
              </article>
              <article>
                <span>核心增强</span>
                <strong>3 类</strong>
              </article>
              <article>
                <span>实验分支</span>
                <strong>5 组</strong>
              </article>
            </div>
          </div>

          <div className="hero-panel">
            <div className="hero-panel-header">
              <span>成果概览</span>
              <strong>Full Model v1</strong>
            </div>
            <div className="hero-scoreboard">
              {Object.entries(datasets.houston2013.heroStats).map(([key, value]) => (
                <div key={key} className="score-item">
                  <span>{key.toUpperCase()}</span>
                  <strong>{value.toFixed(2)}</strong>
                </div>
              ))}
            </div>
            <div className="hero-preview">
              <img src="/showcase/assets/fig5_extra_houston2013_metrics_dashboard.png" alt="Houston2013 overall metric comparison" />
            </div>
          </div>
        </section>

        <section id="methods" className="content-section">
          <div className="section-heading">
            <p>01 / 方法改进</p>
            <h2>不仅是换模型，而是完成了三类增强与完整整合</h2>
          </div>

          <div className="method-layout">
            <div className="method-flow">
              <div className="flow-chip">
                <Binary size={18} />
                <span>HSI + Auxiliary 输入</span>
              </div>
              <ChevronRight size={18} />
              <div className="flow-chip">
                <ScanSearch size={18} />
                <span>MSFMamba 融合主干</span>
              </div>
              <ChevronRight size={18} />
              <div className="flow-chip emphasis">
                <ShieldCheck size={18} />
                <span>增强后的 Full Model v1</span>
              </div>
            </div>

            <div className="method-card-grid">
              {moduleCards.map((card, index) => {
                const Icon = iconMap[index % iconMap.length]
                return (
                  <article key={card.title} className="method-card">
                    <div className="method-card-top">
                      <Icon size={22} />
                      <span>{card.location}</span>
                    </div>
                    <h3>{card.title}</h3>
                    <p>{card.summary}</p>
                  </article>
                )
              })}
            </div>
          </div>
        </section>

        <section id="pipeline" className="content-section">
          <div className="section-heading">
            <p>02 / 工程实现</p>
            <h2>训练、评估、可视化、论文图表整理形成完整闭环</h2>
          </div>

          <div className="pipeline-layout">
            <div className="pipeline-card">
              <div className="pipeline-card-top">
                <FolderKanban size={18} />
                <span>Implementation Pipeline</span>
              </div>
              <div className="pipeline-track">
                {pipelineSteps.map((step, index) => (
                  <div key={step} className="pipeline-step">
                    <div className="pipeline-index">{`0${index + 1}`}</div>
                    <div>
                      <h3>{step}</h3>
                      <p>
                        {index === 0 && '完成数据准备与实验口径统一。'}
                        {index === 1 && '统一训练入口支撑多分支与鲁棒训练配置。'}
                        {index === 2 && '统一评估入口支撑正常与退化测试。'}
                        {index === 3 && '自动输出预测图、GT、输入视图与论文图。'}
                        {index === 4 && '将指标、图像与论文分析组织为可交付结果。'}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="workload-panel">
              {workStats.map((item) => (
                <article key={item.label} className="workload-card">
                  <strong>{item.value}</strong>
                  <span>{item.label}</span>
                  <p>{item.detail}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section id="results" className="content-section">
          <div className="section-heading section-heading-row">
            <div>
              <p>03 / 主结果</p>
              <h2>综合指标体现完整模型的最终效果</h2>
            </div>
            <div className="dataset-switch">
              {datasetTabs.map((tab) => (
                <button
                  key={tab.key}
                  type="button"
                  className={selectedDataset === tab.key ? 'active' : ''}
                  onClick={() => {
                    setSelectedDataset(tab.key)
                    setSelectedMap('full')
                  }}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>

          <div className="results-layout">
            <div className="results-chart-card">
              <div className="card-header-line">
                <div>
                  <span>{activeDataset.highlight}</span>
                  <h3>{activeDataset.label} 五分支 OA 对比</h3>
                </div>
                <strong className="mini-badge">Full Model v1 最优</strong>
              </div>
              <div className="oa-chart">
                {activeDataset.methods.map((entry, index) => {
                  const barHeight = ((entry.oa - 89.5) / (95 - 89.5)) * 100
                  const isHighlight = entry.name === 'Full Model v1'
                  return (
                    <div key={entry.name} className="oa-column">
                      <span className="oa-value">{entry.oa.toFixed(2)}</span>
                      <div className="oa-bar-shell">
                        <div
                          className={`oa-bar ${isHighlight ? 'highlight' : ''}`}
                          style={{
                            height: `${Math.max(barHeight, 8)}%`,
                            background: isHighlight
                              ? 'linear-gradient(180deg, #ffb17d 0%, #ff7f50 100%)'
                              : `linear-gradient(180deg, ${chartPalette[index % chartPalette.length]} 0%, rgba(255,255,255,0.08) 100%)`,
                          }}
                        ></div>
                      </div>
                      <span className="oa-label">{entry.name}</span>
                    </div>
                  )
                })}
              </div>
            </div>

            <div className="results-summary-card">
              <div className="card-header-line">
                <div>
                  <span>结论摘要</span>
                  <h3>{activeDataset.label}</h3>
                </div>
              </div>
              <div className="summary-metrics">
                {Object.entries(activeDataset.heroStats).map(([key, value]) => (
                  <div key={key}>
                    <span>{key.toUpperCase()}</span>
                    <strong>{value.toFixed(2)}%</strong>
                  </div>
                ))}
              </div>
              <ul className="summary-points">
                {activeDataset.talkingPoints.map((point) => (
                  <li key={point}>{point}</li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        <section id="visuals" className="content-section">
          <div className="section-heading">
            <p>04 / 可视化证据</p>
            <h2>从预测图、局部 ROI 与分析图表三个层面说明改进效果</h2>
          </div>

          <div className="visual-layout">
            <div className="map-viewer-card">
              <div className="card-header-line">
                <div>
                  <span>预测图预览</span>
                  <h3>{activeDataset.label} 分类结果</h3>
                </div>
              </div>
              <div className="viewer-tabs">
                {availableMapTabs.map((tab) => (
                  <button
                    key={tab.key}
                    type="button"
                    className={selectedMap === tab.key ? 'active' : ''}
                    onClick={() => setSelectedMap(tab.key)}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
              <div className="map-preview">
                <img src={activeMap} alt={`${activeDataset.label} ${selectedMap} preview`} />
              </div>
            </div>

            <div className="gallery-card">
              <div className="card-header-line">
                <div>
                  <span>图像证据</span>
                  <h3>{activeFeature.title}</h3>
                </div>
              </div>
              <div className="gallery-tab-strip">
                {featureGallery.map((item) => (
                  <button
                    key={item.key}
                    type="button"
                    className={selectedFeature === item.key ? 'active' : ''}
                    onClick={() => setSelectedFeature(item.key)}
                  >
                    {item.title}
                  </button>
                ))}
              </div>
              <div className="gallery-preview">
                <img src={activeFeature.image} alt={activeFeature.title} />
              </div>
              <p className="gallery-note">{activeFeature.description}</p>
            </div>
          </div>

          <div className="supporting-layout">
            <div className="supporting-chart-card">
              <div className="card-header-line">
                <div>
                  <span>补充图表</span>
                  <h3>{activeChart.title}</h3>
                </div>
              </div>
              <div className="gallery-tab-strip compact">
                {supportingCharts.map((item) => (
                  <button
                    key={item.key}
                    type="button"
                    className={selectedChart === item.key ? 'active' : ''}
                    onClick={() => setSelectedChart(item.key)}
                  >
                    {item.title}
                  </button>
                ))}
              </div>
              <div className="supporting-preview">
                <img src={activeChart.image} alt={activeChart.title} />
              </div>
              <p className="gallery-note">{activeChart.note}</p>
            </div>

            <aside className="conclusion-card">
              <div className="card-header-line">
                <div>
                  <span>答辩收束</span>
                  <h3>项目最终结论</h3>
                </div>
              </div>
              <div className="conclusion-points">
                <article>
                  <CircuitBoard size={18} />
                  <div>
                    <strong>完整模型综合最好</strong>
                    <p>Full Model v1 在主结果中保持综合最优或接近最优。</p>
                  </div>
                </article>
                <article>
                  <ShieldCheck size={18} />
                  <div>
                    <strong>鲁棒训练贡献最稳定</strong>
                    <p>退化评估与随机种子结果都说明 Robust Training 是核心增益来源。</p>
                  </div>
                </article>
                <article>
                  <Boxes size={18} />
                  <div>
                    <strong>形成完整工程闭环</strong>
                    <p>从结构改进、训练流程到图表整理，完成了可复现的项目实现。</p>
                  </div>
                </article>
              </div>
            </aside>
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
