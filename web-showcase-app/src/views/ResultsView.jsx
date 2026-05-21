import { BarChart3, ScanSearch, Sparkles, TrendingUp } from 'lucide-react'
import { methodSequence, methodShort } from '../config/uiConfig'

function ResultsView({ dataset, selectedMetric, onSelectMetric, methodMetrics, bestMethod }) {
  const scaleMin = Math.min(...methodMetrics.map((item) => item.value)) - 0.2
  const scaleMax = Math.max(...methodMetrics.map((item) => item.value)) + 0.2
  const sortedMethods = [...methodMetrics].sort((left, right) => right.value - left.value)
  const topMethod = sortedMethods[0]
  const runnerUp = sortedMethods[1]
  const gain = topMethod && runnerUp ? (topMethod.value - runnerUp.value).toFixed(2) : '0.00'

  return (
    <div className="view-grid results-grid">
      <section className="panel panel-hero-strip">
        <div className="panel-header">
          <span>Result Signals</span>
          <Sparkles size={16} />
        </div>
        <div className="signal-grid">
          <article className="signal-card">
            <span>Selected Metric</span>
            <strong>{selectedMetric.toUpperCase()}</strong>
          </article>
          <article className="signal-card">
            <span>Current Leader</span>
            <strong>{bestMethod.name}</strong>
          </article>
          <article className="signal-card">
            <span>Lead Margin</span>
            <strong>{gain}</strong>
          </article>
        </div>
      </section>

      <section className="panel chart-panel">
        <div className="panel-header">
          <span>Metric Switcher</span>
          <BarChart3 size={16} />
        </div>
        <div className="control-strip">
          {[
            ['oa', 'OA'],
            ['aa', 'AA'],
            ['kappa', 'Kappa'],
            ['macroF1', 'Macro-F1'],
          ].map(([key, label]) => (
            <button
              key={key}
              type="button"
              className={selectedMetric === key ? 'active' : ''}
              onClick={() => onSelectMetric(key)}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="bar-stage">
          {methodMetrics.map((method) => {
            const height = ((method.value - scaleMin) / (scaleMax - scaleMin)) * 100
            const isBest = method.name === bestMethod.name
            return (
              <div key={method.name} className="bar-column">
                <span>{method.value.toFixed(2)}</span>
                <div className="bar-track">
                  <div
                    className={`bar-fill ${isBest ? 'best' : ''}`}
                    style={{ height: `${Math.max(height, 8)}%` }}
                  ></div>
                </div>
                <strong>{methodShort[method.name]}</strong>
              </div>
            )
          })}
        </div>
      </section>

      <section className="panel result-summary-panel">
        <div className="panel-header">
          <span>Result Summary</span>
          <Sparkles size={16} />
        </div>
        <div className="summary-highlight">
          <strong>{bestMethod.name}</strong>
          <p>
            {selectedMetric.toUpperCase()} leader on {dataset.label}.
          </p>
        </div>
        <div className="preview-callout">
          <div className="preview-callout-head">
            <span>Interpretation</span>
            <TrendingUp size={15} />
          </div>
          <strong>{`${selectedMetric.toUpperCase()} +${gain}`}</strong>
          <p>这张图适合先讲领先者，再讲 Robust Training 和 Cross-State 各自带来的偏向收益。</p>
        </div>
        <div className="result-table">
          {methodSequence.map((name) => {
            const method = dataset.methods.find((item) => item.name === name)
            return (
              <article key={name}>
                <strong>{name}</strong>
                <span>{method?.oa.toFixed(3)} OA</span>
                <span>{method?.aa.toFixed(3)} AA</span>
                <span>{method?.kappa.toFixed(3)} Kappa</span>
              </article>
            )
          })}
        </div>
      </section>

      <section className="panel talking-panel">
        <div className="panel-header">
          <span>Talking Points</span>
          <ScanSearch size={16} />
        </div>
        <div className="talking-list">
          {dataset.talkingPoints.map((point) => (
            <article key={point}>
              <div className="talking-dot"></div>
              <p>{point}</p>
            </article>
          ))}
        </div>
      </section>
    </div>
  )
}

export default ResultsView
