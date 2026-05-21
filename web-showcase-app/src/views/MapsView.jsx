import { ArrowRightLeft, Eye, Map, Radar, ZoomIn } from 'lucide-react'
import { featureGallery, supportingCharts } from '../showcaseData'
import { mapOptions } from '../config/uiConfig'

function MapsView({
  dataset,
  mapTabs,
  primaryMap,
  secondaryMap,
  onPrimaryChange,
  onSecondaryChange,
  primaryMapSrc,
  secondaryMapSrc,
  selectedFeature,
  onSelectFeature,
  activeFeature,
  selectedChart,
  onSelectChart,
  activeChart,
  onOpenMapPreview,
}) {
  const leftLabel = mapOptions.find((item) => item.key === primaryMap)?.label
  const rightLabel = mapOptions.find((item) => item.key === secondaryMap)?.label

  return (
    <div className="view-grid maps-grid">
      <section className="panel panel-hero-strip">
        <div className="panel-header">
          <span>Map Signals</span>
          <Eye size={16} />
        </div>
        <div className="signal-grid">
          <article className="signal-card">
            <span>Left View</span>
            <strong>{leftLabel}</strong>
          </article>
          <article className="signal-card">
            <span>Right View</span>
            <strong>{rightLabel}</strong>
          </article>
          <article className="signal-card">
            <span>Dataset</span>
            <strong>{dataset.label}</strong>
          </article>
        </div>
      </section>

      <section className="panel compare-panel">
        <div className="panel-header">
          <span>Map Comparator</span>
          <ArrowRightLeft size={16} />
        </div>
        <div className="dual-control-row">
          <div>
            <p>左侧视图</p>
            <div className="control-strip">
              {mapTabs.map((option) => (
                <button
                  key={`left-${option.key}`}
                  type="button"
                  className={primaryMap === option.key ? 'active' : ''}
                  onClick={() => onPrimaryChange(option.key)}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
          <div>
            <p>右侧视图</p>
            <div className="control-strip">
              {mapTabs.map((option) => (
                <button
                  key={`right-${option.key}`}
                  type="button"
                  className={secondaryMap === option.key ? 'active' : ''}
                  onClick={() => onSecondaryChange(option.key)}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
        </div>
        <div className="map-compare-stage">
          <MapPreviewCard
            title={leftLabel}
            imageSrc={primaryMapSrc}
            altText={`${dataset.label} ${primaryMap}`}
            onOpenPreview={() =>
              onOpenMapPreview({
                title: `${dataset.label} · ${leftLabel}`,
                src: primaryMapSrc,
                note: '放大查看局部边界与类别分布。',
              })
            }
          />
          <MapPreviewCard
            title={rightLabel}
            imageSrc={secondaryMapSrc}
            altText={`${dataset.label} ${secondaryMap}`}
            onOpenPreview={() =>
              onOpenMapPreview({
                title: `${dataset.label} · ${rightLabel}`,
                src: secondaryMapSrc,
                note: '用于对比不同方法的局部差异。',
              })
            }
          />
        </div>
      </section>

      <section className="panel gallery-panel">
        <div className="panel-header">
          <span>ROI / Figure Viewer</span>
          <Map size={16} />
        </div>
        <div className="control-strip">
          {featureGallery.map((item) => (
            <button
              key={item.key}
              type="button"
              className={selectedFeature === item.key ? 'active' : ''}
              onClick={() => onSelectFeature(item.key)}
            >
              {item.title}
            </button>
          ))}
        </div>
        <div className="viewer-image">
          <img src={activeFeature.image} alt={activeFeature.title} />
        </div>
        <p className="viewer-note">{activeFeature.description}</p>
        <div className="preview-callout">
          <div className="preview-callout-head">
            <span>Feature Lens</span>
            <Map size={15} />
          </div>
          <strong>{activeFeature.title}</strong>
          <p>用这张图解释 Full Model v1 到底在什么位置修正了基线的错误。</p>
        </div>
      </section>

      <section className="panel supporting-panel">
        <div className="panel-header">
          <span>Supporting Charts</span>
          <Radar size={16} />
        </div>
        <div className="control-strip">
          {supportingCharts.map((item) => (
            <button
              key={item.key}
              type="button"
              className={selectedChart === item.key ? 'active' : ''}
              onClick={() => onSelectChart(item.key)}
            >
              {item.title}
            </button>
          ))}
        </div>
        <div className="viewer-image compact">
          <img src={activeChart.image} alt={activeChart.title} />
        </div>
        <p className="viewer-note">{activeChart.note}</p>
      </section>
    </div>
  )
}

function MapPreviewCard({ title, imageSrc, altText, onOpenPreview }) {
  return (
    <article className="map-preview-card">
      <div className="map-preview-header">
        <span>{title}</span>
        <button type="button" className="map-preview-action" onClick={onOpenPreview}>
          <ZoomIn size={14} />
          <span>点击放大</span>
        </button>
      </div>
      <button type="button" className="map-preview-button" onClick={onOpenPreview}>
        <img src={imageSrc} alt={altText} />
      </button>
    </article>
  )
}

export default MapsView
