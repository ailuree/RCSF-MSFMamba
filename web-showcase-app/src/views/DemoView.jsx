import { MonitorPlay, MousePointerClick, Sparkles } from 'lucide-react'
import { inferenceNarrative } from '../config/uiConfig'

function DemoView({ demoData, demoState, onRun, selectedPixel, onSelectPixel }) {
  if (!demoData) {
    return (
      <div className="view-grid demo-grid">
        <section className="panel demo-stage-panel demo-empty-panel">
          <div className="panel-header">
            <span>Loading Demo</span>
            <MonitorPlay size={16} />
          </div>
          <div className="demo-placeholder">
            <strong>正在读取 Houston2013 样例推理资产</strong>
            <p>稍后会显示输入图、推理按钮和分类结果图。</p>
          </div>
        </section>
      </div>
    )
  }

  if (demoData.error) {
    return (
      <div className="view-grid demo-grid">
        <section className="panel demo-stage-panel demo-empty-panel">
          <div className="panel-header">
            <span>Demo Error</span>
            <MonitorPlay size={16} />
          </div>
          <div className="demo-placeholder">
            <strong>样例推理资产读取失败</strong>
            <p>{demoData.error}</p>
          </div>
        </section>
      </div>
    )
  }

  const activeClass = getPixelInfo(demoData, selectedPixel)
  const isDone = demoState === 'done'
  const isRunning = demoState === 'running'

  return (
    <div className="view-grid demo-grid">
      <section className="panel demo-script-panel">
        <div className="panel-header">
          <span>Inference Flow</span>
          <MonitorPlay size={16} />
        </div>
        <div className="demo-callout">
          <strong>Houston2013 实际推理 ROI</strong>
          <p>
            当前展示的是 Full Model v1 在 <span>160 × 160</span> ROI 上的逐像素密集推理结果，不是静态示意图。
          </p>
        </div>
        <div className="demo-step-list">
          {inferenceNarrative.map((item, index) => (
            <article key={item} className={`demo-note ${isDone || index === 0 ? 'active' : ''}`}>
              <span>{`0${index + 1}`}</span>
              <p>{item}</p>
            </article>
          ))}
        </div>
        <div className="demo-cta-group">
          <button type="button" className="primary-button" onClick={onRun} disabled={isRunning}>
            {isRunning ? 'Inferencing...' : isDone ? 'Run Again' : 'Run Inference'}
          </button>
          <p>建议答辩时先看左侧输入，再点击按钮切出右侧分类图。</p>
        </div>
      </section>

      <section className="panel demo-stage-panel">
        <div className="panel-header">
          <span>Input / Output Workspace</span>
          <MousePointerClick size={16} />
        </div>
        <div className="demo-stage-grid">
          <article className="demo-image-card">
            <span>HSI pseudo-RGB</span>
            <img src="/showcase/demo/Houston2013/roi_hsi.png" alt="Houston2013 ROI HSI pseudo RGB" />
          </article>
          <article className="demo-image-card">
            <span>LiDAR view</span>
            <img src="/showcase/demo/Houston2013/roi_aux.png" alt="Houston2013 ROI LiDAR view" />
          </article>
          <article className="demo-image-card demo-output-card">
            <span>Predicted land-cover map</span>
            <SelectablePredictionMap
              demoData={demoData}
              visible={isDone}
              selectedPixel={selectedPixel}
              onSelectPixel={onSelectPixel}
            />
          </article>
          <article className="demo-image-card">
            <span>Reference sparse GT</span>
            <img src="/showcase/demo/Houston2013/roi_gt.png" alt="Houston2013 ROI sparse ground truth" />
          </article>
        </div>
      </section>

      <section className="panel demo-advice-panel">
        <div className="panel-header">
          <span>Class Explanation</span>
          <Sparkles size={16} />
        </div>
        <div className="pixel-info-card">
          <strong>{activeClass?.classInfo?.name ?? '尚未选择像素'}</strong>
          <p>
            {activeClass
              ? `位置 (${activeClass.x}, ${activeClass.y}) 的预测类别为 ${activeClass.classInfo.name}，置信度 ${activeClass.confidence.toFixed(2)}。`
              : '点击预测图上的任意位置，页面会解释该像素被分类成什么地物。'}
          </p>
          {activeClass && (
            <div className="pixel-meta-row">
              <span style={{ backgroundColor: activeClass.classInfo.color }}></span>
              <small>{activeClass.classInfo.color}</small>
            </div>
          )}
        </div>
        <div className="legend-list">
          {(demoData.top_classes ?? []).map((item) => (
            <article key={item.id}>
              <div className="legend-swatch" style={{ backgroundColor: item.color }}></div>
              <div>
                <strong>{item.name}</strong>
                <p>{`${(item.ratio * 100).toFixed(1)}% of ROI`}</p>
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  )
}

function SelectablePredictionMap({ demoData, visible, selectedPixel, onSelectPixel }) {
  function handleMapClick(event) {
    if (!visible) {
      return
    }
    const rect = event.currentTarget.getBoundingClientRect()
    const x = Math.max(0, Math.min(demoData.roi.width - 1, Math.floor(((event.clientX - rect.left) / rect.width) * demoData.roi.width)))
    const y = Math.max(0, Math.min(demoData.roi.height - 1, Math.floor(((event.clientY - rect.top) / rect.height) * demoData.roi.height)))
    onSelectPixel({ x, y })
  }

  return (
    <button type="button" className={`prediction-map-button ${visible ? 'visible' : ''}`} onClick={handleMapClick}>
      {visible ? (
        <>
          <img src="/showcase/demo/Houston2013/roi_prediction.png" alt="Houston2013 ROI prediction map" />
          {selectedPixel && (
            <div
              className="prediction-cursor"
              style={{
                left: `${(selectedPixel.x / demoData.roi.width) * 100}%`,
                top: `${(selectedPixel.y / demoData.roi.height) * 100}%`,
              }}
            ></div>
          )}
        </>
      ) : (
        <div className="prediction-placeholder">
          <MonitorPlay size={20} />
          <strong>{visible ? 'Prediction Ready' : 'Result Hidden'}</strong>
          <p>{visible ? '点击查看类别' : '点击 Run Inference 后显示分类图'}</p>
        </div>
      )}
    </button>
  )
}

function getPixelInfo(demoData, selectedPixel) {
  if (!demoData || !selectedPixel) {
    return null
  }

  const row = demoData.prediction_labels[selectedPixel.y]
  const confidenceRow = demoData.confidence_map[selectedPixel.y]
  if (!row || !confidenceRow) {
    return null
  }

  const labelId = row[selectedPixel.x]
  const classInfo = demoData.classes.find((item) => item.id === labelId)
  if (!classInfo) {
    return null
  }

  return {
    x: selectedPixel.x,
    y: selectedPixel.y,
    labelId,
    confidence: confidenceRow[selectedPixel.x],
    classInfo,
  }
}

export default DemoView
