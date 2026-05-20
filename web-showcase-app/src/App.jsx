import { useEffect, useState } from 'react'
import { ChevronRight } from 'lucide-react'
import './App.css'
import { datasets, featureGallery, moduleCards, supportingCharts } from './showcaseData'
import { datasetOptions, mapOptions, views } from './config/uiConfig'
import ImageLightbox from './components/ImageLightbox'
import OverviewView from './views/OverviewView'
import MethodsView from './views/MethodsView'
import ResultsView from './views/ResultsView'
import MapsView from './views/MapsView'
import DemoView from './views/DemoView'

function App() {
  const [activeView, setActiveView] = useState('overview')
  const [selectedDataset, setSelectedDataset] = useState('houston2013')
  const [selectedMetric, setSelectedMetric] = useState('oa')
  const [primaryMap, setPrimaryMap] = useState('full')
  const [secondaryMap, setSecondaryMap] = useState('baseline')
  const [selectedFeature, setSelectedFeature] = useState('improvement')
  const [selectedChart, setSelectedChart] = useState('metrics')
  const [selectedModule, setSelectedModule] = useState(moduleCards[1].title)
  const [demoData, setDemoData] = useState(null)
  const [demoState, setDemoState] = useState('idle')
  const [selectedPixel, setSelectedPixel] = useState(null)
  const [previewItem, setPreviewItem] = useState(null)

  useEffect(() => {
    let cancelled = false

    async function loadDemo() {
      try {
        const response = await fetch('/showcase/demo/Houston2013/roi_metadata.json')
        if (!response.ok) {
          throw new Error(`Failed to load demo metadata: ${response.status}`)
        }
        const metadata = await response.json()
        if (!cancelled) {
          setDemoData(metadata)
          setSelectedPixel({
            x: Math.floor(metadata.roi.width * 0.56),
            y: Math.floor(metadata.roi.height * 0.5),
          })
        }
      } catch (error) {
        if (!cancelled) {
          setDemoData({ error: error.message })
        }
      }
    }

    loadDemo()
    return () => {
      cancelled = true
    }
  }, [])

  const dataset = datasets[selectedDataset]
  const activeFeature = featureGallery.find((item) => item.key === selectedFeature) ?? featureGallery[0]
  const activeChart = supportingCharts.find((item) => item.key === selectedChart) ?? supportingCharts[0]
  const selectedModuleInfo = moduleCards.find((item) => item.title === selectedModule) ?? moduleCards[0]

  const methodMetrics = dataset.methods.map((method) => ({
    ...method,
    value: method[selectedMetric],
  }))

  const bestMethod = [...dataset.methods].sort((left, right) => right[selectedMetric] - left[selectedMetric])[0]
  const mapTabs = mapOptions.filter((option) => dataset.previewMaps[option.key])
  const primaryMapSrc = dataset.previewMaps[primaryMap] ?? dataset.previewMaps[mapTabs[0].key]
  const secondaryMapSrc = dataset.previewMaps[secondaryMap] ?? dataset.previewMaps[mapTabs[0].key]

  function runInferenceDemo() {
    setDemoState('running')
    window.setTimeout(() => {
      setDemoState('done')
    }, 1400)
  }

  return (
    <>
      <div className="app-shell">
        <div className="ambient ambient-a"></div>
        <div className="ambient ambient-b"></div>

        <aside className="sidebar">
          <div className="brand-block">
            <div className="brand-icon">RS</div>
            <div>
              <p>本科毕业设计应用</p>
              <strong>Multimodal Land Cover Lab</strong>
            </div>
          </div>

          <div className="sidebar-section">
            <span className="sidebar-label">Workspace</span>
            <div className="nav-list">
              {views.map((view) => {
                const Icon = view.icon
                return (
                  <button
                    key={view.key}
                    type="button"
                    className={`nav-button ${activeView === view.key ? 'active' : ''}`}
                    onClick={() => setActiveView(view.key)}
                  >
                    <Icon size={18} />
                    <span>{view.label}</span>
                    <ChevronRight size={16} />
                  </button>
                )
              })}
            </div>
          </div>

          <div className="sidebar-section sidebar-dataset">
            <span className="sidebar-label">Dataset</span>
            <div className="dataset-pill-group">
              {datasetOptions.map((option) => (
                <button
                  key={option.key}
                  type="button"
                  className={selectedDataset === option.key ? 'active' : ''}
                  onClick={() => setSelectedDataset(option.key)}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          <div className="sidebar-footer">
            <div className="footer-card">
              <span>当前模块</span>
              <strong>{views.find((view) => view.key === activeView)?.label}</strong>
              <p>当前版本优先展示 Houston2013 主结果与实际 ROI 推理流程。</p>
            </div>
          </div>
        </aside>

        <main className="workspace">
          <header className="workspace-topbar">
            <div>
              <p className="eyebrow">Multimodal Remote Sensing</p>
              <h1>{views.find((view) => view.key === activeView)?.label}</h1>
            </div>
            <div className="status-badges">
              <span>{activeView === 'demo' ? 'Houston2013 ROI Demo' : dataset.label}</span>
              <span>Full Model v1</span>
              <span>Presentation Ready</span>
            </div>
          </header>

          <section className="workspace-stage">
            {activeView === 'overview' && (
              <OverviewView
                dataset={dataset}
                onOpenResults={() => setActiveView('results')}
                onOpenMaps={() => setActiveView('maps')}
              />
            )}
            {activeView === 'methods' && (
              <MethodsView
                selectedModule={selectedModule}
                onSelectModule={setSelectedModule}
                selectedModuleInfo={selectedModuleInfo}
              />
            )}
            {activeView === 'results' && (
              <ResultsView
                dataset={dataset}
                selectedMetric={selectedMetric}
                onSelectMetric={setSelectedMetric}
                methodMetrics={methodMetrics}
                bestMethod={bestMethod}
              />
            )}
            {activeView === 'maps' && (
              <MapsView
                dataset={dataset}
                mapTabs={mapTabs}
                primaryMap={primaryMap}
                secondaryMap={secondaryMap}
                onPrimaryChange={setPrimaryMap}
                onSecondaryChange={setSecondaryMap}
                primaryMapSrc={primaryMapSrc}
                secondaryMapSrc={secondaryMapSrc}
                selectedFeature={selectedFeature}
                onSelectFeature={setSelectedFeature}
                activeFeature={activeFeature}
                selectedChart={selectedChart}
                onSelectChart={setSelectedChart}
                activeChart={activeChart}
                onOpenMapPreview={setPreviewItem}
              />
            )}
            {activeView === 'demo' && (
              <DemoView
                demoData={demoData}
                demoState={demoState}
                onRun={runInferenceDemo}
                selectedPixel={selectedPixel}
                onSelectPixel={setSelectedPixel}
              />
            )}
          </section>
        </main>
      </div>

      <ImageLightbox item={previewItem} onClose={() => setPreviewItem(null)} />
    </>
  )
}

export default App
