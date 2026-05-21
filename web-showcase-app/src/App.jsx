import { useEffect, useMemo, useState } from 'react'
import { ArrowRight, Database, Sparkles } from 'lucide-react'
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
  const activeViewInfo = views.find((view) => view.key === activeView) ?? views[0]

  const methodMetrics = useMemo(
    () =>
      dataset.methods.map((method) => ({
        ...method,
        value: method[selectedMetric],
      })),
    [dataset.methods, selectedMetric],
  )

  const bestMethod = [...dataset.methods].sort((left, right) => right[selectedMetric] - left[selectedMetric])[0]
  const mapTabs = mapOptions.filter((option) => dataset.previewMaps[option.key])
  const primaryMapSrc = dataset.previewMaps[primaryMap] ?? dataset.previewMaps[mapTabs[0].key]
  const secondaryMapSrc = dataset.previewMaps[secondaryMap] ?? dataset.previewMaps[mapTabs[0].key]
  const topStats = [
    { label: 'OA', value: dataset.heroStats.oa.toFixed(2) },
    { label: 'AA', value: dataset.heroStats.aa.toFixed(2) },
    { label: 'Kappa', value: dataset.heroStats.kappa.toFixed(2) },
    { label: 'Macro-F1', value: dataset.heroStats.macroF1.toFixed(2) },
  ]

  function runInferenceDemo() {
    setDemoState('running')
    window.setTimeout(() => {
      setDemoState('done')
    }, 1400)
  }

  return (
    <>
      <div className="app-shell">
        <aside className="sidebar">
          <div className="brand-block">
            <div className="brand-icon">RS</div>
            <div className="brand-copy">
              <strong>Land Cover Lab</strong>
              <p>Showcase</p>
            </div>
          </div>

          <nav className="nav-list" aria-label="Workspace views">
            {views.map((view) => {
              const Icon = view.icon
              return (
                <button
                  key={view.key}
                  type="button"
                  className={`nav-button ${activeView === view.key ? 'active' : ''}`}
                  onClick={() => setActiveView(view.key)}
                >
                  <span className="nav-icon">
                    <Icon size={17} />
                  </span>
                  <span className="nav-text">{view.label}</span>
                </button>
              )
            })}
          </nav>

          <div className="sidebar-footer">
            <span>{dataset.label}</span>
            <strong>{bestMethod.name}</strong>
          </div>
        </aside>

        <main className="workspace">
          <header className="workspace-topbar">
            <div className="workspace-title-block">
              <span className="workspace-kicker">Research Workspace</span>
              <h1>{activeViewInfo.label}</h1>
              <p>{activeViewInfo.summary}</p>
            </div>

            <div className="workspace-controls">
              <div className="dataset-switch" aria-label="Dataset switcher">
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

              <div className="header-actions">
                <button type="button" className="ghost-button" onClick={() => setActiveView('results')}>
                  Results
                </button>
                <button type="button" className="primary-button" onClick={() => setActiveView('demo')}>
                  Demo
                  <ArrowRight size={15} />
                </button>
              </div>
            </div>
          </header>

          <section className="workspace-ribbon">
            <article className="ribbon-card ribbon-card-wide ribbon-card-dataset">
              <div className="ribbon-head">
                <Database size={15} />
                <span>{dataset.highlight}</span>
              </div>
              <strong>{dataset.label}</strong>
            </article>
            {topStats.map((item) => (
              <article key={item.label} className="ribbon-stat">
                <span>{item.label}</span>
                <strong>{item.value}%</strong>
              </article>
            ))}
            <article className="ribbon-card ribbon-card-current">
              <div className="ribbon-head">
                <Sparkles size={15} />
                <span>Current</span>
              </div>
              <strong>{bestMethod.name}</strong>
            </article>
          </section>

          <section className="workspace-stage">
            {renderActiveView({
              activeView,
              dataset,
              selectedMetric,
              setSelectedMetric,
              methodMetrics,
              bestMethod,
              selectedModule,
              setSelectedModule,
              selectedModuleInfo,
              mapTabs,
              primaryMap,
              secondaryMap,
              setPrimaryMap,
              setSecondaryMap,
              primaryMapSrc,
              secondaryMapSrc,
              selectedFeature,
              setSelectedFeature,
              activeFeature,
              selectedChart,
              setSelectedChart,
              activeChart,
              setPreviewItem,
              demoData,
              demoState,
              runInferenceDemo,
              selectedPixel,
              setSelectedPixel,
              setActiveView,
            })}
          </section>
        </main>
      </div>

      <ImageLightbox item={previewItem} onClose={() => setPreviewItem(null)} />
    </>
  )
}

function renderActiveView({
  activeView,
  dataset,
  selectedMetric,
  setSelectedMetric,
  methodMetrics,
  bestMethod,
  selectedModule,
  setSelectedModule,
  selectedModuleInfo,
  mapTabs,
  primaryMap,
  secondaryMap,
  setPrimaryMap,
  setSecondaryMap,
  primaryMapSrc,
  secondaryMapSrc,
  selectedFeature,
  setSelectedFeature,
  activeFeature,
  selectedChart,
  setSelectedChart,
  activeChart,
  setPreviewItem,
  demoData,
  demoState,
  runInferenceDemo,
  selectedPixel,
  setSelectedPixel,
  setActiveView,
}) {
  if (activeView === 'overview') {
    return (
      <OverviewView dataset={dataset} onOpenResults={() => setActiveView('results')} onOpenMaps={() => setActiveView('maps')} />
    )
  }

  if (activeView === 'methods') {
    return (
      <MethodsView
        selectedModule={selectedModule}
        onSelectModule={setSelectedModule}
        selectedModuleInfo={selectedModuleInfo}
      />
    )
  }

  if (activeView === 'results') {
    return (
      <ResultsView
        dataset={dataset}
        selectedMetric={selectedMetric}
        onSelectMetric={setSelectedMetric}
        methodMetrics={methodMetrics}
        bestMethod={bestMethod}
      />
    )
  }

  if (activeView === 'maps') {
    return (
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
    )
  }

  return (
    <DemoView
      demoData={demoData}
      demoState={demoState}
      onRun={runInferenceDemo}
      selectedPixel={selectedPixel}
      onSelectPixel={setSelectedPixel}
    />
  )
}

export default App
