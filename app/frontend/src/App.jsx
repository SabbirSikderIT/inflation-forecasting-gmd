import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import {
  clearHistory, deleteRecord, errorMessage, getHealth, getHistory,
  getModels, getSampleFile, getSamples, predict,
} from './api'

const MODEL_COLORS = { RandomForest: '#2f855a', XGBoost: '#c05621', Persistence: '#3182ce' }
const STUDY_COLORS = { 'Random Forest': MODEL_COLORS.RandomForest, XGBoost: MODEL_COLORS.XGBoost, Persistence: MODEL_COLORS.Persistence }
const COUNTRY = { USA: 'United States', GBR: 'United Kingdom', IND: 'India' }
const fmt = (v, d = 2) => (v === null || v === undefined ? '—' : Number(v).toFixed(d))

export default function App() {
  const [health, setHealth] = useState(null) // null = checking, false = offline
  const [info, setInfo] = useState(null)
  const [samples, setSamples] = useState([])
  const [history, setHistory] = useState([])
  const [file, setFile] = useState(null)
  const [model, setModel] = useState('RandomForest')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const fileInput = useRef(null)
  const demoStarted = useRef(false)

  const refreshHistory = useCallback(() => getHistory().then(setHistory).catch(() => {}), [])

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(false))
    getModels().then(setInfo).catch(() => {})
    getSamples().then(setSamples).catch(() => {})
    refreshHistory()
  }, [refreshHistory])


  const run = useCallback(async (f, m) => {
    setError('')
    setLoading(true)
    try {
      const data = await predict(f, m)
      setResult({ ...data, filename: f.name })
      refreshHistory()
    } catch (e) {
      setResult(null)
      setError(errorMessage(e))
    } finally {
      setLoading(false)
    }
  }, [refreshHistory])

  const onPredict = () => {
    if (!file) return setError('Please choose a CSV file first.')
    run(file, model)
  }

  const runSample = useCallback(async (name) => {
    setError('')
    try {
      const f = await getSampleFile(name)
      setFile(f)
      if (fileInput.current) fileInput.current.value = ''
      run(f, model)
    } catch (e) {
      setError(errorMessage(e))
    }
  }, [run, model])

  // Shareable demo link, e.g. http://localhost:5173/#demo=USA
  useEffect(() => {
    const iso = new URLSearchParams(window.location.hash.slice(1)).get('demo')
    if (iso && /^[A-Za-z]{3}$/.test(iso) && !demoStarted.current) {
      demoStarted.current = true
      runSample(`sample_${iso.toUpperCase()}.csv`)
    }
  }, [runSample])

  const best = useMemo(() => {
    if (!info) return null
    return [...info.benchmark].sort((a, b) => b['Mean R²'] - a['Mean R²'])[0]
  }, [info])

  return (
    <div className="page">
      <header className="top">
        <div>
          <h1>Inflation Forecaster</h1>
          <p className="sub">
            Predict the average inflation of the <strong>next 5 years</strong> from 10 years of macro data.
          </p>
        </div>
        <span className={`status ${health ? 'ok' : health === false ? 'down' : ''}`}>
          <i /> {health ? 'API online' : health === false ? 'API offline' : 'Connecting…'}
        </span>
      </header>

      {health === false && (
        <div className="alert">
          The backend is not reachable. In the <code>backend</code> folder run{' '}
          <code>uvicorn main:app --reload</code>, then refresh this page.
        </div>
      )}

      {info && (
        <section className="tiles" aria-label="Project summary">
          <Tile label="Dataset" value="GMD 2025" note={`${info.n_countries} countries · ${info.year_range}`} />
          <Tile label="Input window" value={`${info.window} years`} note={`${info.features.length} macro variables`} />
          <Tile label="Forecast horizon" value={`${info.horizon} years`} note="average inflation, %" />
          <Tile label="Best model (study)" value={best.Model} note={`mean R² = ${fmt(best['Mean R²'], 3)}`} />
        </section>
      )}

      <section className="card">
        <h2>1 · Upload data</h2>
        <p className="hint">
          CSV with the columns <code>year</code>
          {info && info.features.map((f) => <code key={f}>{f}</code>)}. At least 10 consecutive years are needed; the
          most recent 10 are used. <code>infl</code> is annual inflation in %.
        </p>

        <div className="row">
          <label className="file">
            <input
              ref={fileInput}
              type="file"
              accept=".csv,text/csv"
              onChange={(e) => { setFile(e.target.files[0] ?? null); setError('') }}
            />
            <span>{file ? file.name : 'Choose a CSV file…'}</span>
          </label>
          <select value={model} onChange={(e) => setModel(e.target.value)} aria-label="Model">
            {(info?.models ?? [{ key: 'RandomForest', label: 'Random Forest', available: true }]).map((m) => (
              <option key={m.key} value={m.key} disabled={!m.available}>
                {m.label}{m.available ? '' : ' (unavailable)'}
              </option>
            ))}
          </select>
        </div>

        <button className="primary" onClick={onPredict} disabled={loading || health === false}>
          {loading ? 'Predicting…' : 'Predict inflation'}
        </button>

        {samples.length > 0 && (
          <div className="samples">
            <span>No data handy? Try real examples (2011–2020):</span>
            {samples.map((s) => {
              const iso = s.replace('sample_', '').replace('.csv', '')
              return (
                <button key={s} className="chip" onClick={() => runSample(s)} disabled={loading}>
                  {COUNTRY[iso] ?? iso}
                </button>
              )
            })}
            <a className="chip" href={`/api/samples/${samples[0]}`} download>Download CSV template</a>
          </div>
        )}

        {error && <div className="alert" role="alert">{error}</div>}
      </section>

      {result && <Result result={result} />}

      {info && <Benchmark info={info} />}

      <History
        rows={history}
        onDelete={async (id) => { await deleteRecord(id); refreshHistory() }}
        onClear={async () => { await clearHistory(); refreshHistory() }}
      />

      <footer>
        Built on the leakage-free GMD inflation-forecasting benchmark (2-fold temporal cross-validation, seed 42).
        Forecasts are research output, not financial advice.
      </footer>
    </div>
  )
}

function Tile({ label, value, note }) {
  return (
    <div className="tile">
      <span className="tl">{label}</span>
      <strong>{value}</strong>
      <span className="tn">{note}</span>
    </div>
  )
}

function Result({ result }) {
  const { prediction, model, model_label, all_predictions, history, forecast_start, forecast_end, filename } = result
  const last = history[history.length - 1]

  const chartData = history.map((h) => ({ year: h.year, Historical: h.infl }))
  chartData[chartData.length - 1].Forecast = last.infl // join the two lines
  for (let y = forecast_start; y <= forecast_end; y += 1) chartData.push({ year: y, Forecast: prediction })

  const bars = Object.entries(all_predictions).map(([key, value]) => ({
    key, value, name: key === 'RandomForest' ? 'Random Forest' : key,
  }))

  return (
    <section className="card">
      <h2>2 · Forecast</h2>
      <div className="grid2">
        <div className="big">
          <span className="tl">Average inflation {forecast_start}–{forecast_end}</span>
          <strong>{fmt(prediction)}%</strong>
          <span className="badge" style={{ background: MODEL_COLORS[model] }}>{model_label}</span>
          <span className="tn">from {filename}</span>
        </div>
        <div>
          <h3>All models, same data</h3>
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={bars} layout="vertical" margin={{ left: 10, right: 40 }}>
              <XAxis type="number" hide domain={[0, 'dataMax + 1']} />
              <YAxis type="category" dataKey="name" width={110} tickLine={false} axisLine={false} />
              <Tooltip formatter={(v) => `${fmt(v)}%`} cursor={{ fill: 'rgba(128,128,128,.12)' }} />
              <Bar
                dataKey="value"
                radius={4}
                isAnimationActive={false}
                label={{ position: 'right', formatter: (v) => `${fmt(v)}%`, fill: 'currentColor' }}
              >
                {bars.map((b) => (
                  <Cell key={b.key} fill={MODEL_COLORS[b.key]} opacity={b.key === model ? 1 : 0.55} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <h3>Inflation history and forecast</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis dataKey="year" type="number" domain={['dataMin', 'dataMax']} ticks={chartData.map((d) => d.year)} />
          <YAxis unit="%" domain={[(min) => Math.min(0, Math.floor(min)), (max) => Math.ceil(max + 1)]} />
          <Tooltip formatter={(v) => `${fmt(v)}%`} />
          <Legend />
          <Line type="linear" dataKey="Historical" name="Historical inflation" stroke="#718096" strokeWidth={2} dot={{ r: 3 }} connectNulls isAnimationActive={false} />
          <Line type="linear" dataKey="Forecast" name="5-year average forecast" stroke={MODEL_COLORS[model]} strokeWidth={2.5} strokeDasharray="6 4" dot={false} connectNulls isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
      <p className="hint">
        The dashed line is the forecast of the <em>average</em> inflation over {forecast_start}–{forecast_end}: one
        number, drawn flat. Forecasts cannot anticipate shocks.
      </p>
    </section>
  )
}

function Benchmark({ info }) {
  const rows = [...info.benchmark].sort((a, b) => b['Mean R²'] - a['Mean R²'])
  return (
    <section className="card">
      <h2>Study results · model comparison</h2>
      <p className="hint">
        Out-of-sample R² from 2-fold temporal cross-validation (Fold 1: test 2006–2015, Fold 2: test 2016–2020). The
        simple persistence benchmark (average of the last 5 years) is the most robust.
      </p>
      <div className="grid2">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={rows} layout="vertical" margin={{ left: 20, right: 30 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} horizontal={false} />
            <XAxis type="number" domain={[-0.3, 0.8]} />
            <YAxis type="category" dataKey="Model" width={120} tickLine={false} />
            <Tooltip formatter={(v) => fmt(v, 3)} cursor={{ fill: 'rgba(128,128,128,.12)' }} />
            <Bar dataKey="Mean R²" radius={4} isAnimationActive={false}>
              {rows.map((r) => <Cell key={r.Model} fill={STUDY_COLORS[r.Model] ?? '#718096'} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="tablewrap">
          <table>
            <thead>
              <tr><th>Model</th><th>Fold 1</th><th>Fold 2</th><th>Mean R²</th></tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.Model}>
                  <td>{r.Model}</td>
                  <td>{fmt(r['Fold 1 R²'], 3)}</td>
                  <td>{fmt(r['Fold 2 R²'], 3)}</td>
                  <td><strong>{fmt(r['Mean R²'], 3)}</strong></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  )
}

function History({ rows, onDelete, onClear }) {
  if (rows.length === 0) return null
  return (
    <section className="card">
      <div className="between">
        <h2>Recent predictions</h2>
        <button className="chip danger" onClick={() => window.confirm('Delete all saved predictions?') && onClear()}>
          Clear all
        </button>
      </div>
      <div className="tablewrap">
        <table>
          <thead>
            <tr><th>When</th><th>File</th><th>Last year</th><th>Model</th><th>Forecast</th><th /></tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id}>
                <td>{new Date(r.created_at).toLocaleString()}</td>
                <td>{r.filename || '—'}</td>
                <td>{r.last_year}</td>
                <td>{r.model}</td>
                <td><strong>{fmt(r.prediction)}%</strong></td>
                <td><button className="x" aria-label="Delete" onClick={() => onDelete(r.id)}>×</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
