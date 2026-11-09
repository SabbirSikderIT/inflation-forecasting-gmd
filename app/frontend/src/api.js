import axios from 'axios'

// In dev/preview, Vite proxies /api to the FastAPI server (see vite.config.js).
// For a separate deployment set VITE_API_URL, e.g. https://api.example.com
const client = axios.create({ baseURL: import.meta.env.VITE_API_URL || '' })

export function errorMessage(err) {
  if (err.response) {
    const d = err.response.data?.detail
    if (typeof d === 'string') return d
    if (Array.isArray(d)) return d.map((x) => x.msg).join('; ')
    return `Server error (${err.response.status}).`
  }
  return 'Cannot reach the backend. Start it with "uvicorn main:app --reload" in the backend folder.'
}

export const getHealth = () => client.get('/api/health').then((r) => r.data)
export const getModels = () => client.get('/api/models').then((r) => r.data)
export const getSamples = () => client.get('/api/samples').then((r) => r.data)
export const getSampleFile = (name) =>
  client
    .get(`/api/samples/${name}`, { responseType: 'blob' })
    .then((r) => new File([r.data], name, { type: 'text/csv' }))
export const getHistory = () =>
  client.get('/api/predictions', { params: { limit: 25 } }).then((r) => r.data)
export const deleteRecord = (id) => client.delete(`/api/predictions/${id}`)
export const clearHistory = () => client.delete('/api/predictions')

export const predict = (file, modelName) => {
  const form = new FormData()
  form.append('file', file)
  form.append('model_name', modelName)
  return client.post('/api/predict', form).then((r) => r.data)
}
