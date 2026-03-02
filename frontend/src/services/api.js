const API_BASE = import.meta.env.VITE_API_HTTP_BASE || 'http://127.0.0.1:8080'
const API_PREFIX = import.meta.env.VITE_API_PREFIX || ''

function withPrefix(path) {
  const base = API_BASE.endsWith('/') ? API_BASE.slice(0, -1) : API_BASE
  const prefix = API_PREFIX.endsWith('/') ? API_PREFIX.slice(0, -1) : API_PREFIX
  return `${base}${prefix}${path}`
}

export async function predictImage(file) {
  const form = new FormData()
  form.append('image', file)

  const response = await fetch(withPrefix('/predict/image'), {
    method: 'POST',
    body: form,
  })

  if (!response.ok) {
    throw new Error(`Prediction failed (${response.status})`)
  }

  return response.json()
}

export async function healthCheck() {
  const response = await fetch(withPrefix('/health'))
  if (!response.ok) {
    throw new Error('Health check failed')
  }
  return response.json()
}
