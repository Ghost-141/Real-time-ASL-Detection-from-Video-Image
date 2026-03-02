import { logError } from './errorLogger'

const API_BASE = import.meta.env.VITE_API_HTTP_BASE || 'http://127.0.0.1:8080'
const API_PREFIX = import.meta.env.VITE_API_PREFIX || '/api/v1'

function withPrefix(path) {
  const base = API_BASE.endsWith('/') ? API_BASE.slice(0, -1) : API_BASE
  const prefix = API_PREFIX.endsWith('/') ? API_PREFIX.slice(0, -1) : API_PREFIX
  return `${base}${prefix}${path}`
}

export async function predictImage(file) {
  const form = new FormData()
  form.append('image', file)

  let response
  try {
    response = await fetch(withPrefix('/predict/image'), {
      method: 'POST',
      body: form,
    })
  } catch (error) {
    logError('api.predictImage.network', error, { url: withPrefix('/predict/image') })
    throw error
  }

  if (!response.ok) {
    logError('api.predictImage.http', new Error(`Prediction failed (${response.status})`), {
      status: response.status,
      statusText: response.statusText,
      url: response.url,
    })
    throw new Error(`Prediction failed (${response.status})`)
  }

  return response.json()
}

export async function healthCheck() {
  let response
  try {
    response = await fetch(withPrefix('/health'))
  } catch (error) {
    logError('api.healthCheck.network', error, { url: withPrefix('/health') })
    throw error
  }
  if (!response.ok) {
    logError('api.healthCheck.http', new Error('Health check failed'), {
      status: response.status,
      statusText: response.statusText,
      url: response.url,
    })
    throw new Error('Health check failed')
  }
  return response.json()
}
