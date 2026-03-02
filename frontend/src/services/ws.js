import ReconnectingWebSocket from 'reconnecting-websocket'

const WS_BASE = import.meta.env.VITE_API_WS_BASE || 'ws://127.0.0.1:8080'
const API_PREFIX = import.meta.env.VITE_API_PREFIX || '/api/v1'

function withPrefix(path) {
  let base = WS_BASE
  base = base.endsWith('/') ? base.slice(0, -1) : base
  const prefix = API_PREFIX.endsWith('/') ? API_PREFIX.slice(0, -1) : API_PREFIX
  return `${base}${prefix}${path}`
}

export function createPredictSocket() {
  return new ReconnectingWebSocket(withPrefix('/ws/predict'), [], {
    maxRetries: Infinity,
    minReconnectionDelay: 1000,
    maxReconnectionDelay: 5000,
    reconnectionDelayGrowFactor: 1.4,
  })
}
