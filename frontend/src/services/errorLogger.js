const STORAGE_KEY = 'asl_frontend_error_logs'
const MAX_LOGS = 200

function safeSerialize(value) {
  try {
    return JSON.parse(JSON.stringify(value))
  } catch {
    return String(value)
  }
}

function toErrorInfo(error) {
  if (!error) {
    return { message: 'Unknown error' }
  }

  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
      stack: error.stack || '',
    }
  }

  return {
    message: typeof error === 'string' ? error : JSON.stringify(error),
  }
}

function readLogs() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function writeLogs(logs) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(logs.slice(-MAX_LOGS)))
  } catch {
    // Ignore localStorage failures and keep console output.
  }
}

export function logError(source, error, context = {}) {
  const entry = {
    ts: new Date().toISOString(),
    source,
    error: toErrorInfo(error),
    context: safeSerialize(context),
    href: window.location.href,
    userAgent: window.navigator.userAgent,
  }

  const logs = readLogs()
  logs.push(entry)
  writeLogs(logs)

  console.error(`[frontend-error] ${source}`, entry)
}

export function getErrorLogs() {
  return readLogs()
}

export function clearErrorLogs() {
  writeLogs([])
}

let installed = false
export function installGlobalErrorLogger() {
  if (installed) return
  installed = true

  // Debug helper for browser devtools.
  window.__ASL_ERROR_LOGGER__ = {
    getErrorLogs,
    clearErrorLogs,
  }

  window.addEventListener('error', (event) => {
    logError('window.error', event.error || event.message, {
      filename: event.filename,
      lineno: event.lineno,
      colno: event.colno,
    })
  })

  window.addEventListener('unhandledrejection', (event) => {
    logError('window.unhandledrejection', event.reason)
  })
}
