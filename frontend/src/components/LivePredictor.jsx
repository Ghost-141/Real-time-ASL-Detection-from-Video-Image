import { useEffect, useRef, useState } from 'react'
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision'
import { createPredictSocket } from '../services/ws'

const LANDMARK_FPS = 15

function LivePredictor() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const landmarksRef = useRef(null)
  const socketRef = useRef(null)
  const timerRef = useRef(null)
  const landmarkerRef = useRef(null)
  const rafRef = useRef(null)
  const lastLandmarkTsRef = useRef(0)

  const [connected, setConnected] = useState(false)
  const [result, setResult] = useState({ pred: 'nothing', confidence: 0, hand_detected: false })
  const [error, setError] = useState('')
  const [mirrorView, setMirrorView] = useState(true)
  const [threshold, setThreshold] = useState(0.55)
  const [sendFps, setSendFps] = useState(12)
  const [smoothingWindow, setSmoothingWindow] = useState(15)
  const [showFps, setShowFps] = useState(true)
  const [showLandmarks, setShowLandmarks] = useState(false)

  const sendFrame = () => {
    if (!videoRef.current || !canvasRef.current || !socketRef.current) return
    if (socketRef.current.readyState !== WebSocket.OPEN) return

    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')

    canvas.width = video.videoWidth || 640
    canvas.height = video.videoHeight || 480

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    const frame = canvas.toDataURL('image/jpeg', 0.8)
    socketRef.current.send(JSON.stringify({ frame }))
  }

  const sendControl = (control) => {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) return
    socketRef.current.send(JSON.stringify({ control }))
  }

  const drawLandmarks = (landmarks) => {
    const video = videoRef.current
    const canvas = landmarksRef.current
    if (!video || !canvas) return

    const width = video.videoWidth || 640
    const height = video.videoHeight || 480
    canvas.width = width
    canvas.height = height

    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, width, height)
    if (!landmarks || landmarks.length === 0) return

    const edges = [
      [0, 1], [1, 2], [2, 3], [3, 4],
      [0, 5], [5, 6], [6, 7], [7, 8],
      [0, 9], [9, 10], [10, 11], [11, 12],
      [0, 13], [13, 14], [14, 15], [15, 16],
      [0, 17], [17, 18], [18, 19], [19, 20],
    ]

    ctx.lineWidth = 2
    ctx.strokeStyle = 'rgba(100, 200, 255, 0.9)'
    ctx.fillStyle = 'rgba(100, 200, 255, 0.9)'

    for (const [a, b] of edges) {
      const pa = landmarks[a]
      const pb = landmarks[b]
      if (!pa || !pb) continue
      ctx.beginPath()
      ctx.moveTo(pa.x * width, pa.y * height)
      ctx.lineTo(pb.x * width, pb.y * height)
      ctx.stroke()
    }

    for (const p of landmarks) {
      ctx.beginPath()
      ctx.arc(p.x * width, p.y * height, 3, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  useEffect(() => {
    let stream

    async function start() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play()
        }
        connectSocket()
      } catch (err) {
        setError('Camera access failed')
      }
    }

    function connectSocket() {
      const ws = createPredictSocket()
      socketRef.current = ws

      ws.onopen = () => {
        setError('')
        setConnected(true)
      }

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data)
          if (payload.pred) {
            setResult(payload)
          } else if (payload.detail) {
            setError(payload.detail)
          }
        } catch {
          setError('Invalid response from server')
        }
      }

      ws.onerror = () => setError('WebSocket error')
      ws.onclose = (event) => {
        setConnected(false)
        const reason = event?.reason ? ` (${event.reason})` : ''
        setError(`WebSocket disconnected: ${event?.code ?? 'unknown'}${reason}`)
        if (timerRef.current) {
          clearInterval(timerRef.current)
          timerRef.current = null
        }
      }
    }

    start()

    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
      if (socketRef.current) socketRef.current.close(1000, 'Component unmounted')
      if (stream) {
        stream.getTracks().forEach((t) => t.stop())
      }
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [])

  useEffect(() => {
    if (!connected) return
    if (timerRef.current) clearInterval(timerRef.current)
    const interval = Math.max(1, Math.round(1000 / Math.max(1, sendFps)))
    timerRef.current = setInterval(sendFrame, interval)
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [connected, sendFps])

  useEffect(() => {
    if (!connected) return
    sendControl({
      smoothing_window: smoothingWindow,
      confidence_threshold: threshold,
      send_landmarks: false,
    })
  }, [connected, smoothingWindow, threshold])

  useEffect(() => {
    if (!showLandmarks) {
      const canvas = landmarksRef.current
      if (canvas) {
        const ctx = canvas.getContext('2d')
        ctx.clearRect(0, 0, canvas.width, canvas.height)
      }
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      return
    }

    let cancelled = false

    async function initLandmarker() {
      if (landmarkerRef.current) return
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
      )
      const apiBase = import.meta.env.VITE_API_HTTP_BASE || 'http://127.0.0.1:8080'
      const modelUrl = `${apiBase.replace(/\\/$/, '')}/weights/hand_landmarker.task`
      landmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: modelUrl,
        },
        runningMode: 'VIDEO',
        numHands: 1,
        minHandDetectionConfidence: 0.3,
        minTrackingConfidence: 0.5,
        minHandPresenceConfidence: 0.3,
      })
    }

    const loop = async () => {
      if (cancelled) return
      if (!landmarkerRef.current || !videoRef.current) {
        rafRef.current = requestAnimationFrame(loop)
        return
      }

      const video = videoRef.current
      if (video.readyState < 2) {
        rafRef.current = requestAnimationFrame(loop)
        return
      }

      const now = performance.now()
      if (now - lastLandmarkTsRef.current < 1000 / LANDMARK_FPS) {
        rafRef.current = requestAnimationFrame(loop)
        return
      }
      lastLandmarkTsRef.current = now

      const result = landmarkerRef.current.detectForVideo(video, now)
      const landmarks = result?.landmarks?.[0] || []
      drawLandmarks(landmarks)

      rafRef.current = requestAnimationFrame(loop)
    }

    initLandmarker()
      .then(() => {
        if (!cancelled) {
          rafRef.current = requestAnimationFrame(loop)
        }
      })
      .catch((err) => {
        setError(`Landmark init failed: ${err.message || err}`)
      })

    return () => {
      cancelled = true
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [showLandmarks])

  const confidencePct = Math.round(result.confidence * 100)

  return (
    <section className="live-page">
      <header className="live-header">
        <div className="brand">
          <span className="brand-icon">ASL</span>
          <div>
            <h2>ASL Recognizer</h2>
            <p>Real-time hand sign detection</p>
          </div>
        </div>
        <div className="status-row">
          <div className="status-chip">
            <span className={`dot ${connected ? 'on' : 'off'}`} />
            {connected ? 'Connected' : 'Disconnected'}
          </div>
          <div className="status-chip">GPU: Auto</div>
          <div className="status-chip">FPS: {showFps ? sendFps : '--'}</div>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="live-grid">
        <div className="video-card">
          <div className="live-badge">
            <span className="dot live" />
            LIVE
          </div>
          <video ref={videoRef} className={mirrorView ? 'mirror' : ''} muted playsInline />
          {showLandmarks && (
            <canvas
              ref={landmarksRef}
              className={`landmark-canvas ${mirrorView ? 'mirror' : ''}`}
            />
          )}
          <div className="video-overlay">
            <div className="hand-status">
              <span className={`dot ${result.hand_detected ? 'on' : 'off'}`} />
              {result.hand_detected ? 'Hand Detected' : 'No Hand'}
            </div>
          </div>
        </div>

        <aside className="controls-card">
          <div className="controls-header">
            <h3>Controls</h3>
            <button className="ghost">...</button>
          </div>

          <div className="control-block">
            <div className="control-title">
              <span>Confidence Threshold</span>
              <span className="pill">{threshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.95"
              step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
            />
          </div>

          <div className="control-block">
            <div className="control-title">
              <span>Smoothing Window</span>
            </div>
            <div className="segmented">
              {[5, 10, 15, 20].map((val) => (
                <button
                  key={val}
                  className={val === smoothingWindow ? 'active' : ''}
                  type="button"
                  onClick={() => setSmoothingWindow(val)}
                >
                  {val}
                </button>
              ))}
            </div>
          </div>

          <div className="control-block">
            <div className="control-title">
              <span>Send FPS</span>
              <span className="pill">{sendFps}</span>
            </div>
            <input
              type="range"
              min="5"
              max="30"
              step="1"
              value={sendFps}
              onChange={(e) => setSendFps(Number(e.target.value))}
            />
          </div>

          <div className="control-block">
            <div className="control-title">Debug Options</div>
            <label className="toggle">
              <span>Show Landmarks</span>
              <input type="checkbox" checked={showLandmarks} onChange={(e) => setShowLandmarks(e.target.checked)} />
              <span className="slider" />
            </label>
            <label className="toggle">
              <span>Show FPS</span>
              <input type="checkbox" checked={showFps} onChange={(e) => setShowFps(e.target.checked)} />
              <span className="slider" />
            </label>
          </div>

          <div className="ws-block">
            <div>WebSocket</div>
            <div className="ws-status">
              <span className={`dot ${connected ? 'on' : 'off'}`} />
              {connected ? 'Connected' : 'Disconnected'}
            </div>
            <div className="ws-url">/ws/predict</div>
          </div>
        </aside>
      </div>

      <div className="prediction-card">
        <div className="prediction-title">Prediction</div>
        <div className="prediction-main">
          <div className="pred-letter">{result.pred}</div>
          <div className="pred-meta">
            <div className="confidence">
              Confidence: {confidencePct}%
            </div>
            <div className="stability">Stable</div>
          </div>
        </div>
        <div className="confidence-bar">
          <div className="confidence-fill" style={{ width: `${confidencePct}%` }} />
        </div>
      </div>

      <div className="action-row">
        <button className="danger">Stop</button>
        <button className="secondary">Pause</button>
        <button className="secondary" onClick={() => setMirrorView((v) => !v)}>
          {mirrorView ? 'Unmirror Camera' : 'Mirror Camera'}
        </button>
      </div>

      <canvas ref={canvasRef} className="hidden" />
    </section>
  )
}

export default LivePredictor
