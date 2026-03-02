import { useState } from 'react'
import { predictImage } from '../services/api'

function UploadPredictor() {
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function onSelectFile(event) {
    const file = event.target.files?.[0]
    if (!file) return

    setPreview(URL.createObjectURL(file))
    setError('')
    setLoading(true)

    try {
      const payload = await predictImage(file)
      setResult(payload)
    } catch (err) {
      setError(err.message || 'Failed to predict')
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="upload-page">
      <div className="upload-card">
        <div className="upload-header">
          <div>
            <h3>Image Upload</h3>
            <p>Drop a clear hand photo for instant ASL classification</p>
          </div>
          <span className="pill">JPG • PNG • WEBP</span>
        </div>

        <label className="dropzone">
          <input type="file" accept="image/*" onChange={onSelectFile} />
          <div className="dropzone-content">
            <div className="drop-icon">⇪</div>
            <div>
              <div className="drop-title">Click to upload or drag & drop</div>
              <div className="drop-subtitle">We process on-device frames only</div>
            </div>
            <button type="button" className="secondary">
              Choose file
            </button>
          </div>
        </label>

        {preview && (
          <div className="preview-frame">
            <img className="preview" src={preview} alt="Selected" />
          </div>
        )}
      </div>

      <div className="upload-results">
        <div className="result-card">
          <div className="result-title">Prediction</div>
          <div className="result-main">
            <div className="pred-letter">{result?.pred ?? '--'}</div>
            <div className="pred-meta">
              <div className="confidence">
                Confidence: {result ? `${(result.confidence * 100).toFixed(1)}%` : '--'}
              </div>
              <div className="stability">
                {result ? (result.hand_detected ? 'Hand detected' : 'No hand detected') : 'Awaiting image'}
              </div>
            </div>
          </div>
          <div className="confidence-bar">
            <div
              className="confidence-fill"
              style={{ width: `${result ? Math.round(result.confidence * 100) : 0}%` }}
            />
          </div>
        </div>

        <div className="status-stack">
          {loading && <div className="status-banner">Predicting...</div>}
          {error && <div className="error-banner">{error}</div>}
        </div>
      </div>
    </section>
  )
}

export default UploadPredictor
