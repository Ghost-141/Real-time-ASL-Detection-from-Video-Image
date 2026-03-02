import { useState } from 'react'
import UploadPredictor from './components/UploadPredictor'
import LivePredictor from './components/LivePredictor'

function App() {
  const [mode, setMode] = useState('upload')

  return (
    <main className="app-shell">
      <header className="app-header">
        <h1>ASL Detection</h1>
        <p>Image inference and live webcam prediction</p>
      </header>

      <div className="mode-switch">
        <button className={mode === 'upload' ? 'active' : ''} onClick={() => setMode('upload')}>
          Image Upload
        </button>
        <button className={mode === 'live' ? 'active' : ''} onClick={() => setMode('live')}>
          Live Webcam
        </button>
      </div>

      {mode === 'upload' ? <UploadPredictor /> : <LivePredictor />}
    </main>
  )
}

export default App
