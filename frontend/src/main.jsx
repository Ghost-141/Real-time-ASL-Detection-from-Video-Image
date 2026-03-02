import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { installGlobalErrorLogger } from './services/errorLogger'
import './styles.css'

installGlobalErrorLogger()

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
