function StatusPill({ connected }) {
  return <span className={`status-pill ${connected ? 'on' : 'off'}`}>{connected ? 'Connected' : 'Disconnected'}</span>
}

export default StatusPill
