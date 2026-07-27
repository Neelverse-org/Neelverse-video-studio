import {
  Activity,
  Camera,
  ChevronDown,
  CircleStop,
  Clapperboard,
  Clock3,
  Download,
  Film,
  Gauge,
  History,
  Image as ImageIcon,
  Layers3,
  LoaderCircle,
  LogOut,
  Maximize2,
  MonitorPlay,
  MoreHorizontal,
  Pause,
  Play,
  Radio,
  RefreshCw,
  Send,
  Settings2,
  Sparkles,
  UploadCloud,
  Video,
  WandSparkles,
  X,
  Zap,
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api'
import type {
  GenerationMode,
  GenerationSession,
  Health,
  QualityProfile,
  Resolution,
  SessionCreate,
  TransitionMode,
  User,
} from './types'
import { useWebRTC } from './useWebRTC'

interface StudioProps {
  user: User
  onLogout: () => void
}

const modes: Array<{ id: GenerationMode; label: string; hint: string; icon: typeof WandSparkles }> = [
  { id: 'text', label: 'Text', hint: 'Prompt to video', icon: WandSparkles },
  { id: 'image', label: 'Image', hint: 'Animate a still', icon: ImageIcon },
  { id: 'video', label: 'Video', hint: 'Transform footage', icon: Video },
  { id: 'camera', label: 'Camera', hint: 'Live transformation', icon: Camera },
]

const terminalStatuses = new Set(['completed', 'stopped', 'failed'])

function formatTime(date: string) {
  return new Intl.DateTimeFormat('en', { hour: '2-digit', minute: '2-digit' }).format(new Date(date))
}

function VideoStage({
  stream,
  localPreview,
  session,
  connectionStatus,
}: {
  stream: MediaStream | null
  localPreview: MediaStream | null
  session: GenerationSession | null
  connectionStatus: string
}) {
  const remoteRef = useRef<HTMLVideoElement>(null)
  const localRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (remoteRef.current) remoteRef.current.srcObject = stream
  }, [stream])
  useEffect(() => {
    if (localRef.current) localRef.current.srcObject = localPreview
  }, [localPreview])

  // Poll frames when WebRTC is not connected but session is running
  useEffect(() => {
    if (stream || !session?.id || session.status !== 'running') return
    let active = true
    const poll = async () => {
      while (active) {
        try {
          const response = await fetch(`/api/sessions/${session.id}/frame`, { credentials: 'include' })
          if (response.ok && canvasRef.current) {
            const blob = await response.blob()
            const bitmap = await createImageBitmap(blob)
            const ctx = canvasRef.current.getContext('2d')
            if (ctx) {
              canvasRef.current.width = bitmap.width
              canvasRef.current.height = bitmap.height
              ctx.drawImage(bitmap, 0, 0)
            }
            bitmap.close()
          }
        } catch { /* polling error, retry */ }
        await new Promise(resolve => setTimeout(resolve, 120))
      }
    }
    void poll()
    return () => { active = false }
  }, [stream, session?.id, session?.status])

  return (
    <div className="video-stage">
      {stream ? (
        <video ref={remoteRef} autoPlay playsInline muted className="output-video" />
      ) : session?.status === 'running' ? (
        <canvas ref={canvasRef} className="output-video" />
      ) : session?.status === 'loading' || session?.status === 'warming' ? (
        <div className="stage-idle">
          <div className="aurora-orb"><Sparkles size={30} /></div>
          <h2>{session.status === 'loading' ? 'Loading model...' : 'Warming up GPU...'}</h2>
          <p>First generation may take a few minutes while models download</p>
        </div>
      ) : (
        <div className="stage-idle">
          <div className="aurora-orb"><Sparkles size={30} /></div>
          <h2>{session?.status === 'queued' ? 'Your scene is queued' : 'A blank canvas, for now.'}</h2>
          <p>{session?.status === 'queued' ? `Position ${session.queue_position ?? '—'} · The GPU will begin shortly` : 'Choose a mode, describe your scene, then bring it to life.'}</p>
        </div>
      )}
      <div className="stage-grain" />
      <div className="stage-topbar">
        <div className={`stream-pill ${session?.status === 'running' ? 'live' : ''}`}>
          <span /> {session?.status === 'running' ? 'LIVE' : session?.status?.toUpperCase() ?? 'READY'}
        </div>
        <div className="stage-actions">
          <span>{connectionStatus === 'connected' ? 'WebRTC' : 'Preview'}</span>
          <button aria-label="Fullscreen" onClick={() => remoteRef.current?.requestFullscreen()}><Maximize2 size={16} /></button>
          <button aria-label="More options"><MoreHorizontal size={17} /></button>
        </div>
      </div>
      {localPreview && (
        <div className="camera-preview">
          <video ref={localRef} autoPlay playsInline muted />
          <span><Camera size={12} /> Input</span>
        </div>
      )}
      {session && (
        <div className="stage-bottombar">
          <span><MonitorPlay size={14} /> {session.resolution}</span>
          <span><Radio size={13} /> {session.display_fps || 24} fps output</span>
          <span><Clock3 size={13} /> {session.duration_seconds ? `${session.duration_seconds}s` : 'Continuous'}</span>
        </div>
      )}
    </div>
  )
}

export function Studio({ user, onLogout }: StudioProps) {
  const [mode, setMode] = useState<GenerationMode>('text')
  const [prompt, setPrompt] = useState('A cinematic midnight city floating above luminous violet clouds, slow camera drift')
  const [negativePrompt, setNegativePrompt] = useState('flicker, blur, distortion, text, watermark')
  const [transition, setTransition] = useState<TransitionMode>('smooth')
  const [resolution, setResolution] = useState<Resolution>('480p')
  const [quality, setQuality] = useState<QualityProfile>('turbo')
  const [motion, setMotion] = useState(0.55)
  const [duration, setDuration] = useState<SessionCreate['duration_seconds']>(null)
  const [record, setRecord] = useState(true)
  const [asset, setAsset] = useState<File | null>(null)
  const [session, setSession] = useState<GenerationSession | null>(null)
  const [history, setHistory] = useState<GenerationSession[]>([])
  const [health, setHealth] = useState<Health | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [panel, setPanel] = useState<'create' | 'history'>('create')
  const rtc = useWebRTC()

  const refreshHistory = useCallback(async () => {
    try {
      setHistory(await api.sessions())
    } catch {
      // The main action surfaces API errors; history refresh is best effort.
    }
  }, [])

  useEffect(() => {
    void Promise.all([api.health(), api.sessions()]).then(([nextHealth, nextHistory]) => {
      setHealth(nextHealth)
      setHistory(nextHistory)
    })
  }, [])

  const sessionId = session?.id
  const sessionStatus = session?.status
  useEffect(() => {
    if (!sessionId || !sessionStatus || terminalStatuses.has(sessionStatus)) return
    const timer = window.setInterval(async () => {
      try {
        const updated = await api.session(sessionId)
        setSession(updated)
        if (terminalStatuses.has(updated.status)) void refreshHistory()
      } catch {
        // A transient polling error should not terminate the active media stream.
      }
    }, 1000)
    return () => window.clearInterval(timer)
  }, [sessionId, sessionStatus, refreshHistory])

  const isActive = session && !terminalStatuses.has(session.status)
  const sourceRequired = mode === 'image' || mode === 'video'
  const acceptedSource = mode === 'video' ? 'video/mp4,video/webm,video/quicktime,.mkv' : 'image/png,image/jpeg,image/webp'

  const headlineStats = useMemo(
    () => [
      { label: 'Native', value: `${session?.native_fps?.toFixed(1) ?? '0.0'} fps`, icon: Activity },
      { label: 'Latency', value: `${Math.round(session?.latency_ms ?? 0)} ms`, icon: Zap },
      { label: 'VRAM', value: `${session?.vram_used_gb?.toFixed(1) ?? '0.0'} GB`, icon: Gauge },
      { label: 'Frames', value: `${session?.frames_generated ?? 0}`, icon: Layers3 },
    ],
    [session],
  )

  async function startGeneration() {
    if (prompt.trim().length < 3) return setError('Describe the scene before starting.')
    if (sourceRequired && !asset) return setError(`Choose a source ${mode} first.`)
    setBusy(true)
    setError(null)
    try {
      let assetId: string | undefined
      if (asset) assetId = (await api.uploadAsset(asset)).id
      const created = await api.createSession({
        mode,
        prompt: prompt.trim(),
        negative_prompt: negativePrompt.trim(),
        transition,
        resolution,
        quality,
        motion_strength: motion,
        duration_seconds: duration,
        asset_id: assetId,
        record,
      })
      setSession(created)
      try {
        await rtc.connect(created.id, mode === 'camera')
      } catch {
        // WebRTC failed but generation continues — telemetry still updates via polling
        console.warn('WebRTC stream unavailable — generation running without live preview')
      }
      void refreshHistory()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not start generation')
    } finally {
      setBusy(false)
    }
  }

  async function control(action: 'pause' | 'resume' | 'stop') {
    if (!session) return
    setError(null)
    try {
      const updated = await api[action](session.id)
      setSession(updated)
      if (action === 'stop') {
        rtc.disconnect()
        void refreshHistory()
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : `Could not ${action} the session`)
    }
  }

  async function applyPrompt() {
    if (!session || prompt.trim().length < 3) return
    try {
      setSession(await api.updatePrompt(session.id, prompt.trim(), transition))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not update the live prompt')
    }
  }

  return (
    <main className="studio-shell">
      <aside className="studio-sidebar">
        <div className="sidebar-brand"><Film size={20} /><span>N</span></div>
        <nav>
          <button className={panel === 'create' ? 'active' : ''} onClick={() => setPanel('create')} title="Create"><Clapperboard size={19} /></button>
          <button className={panel === 'history' ? 'active' : ''} onClick={() => setPanel('history')} title="History"><History size={19} /></button>
          <button title="Settings"><Settings2 size={19} /></button>
        </nav>
        <button className="sidebar-avatar" title={user.username}>{user.username.slice(0, 1).toUpperCase()}</button>
      </aside>

      <section className="studio-workspace">
        <header className="studio-header">
          <div className="wordmark"><strong>NEELVERSE</strong><span>VIDEO STUDIOS</span></div>
          <div className="header-status">
            <div className="gpu-status"><span className={health?.gpu_available ? 'online' : ''} /> {health?.backend ?? 'Connecting'}</div>
            <div className="queue-status">Queue <strong>{health?.queued_sessions ?? 0}</strong></div>
            <button className="user-menu" onClick={onLogout}><span>{user.username}</span><LogOut size={15} /></button>
          </div>
        </header>

        <div className="studio-content">
          <section className="canvas-column">
            <VideoStage stream={rtc.stream} localPreview={rtc.localPreview} session={session} connectionStatus={rtc.status} />
            <div className="telemetry-row">
              {headlineStats.map(({ label, value, icon: Icon }) => (
                <div className="telemetry-card" key={label}><Icon size={15} /><div><span>{label}</span><strong>{value}</strong></div></div>
              ))}
              <div className="session-controls">
                {session?.status === 'running' && <button onClick={() => void control('pause')}><Pause size={16} /> Pause</button>}
                {session?.status === 'paused' && <button onClick={() => void control('resume')}><Play size={16} /> Resume</button>}
                {isActive && <button className="stop-button" onClick={() => void control('stop')}><CircleStop size={16} /> Stop</button>}
                {session?.output_available && <a href={api.downloadUrl(session.id)}><Download size={16} /> Download</a>}
              </div>
            </div>
          </section>

          <aside className="control-panel">
            <div className="panel-tabs">
              <button className={panel === 'create' ? 'active' : ''} onClick={() => setPanel('create')}>Create</button>
              <button className={panel === 'history' ? 'active' : ''} onClick={() => setPanel('history')}>Sessions</button>
            </div>

            {panel === 'history' ? (
              <div className="history-panel">
                <div className="section-heading"><div><span>RECENT</span><h3>Your generations</h3></div><button onClick={() => void refreshHistory()}><RefreshCw size={15} /></button></div>
                {history.length === 0 ? <div className="empty-history"><History size={28} /><p>No generations yet.</p></div> : history.map((item) => (
                  <button className="history-item" key={item.id} onClick={() => setSession(item)}>
                    <div className="history-thumb"><Film size={17} /></div>
                    <div><strong>{item.prompt}</strong><span>{item.mode} · {formatTime(item.created_at)}</span></div>
                    <span className={`history-state ${item.status}`}>{item.status}</span>
                  </button>
                ))}
              </div>
            ) : (
              <div className="create-panel">
                <section className="control-section">
                  <div className="section-heading"><div><span>01 · INPUT</span><h3>Choose your canvas</h3></div></div>
                  <div className="mode-grid">
                    {modes.map(({ id, label, hint, icon: Icon }) => (
                      <button key={id} className={mode === id ? 'active' : ''} onClick={() => { setMode(id); setAsset(null) }} disabled={Boolean(isActive)}>
                        <Icon size={18} /><strong>{label}</strong><span>{hint}</span>
                      </button>
                    ))}
                  </div>
                </section>

                {sourceRequired && (
                  <section className="control-section source-section">
                    <label className={`upload-zone ${asset ? 'has-file' : ''}`}>
                      <input type="file" accept={acceptedSource} onChange={(event) => setAsset(event.target.files?.[0] ?? null)} />
                      {asset ? <><Film size={20} /><div><strong>{asset.name}</strong><span>{(asset.size / 1024 / 1024).toFixed(1)} MB</span></div><button type="button" onClick={(event) => { event.preventDefault(); setAsset(null) }}><X size={15} /></button></> : <><UploadCloud size={22} /><div><strong>Drop or choose {mode}</strong><span>Up to 250 MB</span></div></>}
                    </label>
                  </section>
                )}

                <section className="control-section">
                  <div className="section-heading"><div><span>02 · DIRECTION</span><h3>Describe the scene</h3></div><span className="char-count">{prompt.length}/4000</span></div>
                  <div className="prompt-box">
                    <textarea value={prompt} maxLength={4000} onChange={(event) => setPrompt(event.target.value)} placeholder="A cinematic scene of..." />
                    {isActive && <button className="send-prompt" onClick={() => void applyPrompt()} title="Apply live prompt"><Send size={15} /></button>}
                  </div>
                  <details className="negative-prompt">
                    <summary>Negative prompt <ChevronDown size={14} /></summary>
                    <textarea value={negativePrompt} onChange={(event) => setNegativePrompt(event.target.value)} />
                  </details>
                  <div className="segmented transition-toggle">
                    <button className={transition === 'smooth' ? 'active' : ''} onClick={() => setTransition('smooth')}>Smooth transition</button>
                    <button className={transition === 'restart' ? 'active' : ''} onClick={() => setTransition('restart')}>Restart scene</button>
                  </div>
                </section>

                <section className="control-section compact-section">
                  <div className="section-heading"><div><span>03 · OUTPUT</span><h3>Generation profile</h3></div></div>
                  <div className="field-pair">
                    <label><span>Resolution</span><select value={resolution} onChange={(event) => setResolution(event.target.value as Resolution)}><option value="480p">480p · Native</option><option value="720p">720p · Upscaled</option></select></label>
                    <label><span>Quality</span><select value={quality} onChange={(event) => setQuality(event.target.value as QualityProfile)}><option value="turbo">Turbo · 4 step</option><option value="balanced">Balanced · 5 step</option><option value="quality">Quality · 6 step</option></select></label>
                  </div>
                  <label className="range-field"><div><span>Motion strength</span><strong>{Math.round(motion * 100)}%</strong></div><input type="range" min="0" max="1" step="0.01" value={motion} onChange={(event) => setMotion(Number(event.target.value))} /></label>
                  <div className="duration-field"><span>Duration</span><div className="duration-chips">{([null, 5, 10, 30, 60, 120, 180] as const).map((value) => <button key={value ?? 'live'} className={duration === value ? 'active' : ''} onClick={() => setDuration(value)}>{value ? `${value}s` : '∞ Live'}</button>)}</div></div>
                  <label className="record-toggle"><input type="checkbox" checked={record} onChange={(event) => setRecord(event.target.checked)} /><span /><div><strong>Record output</strong><small>Save an MP4 for download</small></div></label>
                </section>

                {(error || rtc.error) && <div className="panel-error"><X size={15} /> {error || rtc.error}</div>}
                <button className="generate-button" disabled={busy || Boolean(isActive)} onClick={() => void startGeneration()}>
                  {busy ? <LoaderCircle className="spin" size={18} /> : <><Sparkles size={18} /> Generate live video</>}
                </button>
                <p className="generation-note">One active GPU session · Additional requests enter the queue</p>
              </div>
            )}
          </aside>
        </div>
      </section>
    </main>
  )
}
