import { useState, type FormEvent } from 'react'
import { ArrowRight, Eye, EyeOff, Film, LoaderCircle, LockKeyhole, Sparkles } from 'lucide-react'
import { api } from './api'
import type { User } from './types'

interface LoginProps {
  onLogin: (user: User) => void
}

export function Login({ onLogin }: LoginProps) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function submit(event: FormEvent) {
    event.preventDefault()
    setLoading(true)
    setError(null)
    try {
      onLogin(await api.login(username.trim(), password))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to sign in')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="login-shell">
      <div className="login-glow login-glow-one" />
      <div className="login-glow login-glow-two" />
      <section className="login-story">
        <div className="brand-lockup">
          <div className="brand-mark"><Film size={22} /></div>
          <span>NEELVERSE</span>
        </div>
        <div className="story-copy">
          <div className="eyebrow"><Sparkles size={14} /> Real-time generative canvas</div>
          <h1>Imagine it.<br /><em>Watch it unfold.</em></h1>
          <p>Direct continuous AI video from a single creative studio—live, fluid and under your control.</p>
        </div>
        <div className="story-metrics">
          <div><strong>480p</strong><span>Native generation</span></div>
          <div><strong>24</strong><span>Display FPS</span></div>
          <div><strong>&lt;1.5s</strong><span>Target latency</span></div>
        </div>
      </section>

      <section className="login-panel">
        <form className="login-card" onSubmit={submit}>
          <div className="mobile-brand brand-lockup">
            <div className="brand-mark"><Film size={20} /></div><span>NEELVERSE</span>
          </div>
          <div className="login-icon"><LockKeyhole size={22} /></div>
          <h2>Welcome back</h2>
          <p className="muted">Sign in to enter your video studio.</p>
          <label>
            <span>Username</span>
            <input
              autoComplete="username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              placeholder="studio user"
              required
            />
          </label>
          <label>
            <span>Password</span>
            <div className="password-field">
              <input
                autoComplete="current-password"
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="••••••••••••"
                minLength={8}
                required
              />
              <button type="button" aria-label="Toggle password visibility" onClick={() => setShowPassword(!showPassword)}>
                {showPassword ? <EyeOff size={17} /> : <Eye size={17} />}
              </button>
            </div>
          </label>
          {error && <div className="form-error">{error}</div>}
          <button className="primary-button login-button" disabled={loading}>
            {loading ? <LoaderCircle className="spin" size={18} /> : <>Enter studio <ArrowRight size={18} /></>}
          </button>
          <div className="secure-note"><span /> Private workspace · Encrypted session</div>
        </form>
      </section>
    </main>
  )
}
