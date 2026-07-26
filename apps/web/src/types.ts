export type GenerationMode = 'text' | 'image' | 'video' | 'camera'
export type SessionStatus =
  | 'queued'
  | 'loading'
  | 'warming'
  | 'running'
  | 'paused'
  | 'completed'
  | 'stopped'
  | 'failed'
export type TransitionMode = 'smooth' | 'restart'
export type Resolution = '480p' | '720p'
export type QualityProfile = 'turbo' | 'balanced' | 'quality'

export interface User {
  id: string
  username: string
  is_admin: boolean
}

export interface Asset {
  id: string
  filename: string
  content_type: string
  size_bytes: number
}

export interface SessionCreate {
  mode: GenerationMode
  prompt: string
  negative_prompt: string
  transition: TransitionMode
  resolution: Resolution
  quality: QualityProfile
  motion_strength: number
  duration_seconds: 5 | 10 | 30 | 60 | 120 | 180 | null
  asset_id?: string
  record: boolean
}

export interface GenerationSession extends Omit<SessionCreate, 'asset_id' | 'record'> {
  id: string
  status: SessionStatus
  queue_position: number | null
  native_fps: number
  display_fps: number
  latency_ms: number
  vram_used_gb: number
  frames_generated: number
  error: string | null
  output_available: boolean
  created_at: string
  started_at: string | null
  ended_at: string | null
}

export interface Health {
  status: string
  backend: string
  gpu_available: boolean
  active_session_id: string | null
  queued_sessions: number
}

export interface RTCDescription {
  sdp: string
  type: RTCSdpType
}
