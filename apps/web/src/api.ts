import type {
  Asset,
  GenerationSession,
  Health,
  RTCDescription,
  SessionCreate,
  TransitionMode,
  User,
} from './types'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message)
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  if (init.body && !(init.body instanceof FormData)) headers.set('Content-Type', 'application/json')
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
    credentials: 'include',
  })
  if (!response.ok) {
    let message = `Request failed (${response.status})`
    try {
      const payload = (await response.json()) as { detail?: string }
      if (payload.detail) message = payload.detail
    } catch {
      // Keep the HTTP status fallback when the response has no JSON body.
    }
    throw new ApiError(message, response.status)
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export const api = {
  login: (username: string, password: string) =>
    request<User>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    }),
  logout: () => request<void>('/auth/logout', { method: 'POST' }),
  me: () => request<User>('/auth/me'),
  health: () => request<Health>('/health'),
  uploadAsset: async (file: File) => {
    const body = new FormData()
    body.append('file', file)
    return request<Asset>('/assets', { method: 'POST', body })
  },
  createSession: (payload: SessionCreate) =>
    request<GenerationSession>('/sessions', { method: 'POST', body: JSON.stringify(payload) }),
  sessions: () => request<GenerationSession[]>('/sessions'),
  session: (id: string) => request<GenerationSession>(`/sessions/${id}`),
  pause: (id: string) => request<GenerationSession>(`/sessions/${id}/pause`, { method: 'POST' }),
  resume: (id: string) => request<GenerationSession>(`/sessions/${id}/resume`, { method: 'POST' }),
  stop: (id: string) => request<GenerationSession>(`/sessions/${id}/stop`, { method: 'POST' }),
  updatePrompt: (id: string, prompt: string, transition: TransitionMode) =>
    request<GenerationSession>(`/sessions/${id}/prompt`, {
      method: 'PATCH',
      body: JSON.stringify({ prompt, transition }),
    }),
  rtcOffer: (id: string, offer: RTCDescription) =>
    request<RTCDescription>(`/sessions/${id}/rtc/offer`, {
      method: 'POST',
      body: JSON.stringify(offer),
    }),
  downloadUrl: (id: string) => `${API_BASE}/sessions/${id}/download`,
  frameUrl: (id: string) => `${API_BASE}/sessions/${id}/frame`,
}
