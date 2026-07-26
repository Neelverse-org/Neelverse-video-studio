import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api'

interface WebRTCState {
  stream: MediaStream | null
  localPreview: MediaStream | null
  status: 'idle' | 'connecting' | 'connected' | 'failed'
  error: string | null
  connect: (sessionId: string, camera: boolean) => Promise<void>
  disconnect: () => void
}

function waitForIceGathering(peer: RTCPeerConnection): Promise<void> {
  if (peer.iceGatheringState === 'complete') return Promise.resolve()
  return new Promise((resolve) => {
    const listener = () => {
      if (peer.iceGatheringState === 'complete') {
        peer.removeEventListener('icegatheringstatechange', listener)
        resolve()
      }
    }
    peer.addEventListener('icegatheringstatechange', listener)
  })
}

function configuredIceServers(): RTCIceServer[] {
  const raw = import.meta.env.VITE_ICE_SERVERS_JSON
  if (!raw) return [{ urls: 'stun:stun.l.google.com:19302' }]
  try {
    const parsed = JSON.parse(raw) as RTCIceServer[]
    return parsed.length ? parsed : [{ urls: 'stun:stun.l.google.com:19302' }]
  } catch {
    return [{ urls: 'stun:stun.l.google.com:19302' }]
  }
}

export function useWebRTC(): WebRTCState {
  const peerRef = useRef<RTCPeerConnection | null>(null)
  const localRef = useRef<MediaStream | null>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [localPreview, setLocalPreview] = useState<MediaStream | null>(null)
  const [status, setStatus] = useState<WebRTCState['status']>('idle')
  const [error, setError] = useState<string | null>(null)

  const disconnect = useCallback(() => {
    peerRef.current?.close()
    peerRef.current = null
    localRef.current?.getTracks().forEach((track) => track.stop())
    localRef.current = null
    setLocalPreview(null)
    setStream(null)
    setStatus('idle')
  }, [])

  const connect = useCallback(
    async (sessionId: string, camera: boolean) => {
      disconnect()
      setStatus('connecting')
      setError(null)
      try {
        const peer = new RTCPeerConnection({ iceServers: configuredIceServers() })
        peerRef.current = peer
        const remoteStream = new MediaStream()
        peer.ontrack = (event) => {
          remoteStream.addTrack(event.track)
          setStream(new MediaStream(remoteStream.getTracks()))
        }
        peer.onconnectionstatechange = () => {
          if (peer.connectionState === 'connected') setStatus('connected')
          if (peer.connectionState === 'failed') {
            setStatus('failed')
            setError('WebRTC connection failed. Check TURN/firewall configuration.')
          }
        }

        if (camera) {
          const local = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: { width: { ideal: 832 }, height: { ideal: 480 }, frameRate: { ideal: 10, max: 12 } },
          })
          localRef.current = local
          setLocalPreview(local)
          local.getVideoTracks().forEach((track) => peer.addTrack(track, local))
        } else {
          peer.addTransceiver('video', { direction: 'recvonly' })
        }

        const offer = await peer.createOffer()
        await peer.setLocalDescription(offer)
        await waitForIceGathering(peer)
        if (!peer.localDescription) throw new Error('Could not create a WebRTC offer')
        const answer = await api.rtcOffer(sessionId, {
          sdp: peer.localDescription.sdp,
          type: peer.localDescription.type,
        })
        await peer.setRemoteDescription(answer)
      } catch (cause) {
        setStatus('failed')
        setError(cause instanceof Error ? cause.message : 'Unable to start the video stream')
        disconnect()
        throw cause
      }
    },
    [disconnect],
  )

  useEffect(() => disconnect, [disconnect])

  return { stream, localPreview, status, error, connect, disconnect }
}
