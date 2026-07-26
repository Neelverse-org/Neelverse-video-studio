# Neelverse Video Studios

Real-time generative video web application powered by a single NVIDIA L40S GPU. Generate continuous AI video from text prompts, images, existing footage, or live camera — all from a modern browser-based studio interface.

## Features

- **Text → Live Video** — Describe a scene, watch it generate in real-time
- **Image → Video** — Animate any still image with AI
- **Video → Transformed Video** — Restyle existing footage live
- **Camera → Live Transform** — Real-time AI transformation of webcam/camera input
- **Live Prompt Editing** — Change direction mid-generation with smooth transitions or instant restarts
- **480p Native + 720p Upscale** — Configurable resolution profiles
- **Continuous or Timed** — Generate indefinitely or in 5/10/30/60/120/180-second clips
- **WebRTC Streaming** — Low-latency video delivery to the browser
- **Session Queue** — Multiple authenticated users with fair GPU scheduling
- **MP4 Recording** — Download any generation as a video file
- **Real-time Telemetry** — Native FPS, latency, VRAM usage displayed live

## Architecture

```
Browser (React Studio UI)
    │ WebRTC + REST API
    ▼
Nginx reverse proxy
    │
    ▼
FastAPI Backend
├── JWT Authentication (Argon2)
├── Session Queue (single GPU, multi-user)
├── WebRTC Signaling (aiortc)
├── MP4 Recording (PyAV)
└── Inference Adapters
    ├── Mock (procedural, no GPU)
    ├── Krea Realtime 14B (Wan 2.1 Self-Forcing distilled)
    └── Self-Forcing 1.3B (lightweight T2V streaming)
```

## Quick Start

### Prerequisites

- Docker + Docker Compose
- NVIDIA GPU with Container Toolkit (for GPU mode)
- 48 GB VRAM (L40S recommended) for real-time inference

### 1. Configure

```bash
cp .env.example .env
# Edit .env — set SECRET_KEY, ADMIN_PASSWORD, HF_TOKEN
```

### 2. Run (Mock mode — no GPU needed)

```bash
docker compose up -d --build
# Open http://localhost:8080
# Login: admin / <your password from .env>
```

### 3. Run (L40S GPU mode)

```bash
# Verify hardware first
chmod +x scripts/l40s_preflight.sh
./scripts/l40s_preflight.sh

# Start with GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

### 4. Benchmark

```bash
pip install httpx
python scripts/benchmark_l40s.py --base-url http://localhost:8080 --password <admin-password>
```

## Development (without Docker)

### Backend

```bash
cd services/api
uv sync --extra dev
uv run uvicorn neelverse.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
npm install
npm run dev:web
# Opens at http://localhost:5173 with API proxy to :8000
```

## Project Structure

```
├── apps/web/              # React + TypeScript + Vite studio UI
├── services/api/          # FastAPI Python backend
│   └── neelverse/
│       ├── adapters/      # Inference backends (mock/krea/self-forcing)
│       ├── main.py        # API routes
│       ├── generation.py  # Queue, worker, frame hub
│       ├── rtc.py         # WebRTC negotiation
│       ├── auth.py        # JWT authentication
│       ├── db.py          # SQLite persistence
│       └── config.py      # Environment settings
├── scripts/               # Deployment and benchmarking tools
├── docker-compose.yml     # Base (mock) deployment
├── docker-compose.gpu.yml # L40S GPU overlay
└── .env.example           # Configuration template
```

## Performance Targets (L40S)

| Metric | Target |
|--------|--------|
| Time-to-first-frame | ≤ 1.5s |
| Native generated FPS | 7–10 fps |
| Display FPS (interpolated) | 20–24 fps |
| VRAM usage | < 44 GB |
| Prompt change response | ≤ 1.5s |

## Tech Stack

- **Frontend**: React 19, TypeScript, Vite, Lucide Icons
- **Backend**: FastAPI, aiortc, PyAV, Pydantic, SQLite
- **Inference**: Krea Realtime 14B, Self-Forcing (Wan 2.1), SageAttention2, FP8
- **Deployment**: Docker Compose, NVIDIA Container Toolkit, Nginx, coturn (TURN)

## License

MIT
