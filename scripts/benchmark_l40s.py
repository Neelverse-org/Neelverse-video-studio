#!/usr/bin/env python3
import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass

import httpx


@dataclass
class RunResult:
    run: int
    session_id: str
    ttff_seconds: float
    total_seconds: float
    native_fps: float
    latency_ms: float
    vram_used_gb: float
    frames_generated: int
    status: str


def run_once(client: httpx.Client, base_url: str, run_number: int, duration: int) -> RunResult:
    started = time.perf_counter()
    response = client.post(
        f"{base_url}/api/sessions",
        json={
            "mode": "text",
            "prompt": "A cinematic drone flight through a luminous futuristic city at midnight, fluid motion",
            "negative_prompt": "flicker, blur, distortion, text, watermark",
            "transition": "smooth",
            "resolution": "480p",
            "quality": "turbo",
            "motion_strength": 0.55,
            "duration_seconds": duration,
            "record": False,
        },
    )
    response.raise_for_status()
    session_id = response.json()["id"]
    first_frame_at = None
    latest = response.json()
    deadline = time.monotonic() + max(300, duration * 10)
    while time.monotonic() < deadline:
        latest = client.get(f"{base_url}/api/sessions/{session_id}").json()
        if first_frame_at is None and latest["frames_generated"] > 0:
            first_frame_at = time.perf_counter()
        if latest["status"] in {"completed", "stopped", "failed"}:
            break
        time.sleep(0.1)
    else:
        client.post(f"{base_url}/api/sessions/{session_id}/stop")
        raise TimeoutError(f"Session {session_id} did not finish before the benchmark deadline")

    finished = time.perf_counter()
    return RunResult(
        run=run_number,
        session_id=session_id,
        ttff_seconds=round((first_frame_at or finished) - started, 3),
        total_seconds=round(finished - started, 3),
        native_fps=float(latest["native_fps"]),
        latency_ms=float(latest["latency_ms"]),
        vram_used_gb=float(latest["vram_used_gb"]),
        frames_generated=int(latest["frames_generated"]),
        status=latest["status"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Neelverse on an NVIDIA L40S")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--duration", type=int, choices=[5, 10, 30, 60, 120, 180], default=10)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    with httpx.Client(timeout=60) as client:
        login = client.post(
            f"{base_url}/api/auth/login",
            json={"username": args.username, "password": args.password},
        )
        login.raise_for_status()
        health = client.get(f"{base_url}/api/health").json()
        results = [run_once(client, base_url, index + 1, args.duration) for index in range(args.runs)]

    summary = {
        "backend": health["backend"],
        "gpu_available": health["gpu_available"],
        "runs": [asdict(result) for result in results],
        "median_ttff_seconds": round(statistics.median(result.ttff_seconds for result in results), 3),
        "median_native_fps": round(statistics.median(result.native_fps for result in results), 2),
        "max_vram_used_gb": round(max(result.vram_used_gb for result in results), 2),
    }
    summary["acceptance"] = {
        "ttff_under_2s": summary["median_ttff_seconds"] <= 2.0,
        "native_fps_at_least_6": summary["median_native_fps"] >= 6.0,
        "vram_under_44gb": summary["max_vram_used_gb"] < 44.0,
        "all_runs_completed": all(result.status == "completed" for result in results),
    }
    print(json.dumps(summary, indent=2))
    return 0 if all(summary["acceptance"].values()) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (httpx.HTTPError, TimeoutError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
