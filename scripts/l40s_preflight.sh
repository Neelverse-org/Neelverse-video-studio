#!/usr/bin/env bash
set -euo pipefail

failures=0
warn() { printf 'WARN: %s\n' "$1"; }
fail() { printf 'FAIL: %s\n' "$1"; failures=$((failures + 1)); }
pass() { printf ' OK : %s\n' "$1"; }

printf '\nNeelverse L40S deployment preflight\n===================================\n'

if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail 'nvidia-smi is unavailable'
else
  gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
  gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1)
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
  printf 'GPU : %s (%s), driver %s\n' "$gpu_name" "$gpu_memory" "$driver"
  [[ "$gpu_name" == *"L40S"* ]] && pass 'L40S detected' || warn 'GPU is not reported as L40S'
fi

if ! command -v docker >/dev/null 2>&1; then
  fail 'Docker is unavailable'
else
  docker info >/dev/null 2>&1 && pass 'Docker daemon is reachable' || fail 'Docker daemon is not reachable'
  docker compose version >/dev/null 2>&1 && pass 'Docker Compose plugin is available' || fail 'Docker Compose plugin is unavailable'
  if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
    pass 'NVIDIA container runtime is registered'
  else
    fail 'NVIDIA container runtime is not registered'
  fi
fi

ram_gb=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
disk_gb=$(df -BG . | awk 'NR==2 {gsub("G", "", $4); print $4}')
printf 'RAM : %s GB\nDisk: %s GB available\n' "$ram_gb" "$disk_gb"
(( ram_gb >= 30 )) && pass 'Host RAM meets the 32 GB minimum' || fail 'At least 32 GB RAM is required'
(( disk_gb >= 80 )) && pass 'Enough free disk for images and model cache' || fail 'At least 80 GB free disk is required'

if (( failures > 0 )); then
  printf '\nPreflight failed with %d blocking issue(s).\n' "$failures"
  exit 1
fi
printf '\nPreflight passed. The first GPU start will download and compile model artifacts.\n'
