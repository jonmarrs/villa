#!/usr/bin/env bash
set -euo pipefail

if [[ "${AGENTS_AGENT_MODE:-0}" == "1" && "${AGENTS_ALLOW_INSTALL:-0}" != "1" ]]; then
  echo "INFO: smoke_primus_docker.sh is disabled by default in agent mode."
  echo "Set AGENTS_ALLOW_INSTALL=1 to run this script."
  echo "Example: AGENTS_AGENT_MODE=1 AGENTS_ALLOW_INSTALL=1 ./smoke_primus_docker.sh"
  exit 0
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-ink-detection-optimized-inference:gpu-primus-smoke}"
VILLA_REPO="${VILLA_REPO:-https://github.com/ScrollPrize/villa.git}"
VILLA_REF="${VILLA_REF:-main}"
DOCKER_GPU_ARGS="${DOCKER_GPU_ARGS:---gpus all}"
DOCKER_PREFLIGHT_IMAGE="${DOCKER_PREFLIGHT_IMAGE:-hello-world:latest}"
DOCKER_PREFLIGHT_ARGS="${DOCKER_PREFLIGHT_ARGS:---network=none --security-opt apparmor=unconfined}"

print_host_fix_hint() {
  cat <<'EOF'

Host fix candidates:
  1. Use a normal system Docker daemon, then rerun:
       sudo apt-get update
       sudo apt-get install -y docker.io
       sudo usermod -aG docker "$USER"
       newgrp docker

  2. For rootless/user-namespace Docker, install privileged helpers first:
       sudo apt-get update
       sudo apt-get install -y uidmap apparmor-utils slirp4netns fuse-overlayfs

     Ubuntu hosts with AppArmor-restricted unprivileged user namespaces may also
     need a root-owned AppArmor profile for the rootlesskit binary path.

  See PR899_DOCKER_HOST_FIX.md for the full host setup checklist.
EOF
}

check_docker_daemon() {
  if ! command -v docker >/dev/null 2>&1; then
    cat >&2 <<'EOF'
ERROR: Docker CLI is not installed or is not on PATH.

Install Docker or run this smoke from a host with Docker available before
validating the Primus optimized-inference container path.
EOF
    print_host_fix_hint >&2
    exit 1
  fi

  set +e
  docker_info_output="$(docker info 2>&1)"
  docker_info_status=$?
  set -e

  if [[ $docker_info_status -eq 0 ]]; then
    return 0
  fi

  cat >&2 <<'EOF'
ERROR: Docker CLI is available, but it cannot reach a Docker daemon.

The Primus smoke needs Docker daemon access to build and run the GPU image.
Start Docker, set DOCKER_HOST for a non-default daemon, or run on a host with
system Docker available.
EOF
  {
    echo
    echo "Docker host:"
    echo "  ${DOCKER_HOST:-default}"
    echo
    echo "docker info output:"
    printf '%s\n' "$docker_info_output" | sed 's/^/  /'
    print_host_fix_hint | sed 's/^/  /'
  } >&2
  exit 1
}

print_docker_smoke_diagnostics() {
  local failure_output="$1"

  echo
  echo "Docker diagnostics:"
  echo "  docker_cli=$(docker version --format '{{.Client.Version}}' 2>/dev/null || echo unavailable)"
  echo "  docker_server=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo unavailable)"
  echo "  docker_host=${DOCKER_HOST:-default}"
  echo "  storage_driver=$(docker info --format '{{.Driver}}' 2>/dev/null || echo unavailable)"
  echo "  cgroup_driver=$(docker info --format '{{.CgroupDriver}}' 2>/dev/null || echo unavailable)"
  echo "  cgroup_version=$(docker info --format '{{.CgroupVersion}}' 2>/dev/null || echo unavailable)"
  if command -v newuidmap >/dev/null 2>&1 && command -v newgidmap >/dev/null 2>&1; then
    echo "  uidmap=available"
  else
    echo "  uidmap=missing newuidmap/newgidmap"
  fi
  if docker info --format '{{json .SecurityOptions}}' 2>/dev/null | grep -qi rootless; then
    echo "  rootless=true"
  else
    echo "  rootless=false_or_unreported"
  fi

  case "$failure_output" in
    *"error mounting \"devpts\""* | *"gid=5: invalid argument"*)
      echo
      echo "Likely cause:"
      echo "  Docker reached OCI runtime startup, but this host cannot mount devpts"
      echo "  with the gid mapping requested by runc. On rootless/user-namespace"
      echo "  Docker setups, install uidmap/newuidmap/newgidmap or use a system"
      echo "  Docker daemon with a compatible AppArmor/user-namespace policy."
      print_host_fix_hint
      ;;
    *"failed to Lchown"* | *"invalid argument"*"Lchown"*)
      echo
      echo "Likely cause:"
      echo "  Docker pulled image layers but cannot register files owned by IDs"
      echo "  outside the current user namespace. Install uidmap/newuidmap/newgidmap"
      echo "  or use a system Docker daemon before treating this as an image build"
      echo "  failure."
      print_host_fix_hint
      ;;
    *"AppArmor"* | *"apparmor"* | *"permission denied"*)
      echo
      echo "Likely cause:"
      echo "  Docker container execution is blocked by the host security policy."
      echo "  Check AppArmor unprivileged user namespace policy or use system Docker."
      print_host_fix_hint
      ;;
  esac
}

if [[ "${DOCKER_SMOKE_PREFLIGHT:-1}" != "0" ]]; then
  check_docker_daemon
  set +e
  preflight_output="$(docker run --rm $DOCKER_PREFLIGHT_ARGS "$DOCKER_PREFLIGHT_IMAGE" 2>&1)"
  preflight_status=$?
  set -e

  if [[ $preflight_status -ne 0 ]]; then
    cat >&2 <<'EOF'
ERROR: Docker cannot execute a trivial container on this host.

The Primus smoke builds a CUDA image and then runs a checkpoint/load/forward
test inside that image. Fix Docker container execution first, or set
DOCKER_SMOKE_PREFLIGHT=0 if you intentionally want to skip this fast check.
EOF
    {
      echo
      echo "Preflight command:"
      echo "  docker run --rm $DOCKER_PREFLIGHT_ARGS $DOCKER_PREFLIGHT_IMAGE"
      echo
      echo "Docker host:"
      echo "  ${DOCKER_HOST:-default}"
      echo
      echo "Preflight output:"
      printf '%s\n' "$preflight_output" | sed 's/^/  /'
      print_docker_smoke_diagnostics "$preflight_output" | sed 's/^/  /'
    } >&2
    exit 1
  fi
fi

docker build --target gpu \
  --build-arg INSTALL_PRIMUS_DEPS=1 \
  --build-arg VILLA_REPO="$VILLA_REPO" \
  --build-arg VILLA_REF="$VILLA_REF" \
  -t "$IMAGE" \
  "$DIR"

cat <<'PY' | docker run --rm -i $DOCKER_GPU_ARGS \
  -e MODEL=dummy \
  -e MODEL_TYPE=primus \
  -e START_LAYER=0 \
  -e END_LAYER=16 \
  --entrypoint python \
  "$IMAGE" -
from pathlib import Path
import tempfile

import torch

import model_primus
from vesuvius.models.build.build_network_from_config import NetworkFromConfig

model_config = {
    "architecture_type": "primus_s",
    "primus_variant": "S",
    "patch_embed_size": [8, 8, 8],
    "input_shape": [16, 16, 16],
    "in_channels": 1,
    "targets": {"ink": {"out_channels": 1, "activation": "none"}},
    "decoder_head_channels": 4,
    "drop_path_rate": 0.0,
    "patch_drop_rate": 0.0,
    "proj_drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "num_register_tokens": 0,
}

source_model = NetworkFromConfig(model_primus._PrimusConfigShim(model_config))
checkpoint = {
    "model_config": model_config,
    "state_dict": source_model.state_dict(),
    "target_key": "ink",
}

with tempfile.TemporaryDirectory() as tmpdir:
    checkpoint_path = Path(tmpdir) / "primus-smoke.ckpt"
    torch.save(checkpoint, checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = model_primus.load_model(str(checkpoint_path), device, num_frames=16)
    x = torch.randn(1, 1, 16, 16, 16, device=device)
    with torch.inference_mode():
        y = wrapper.forward(x)

assert y.shape == (1, 1, 16, 16, 16), y.shape
assert torch.isfinite(y).all()
print({"device": str(device), "shape": tuple(y.shape)})
PY
