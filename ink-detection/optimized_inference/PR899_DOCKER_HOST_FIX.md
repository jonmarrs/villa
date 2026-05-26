# PR #899 Docker Host Fix Checklist

The Primus optimized-inference smoke requires Docker to build the GPU image and
run a checkpoint/load/forward pass inside that image. The Python loader tests
can pass while the Docker smoke is still blocked by host Docker setup.

## Required Working State

From `ink-detection/optimized_inference`:

```bash
docker info
docker run --rm --network=none --security-opt apparmor=unconfined hello-world:latest
VILLA_REPO=https://github.com/jonmarrs/villa.git \
VILLA_REF=primus-loader-optimized-inference \
./smoke_primus_docker.sh
```

The first command must reach a daemon. The second command must run a trivial
container. Only then is the Primus smoke a meaningful test of the PR's Docker
image and loader path.

## Option A: System Docker

Use this when a normal privileged Docker install is acceptable:

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker "$USER"
newgrp docker
docker info
docker run --rm hello-world:latest
```

Then rerun:

```bash
cd ink-detection/optimized_inference
VILLA_REPO=https://github.com/jonmarrs/villa.git \
VILLA_REF=primus-loader-optimized-inference \
./smoke_primus_docker.sh
```

## Option B: Rootless/User-Namespace Docker

Use this when system Docker is unavailable but privileged setup is possible:

```bash
sudo apt-get update
sudo apt-get install -y uidmap apparmor-utils slirp4netns fuse-overlayfs
```

The `uidmap` package provides setuid-root `newuidmap` and `newgidmap`; those
cannot be emulated from an unprivileged shell. Without them, rootless or
user-namespace Docker commonly fails during image layer registration with
`failed to Lchown ... invalid argument`.

On Ubuntu hosts with AppArmor-restricted unprivileged user namespaces, add an
AppArmor profile for the exact `rootlesskit` binary path. Example for a static
rootlesskit installed under `~/.local`:

```bash
ROOTLESSKIT_PATH="$HOME/.local/docker-static/docker-rootless-extras/rootlesskit"
PROFILE_NAME="$(printf '%s' "$ROOTLESSKIT_PATH" | sed 's#/#.#g; s#^\\.##')"
PROFILE_PATH="/etc/apparmor.d/${PROFILE_NAME}"

cat <<EOF | sudo tee "$PROFILE_PATH" >/dev/null
abi <abi/4.0>,
include <tunables/global>

$ROOTLESSKIT_PATH flags=(unconfined) {
  userns,

  include if exists <local/$PROFILE_NAME>
}
EOF

sudo systemctl restart apparmor.service
```

Then start the rootless/user-owned daemon and rerun the smoke with its socket:

```bash
export DOCKER_HOST=unix://$HOME/.local/run/openclaw-docker/docker.sock
docker info
docker run --rm --network=none --security-opt apparmor=unconfined hello-world:latest

cd ink-detection/optimized_inference
VILLA_REPO=https://github.com/jonmarrs/villa.git \
VILLA_REF=primus-loader-optimized-inference \
./smoke_primus_docker.sh
```

## Failure Interpretation

- `Cannot connect to the Docker daemon`: Docker CLI is installed, but `DOCKER_HOST`
  is wrong or no daemon is running.
- `error mounting "devpts" ... gid=5: invalid argument`: Docker reached OCI
  runtime startup, but the host user namespace cannot perform the requested
  mount/mapping. Install `uidmap` helpers or use system Docker.
- `failed to Lchown ... invalid argument`: Docker pulled layers, but cannot map
  file ownership for root-owned image contents. Install `newuidmap`/`newgidmap`
  or use system Docker.
- AppArmor or `permission denied` errors before the container starts: check the
  Ubuntu unprivileged user namespace policy or use system Docker.
