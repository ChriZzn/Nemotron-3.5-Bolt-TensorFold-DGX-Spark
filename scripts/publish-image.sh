#!/usr/bin/env bash
# Push the image scripts/prepare.sh built to GitHub Container Registry as $GHCR_IMAGE:<TensorFold pin>-<patches hash>
# and :latest, labelled with this repository (so the package takes the repository's visibility and access).
# Needs a GitHub token with write:packages: `gh auth login -s write:packages`, or GHCR_TOKEN / GITHUB_TOKEN set.
# Usage: scripts/publish-image.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh

REPO_URL="${REPO_URL:-https://github.com/MiaAI-Lab/Nemotron-3.5-Bolt-TensorFold-DGX-Spark}"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE missing, run scripts/prepare.sh first"
hash=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE")
[[ "$hash" == "$(image_hash)" ]] || die "$IMAGE was built from other patches; run scripts/prepare.sh first"
[[ "$(docker image inspect -f '{{index .Config.Labels "tf.gate"}}' "$IMAGE")" == bolt ]] ||
  die "$IMAGE has no Bolt gate label: it is not published"
tag="${TF_TAG}-${hash}"
alias=latest

token="${GHCR_TOKEN:-${GITHUB_TOKEN:-$(gh auth token 2>/dev/null || true)}}"
[[ -n "$token" ]] || die "no GitHub token: run gh auth login -s write:packages, or set GHCR_TOKEN"
user="${GHCR_USER:-$(gh api user --jq .login 2>/dev/null || echo "${GHCR_IMAGE#ghcr.io/}" | cut -d/ -f1)}"

log "Labelling $IMAGE as $GHCR_IMAGE:$tag"
docker build -q -t "$GHCR_IMAGE:$tag" -t "$GHCR_IMAGE:$alias" \
  --label org.opencontainers.image.source="$REPO_URL" \
  --label org.opencontainers.image.description="Nemotron 3.5 Bolt 74B-A9B 24GB on one DGX Spark: TensorFold ${TF_REF:0:12} with patches $hash" \
  - <<<"FROM $IMAGE" >/dev/null
log "Pushing $GHCR_IMAGE:$tag and :$alias"
login_dir=$(mktemp -d)
trap 'rm -rf -- "$login_dir"' EXIT
echo "$token" | DOCKER_CONFIG="$login_dir" docker login ghcr.io -u "$user" --password-stdin >/dev/null
DOCKER_CONFIG="$login_dir" docker push "$GHCR_IMAGE:$tag"
DOCKER_CONFIG="$login_dir" docker push "$GHCR_IMAGE:$alias"
log "Done: docker pull $GHCR_IMAGE:$tag"
