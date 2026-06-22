#!/usr/bin/env bash
# Build the combined vLLM + LMCache AMD image via Cloud Build on a
# transient AMD EPYC private worker pool. Mirrors build_lmcache_amd.sh
# but uses docker/cloudbuild-combined.yaml and vendors vLLM source from
# a local checkout (Cloud Build has no GitHub credentials for our
# private character-tech/vllm repo).
#
# Override defaults via env vars, e.g.:
#   VLLM_SRC=~/git/vllm VLLM_BRANCH=cai-v0.19.0 ./build_lmcache_vllm_amd.sh

set -euo pipefail

PROJECT="${PROJECT:-character-ai}"
REGION="${REGION:-us-central1}"
POOL="${POOL:-lmcache-amd-pool}"
MACHINE="${MACHINE:-n2d-standard-32}"
DISK_GB="${DISK_GB:-300}"
VLLM_SRC="${VLLM_SRC:-$HOME/git/vllm}"
VLLM_BRANCH="${VLLM_BRANCH:-cai-v0.19.0}"
# Version slugs used in the descriptive image tag. Derived from branch
# names by default (cai-v0.19.0 → 0.19.0, cai-v0.4.4 → 0.4.4).
VLLM_VER="${VLLM_VER:-${VLLM_BRANCH#cai-v}}"
LMCACHE_VER="${LMCACHE_VER:-0.4.7}"
ROCM_VER="${ROCM_VER:-rocm7.0.2}"
TORCH_VER="${TORCH_VER:-torch2.10.0}"

LMCACHE_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
SHORT_SHA="$(git -C "${LMCACHE_ROOT}" rev-parse --short HEAD)"
STAGE_DIR="${LMCACHE_ROOT}/vllm-src"

if [[ ! -d "${VLLM_SRC}/.git" ]]; then
  echo "VLLM_SRC=${VLLM_SRC} is not a git repository" >&2
  exit 1
fi

cleanup() {
  echo "Deleting worker pool ${POOL} ..."
  gcloud builds worker-pools delete "${POOL}" \
    --region="${REGION}" --project="${PROJECT}" --quiet || true
  echo "Removing staged vLLM source ${STAGE_DIR} ..."
  rm -rf "${STAGE_DIR}"
}
trap cleanup EXIT

echo "Staging vLLM source (${VLLM_SRC} @ ${VLLM_BRANCH}) into ${STAGE_DIR} ..."
rm -rf "${STAGE_DIR}"
# git archive produces a pristine tarball of the branch tip without .git
# history; saves ~1.5 GB on the Cloud Build upload.
git -C "${VLLM_SRC}" archive --format=tar "${VLLM_BRANCH}" \
  | tar -x -C "${STAGE_DIR}" --one-top-level="${STAGE_DIR}" --strip-components=0 \
  2>/dev/null || {
    # Fallback for older tar that doesn't support --one-top-level
    mkdir -p "${STAGE_DIR}"
    git -C "${VLLM_SRC}" archive --format=tar "${VLLM_BRANCH}" | tar -x -C "${STAGE_DIR}"
  }
test -f "${STAGE_DIR}/setup.py" || { echo "vllm-src/setup.py missing — staging failed" >&2; exit 1; }
echo "Staged vLLM source: $(find "${STAGE_DIR}" -maxdepth 1 -type d | wc -l) top-level dirs, $(du -sh "${STAGE_DIR}" | cut -f1)"

echo "Creating worker pool ${POOL} (${MACHINE}, ${DISK_GB}GB) ..."
gcloud builds worker-pools create "${POOL}" \
  --region="${REGION}" --project="${PROJECT}" \
  --worker-machine-type="${MACHINE}" \
  --worker-disk-size="${DISK_GB}"

DESCRIPTIVE_TAG="vllm-${VLLM_VER}-lmcache-${LMCACHE_VER}-${ROCM_VER}-${TORCH_VER}-${SHORT_SHA}"
echo "Submitting build (SHORT_SHA=${SHORT_SHA}, descriptive tag: ${DESCRIPTIVE_TAG}) ..."
cd "${LMCACHE_ROOT}"
gcloud builds submit \
  --config=docker/cloudbuild-combined.yaml \
  --substitutions="SHORT_SHA=${SHORT_SHA},_VLLM_VER=${VLLM_VER},_LMCACHE_VER=${LMCACHE_VER},_ROCM_VER=${ROCM_VER},_TORCH_VER=${TORCH_VER}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  .
