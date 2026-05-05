#!/usr/bin/env bash
# Build the AMD LMCache image via Cloud Build on a transient AMD EPYC
# private worker pool. The pool is created, used for one build, and
# deleted — it is NOT kept around (private pools incur charges while
# they exist).
#
# Override defaults via env vars, e.g.:
#   MACHINE=n2d-standard-64 ./build_lmcache_amd.sh

set -euo pipefail

PROJECT="${PROJECT:-character-ai}"
REGION="${REGION:-us-central1}"
POOL="${POOL:-lmcache-amd-pool}"
MACHINE="${MACHINE:-n2d-standard-32}"
DISK_GB="${DISK_GB:-300}"
SHORT_SHA="$(git rev-parse --short HEAD)"

cleanup() {
  echo "Deleting worker pool $POOL ..."
  gcloud builds worker-pools delete "$POOL" \
    --region="$REGION" --project="$PROJECT" --quiet || true
}
trap cleanup EXIT

echo "Creating worker pool $POOL ($MACHINE, ${DISK_GB}GB) ..."
gcloud builds worker-pools create "$POOL" \
  --region="$REGION" --project="$PROJECT" \
  --worker-machine-type="$MACHINE" \
  --worker-disk-size="$DISK_GB"

echo "Submitting build (SHORT_SHA=$SHORT_SHA) ..."
cd "$(git rev-parse --show-toplevel)"
gcloud builds submit \
  --config=docker/cloudbuild.yaml \
  --substitutions="SHORT_SHA=$SHORT_SHA" \
  --project="$PROJECT" \
  .
