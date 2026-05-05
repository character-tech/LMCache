# LMCache Docker Images

This directory contains Dockerfiles for building different LMCache images. Each Dockerfile serves a specific use case depending on your needs.

## Available Dockerfiles

### 1. `Dockerfile` - Full Integration with vLLM

**Image**: `lmcache/vllm-openai:latest`

**Description**: The main Dockerfile that builds LMCache from source and integrates it with vLLM OpenAI server. This is the recommended image for production deployments with full feature support including Prefill-Decode Disaggregation (PD).

**Features**:
- ✅ LMCache built from source
- ✅ vLLM integration (nightly or stable)
- ✅ Full NIXL support for Prefill-Decode Disaggregation
- ✅ CUDA support
- ✅ Optimized multi-stage build

**Build Targets**:
- `image-build`: Builds with vLLM nightly and LMCache from source
- `image-release`: Uses stable vLLM release and LMCache from PyPI

**Usage**:

```bash
# Build with nightly vLLM
docker build \
  --build-arg CUDA_VERSION=12.8 \
  --build-arg UBUNTU_VERSION=24.04 \
  --target image-build \
  --tag lmcache/vllm-openai:latest \
  --file docker/Dockerfile .

# Build with stable releases
docker build \
  --build-arg CUDA_VERSION=12.8 \
  --build-arg UBUNTU_VERSION=24.04 \
  --target image-release \
  --tag lmcache/vllm-openai:latest \
  --file docker/Dockerfile .
```

**Run Example**:

```bash
export HF_TOKEN=<your_huggingface_token>

docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  lmcache/vllm-openai:latest \
  serve Qwen/Qwen3-0.6B \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

---

### 2. `Dockerfile.standalone` - LMCache Only

**Image**: `lmcache/standalone:latest`

**Description**: A standalone Docker image that builds and installs LMCache from source without vLLM. This will be useful when running LMCache in the standalone mode.

**Features**:
- ✅ LMCache built from source
- ✅ No vLLM dependency
- ✅ CUDA support

**Build Target**:
- `lmcache-final`: Final optimized image with LMCache installed

**Usage**:

```bash
docker build \
  --build-arg CUDA_VERSION=12.8 \
  --build-arg UBUNTU_VERSION=24.04 \
  --target lmcache-final \
  --tag lmcache/standalone:latest \
  --file docker/Dockerfile.standalone .
```

**Run Example**:

```bash
# Start an interactive shell
docker run --runtime nvidia --gpus all -it \
  lmcache/standalone:latest \
  /opt/venv/bin/python3 \
  -m lmcache.v1.multiprocess.server \
  --cpu-buffer-size 60 \
  --max-workers 4 \
  --max-gpu-workers 2 \
  --port 6555
```

---

### 3. `Dockerfile.lightweight` - Quick Setup

**Image**: `lmcache/vllm-openai:lightweight`

**Description**: A lightweight image that extends the official vLLM image and installs LMCache from PyPI. This is the fastest way to get started but does not include NIXL support.

**Features**:
- ✅ Based on official `vllm/vllm-openai:latest` image
- ✅ LMCache installed from PyPI (latest release)
- ✅ Quick build time
- ✅ Small image size
- ❌ No NIXL support (no Prefill-Decode Disaggregation)

**Limitations**:
- Cannot use Prefill-Decode Disaggregation features

**Usage**:

```bash
docker build \
  --tag lmcache/vllm-openai:lightweight \
  --file docker/Dockerfile.lightweight .
```

**Run Example**:

```bash
export HF_TOKEN=<your_huggingface_token>

docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8000:8000 \
  --ipc=host \
  lmcache/vllm-openai:lightweight \
  serve Qwen/Qwen3-0.6B \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

---

### 4. `Dockerfile.amd` - AMD GPU Standalone (MI325X / MI350)

**Image**: `gcr.io/character-ai/lmcache/lmcache-amd:<sha>`

**Description**: Standalone LMCache server image for AMD GPUs. Builds the
LMCache HIP extension (`rocm_extension` in `setup.py`) as a fat binary
covering both `gfx942` (MI325X) and `gfx950` (MI350). Used by the
LMCache operator's DaemonSet for MP-mode deployments. The vLLM
container does NOT need LMCache compiled in for MP mode — it talks to
this server over HIP IPC via `LMCacheMPConnector`.

**Features**:
- ✅ LMCache built from source with `BUILD_WITH_HIP=1`
- ✅ Fat binary: one image runs on MI325X (gfx942) and MI350 (gfx950)
- ✅ ROCm 7.2.3, PyTorch 2.10, Python 3.12 (from `rocm/pytorch` base)
- ✅ Cross-compiled AOT via `hipcc` — no AMD GPU needed at build time

**Build via Cloud Build (recommended)**:

Cloud Build's stock pool only supports Intel hosts. To build on AMD
EPYC hardware (matching the runtime CPU architecture), we use a
**transient private worker pool** that is created before each build
and deleted after. The lifecycle is wrapped in `build_lmcache_amd.sh`:

```bash
cd ~/git/LMCache
./docker/build_lmcache_amd.sh
```

The script:
1. Creates worker pool `lmcache-amd-pool` (`n2d-standard-32`, 300 GB
   disk) in `us-central1`.
2. Submits the build per `docker/cloudbuild.yaml`. The build pushes
   `gcr.io/character-ai/lmcache/lmcache-amd:<short-sha>` and
   `:dev_<build-id>`.
3. Deletes the worker pool (via `trap cleanup EXIT` — runs even on
   build failure).

Override defaults via env vars:

```bash
MACHINE=n2d-standard-64 DISK_GB=400 ./docker/build_lmcache_amd.sh
```

> **Why `n2d` and not `c2d`/`c3d`?** Cloud Build private worker pools
> only accept the `e2`, `n2d`, and `c3` machine families per the
> [worker-pool config schema][wp-schema]. `n2d` is the AMD EPYC
> option. `c2d`/`c3d` exist as GCE machine types but are NOT accepted
> by the worker-pool API.
>
> [wp-schema]: https://docs.cloud.google.com/build/docs/private-pools/worker-pool-config-file-schema

**Required GCP permissions**:
- `cloudbuild.workerPools.create`, `cloudbuild.workerPools.delete`
- `cloudbuild.builds.create`
- `storage.objects.create` on the GCR bucket

**Manual local build** (only useful on a Linux dev box with ROCm
installed):

```bash
cd ~/git/LMCache
docker build -f docker/Dockerfile.amd -t lmcache-amd:test .
```

**Run example** (operator's DaemonSet sets the args; this is just for
ad-hoc smoke testing):

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --ipc=host \
  gcr.io/character-ai/lmcache/lmcache-amd:<sha> --help
```

---

## Which Dockerfile Should I Use?

### Use `Dockerfile` if you:
- Need full LMCache + vLLM integration
- Want Prefill-Decode Disaggregation support
- Are deploying to production
- Need the latest features built from source

### Use `Dockerfile.standalone` if you:
- Want LMCache without vLLM
- Need a clean LMCache installation for development
- Want to integrate LMCache with custom tools

### Use `Dockerfile.lightweight` if you:
- Prefer stable releases from PyPI
- Need fast build times

### Use `Dockerfile.amd` if you:
- Are deploying LMCache MP mode on AMD MI325X or MI350 GPUs
- Need the standalone LMCache server (operator DaemonSet pattern)

---

## Build Arguments

All Dockerfiles support the following build arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `CUDA_VERSION` | `12.8` | CUDA version to use |
| `UBUNTU_VERSION` | `24.04` | Ubuntu base version |
| `PYTHON_VERSION` | `3.12` | Python version |
| `max_jobs` | `2` | Max parallel jobs for build |
| `nvcc_threads` | `8` | Number of nvcc threads |
| `torch_cuda_arch_list` | `7.0 7.5 8.0 8.6 8.9 9.0 10.0 12.0+PTX` | CUDA architectures |

**Example with custom arguments**:

```bash
docker build \
  --build-arg CUDA_VERSION=12.4 \
  --build-arg max_jobs=4 \
  --build-arg nvcc_threads=16 \
  --target image-build \
  --tag lmcache/vllm-openai:cuda12.4 \
  --file docker/Dockerfile .
```

---

## Published Images

Pre-built images are available on Docker Hub:

- `lmcache/vllm-openai:latest` - Latest stable release with vLLM
- `lmcache/vllm-openai:{version}` - Specific version (e.g., `v0.1.0`)
- `lmcache/vllm-openai:lightweight` - Lightweight version
- `lmcache/standalone:latest` - Latest standalone release
- `lmcache/standalone:{version}` - Specific standalone version

```bash
# Pull pre-built images
docker pull lmcache/vllm-openai:latest
docker pull lmcache/standalone:latest
```

---

## Additional Resources

- [LMCache Documentation](https://docs.lmcache.ai/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Installation Guide](https://docs.lmcache.ai/getting_started/installation.html)
- [Docker Deployment Guide](https://docs.lmcache.ai/production/docker_deployment.html)

