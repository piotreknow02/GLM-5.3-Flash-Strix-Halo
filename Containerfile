# GLM-5.3-Flash EXL3 (yamz-labs) — self-contained Kyojin runtime for AMD Strix Halo (gfx1151).
#
# Builds the Kyojin engine from source: uv-managed Python 3.12, pinned ROCm 7.0+ nightly wheels
# (torch / rocm-sdk-core / rocm-sdk-devel for gfx1151), and the exllamav3 C++ extension compiled for gfx1151.
# The ~100 GB EXL3 checkpoint is NOT in the image — bind-mount it at /models (ro).
#
# Do NOT add a system ROCm to this base: the wheels are self-contained
# (_rocm_sdk_devel inside the venv) and env.sh finds it automatically.
FROM docker.io/library/ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

# System deps: mirrors Kyojin's doc/install.md requirements.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git curl ca-certificates pkg-config \
      libnuma-dev libdrm-dev libelf-dev xxd python3-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

# Kyojin engine: pinned revision for gfx1151 + GLM serving.
ARG KYOJIN_REPO=https://github.com/Yamz-Labs/kyojin.git
ARG KYOJIN_REVISION=f852e2a282bc5bacbe74cac0b3c067dfbf68cb74
# Build parallelism for hipcc / ninja kernel compiles.
# Overridable via --build-arg; empty default = auto-detect via nproc in RUN below.
ARG MAX_JOBS=32

RUN git clone ${KYOJIN_REPO} /opt/kyojin \
    && cd /opt/kyojin \
    && git checkout ${KYOJIN_REVISION}

# Create the venv and install ROCm gfx1151 wheels (torch + rocm-sdk-core + rocm-sdk-devel).
# rocm-sdk-core provides libhsa-runtime64.so.1 and the HIP linker libs needed at build time.
# rocm-sdk-devel provides hipsparse/ thrust/ headers and the hipcc compiler binary.
# `rocm-sdk init` expands the devel headers (~12 GB); without it build.sh stops with
# "no hipsparse headers" (see Kyojin doc/troubleshooting.md).
# EXL3_ROCM_SDK points to the devel SDK (headers); EXL3_HSA_LIB points directly to the HSA .so file.
ENV EXL3_ROCM_SDK=/opt/kyojin/.venv/lib/python3.12/site-packages/_rocm_sdk_devel \
    EXL3_HSA_LIB=/opt/kyojin/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/libhsa-runtime64.so.1 \
    PYTORCH_ROCM_ARCH=gfx1151
# NOTE: --seed matters: uv omits pip from new venvs by default, but Kyojin's
# rebuild.sh runs `python -m pip install` to compile the extension.
RUN uv venv --seed /opt/kyojin/.venv \
    && uv pip install \
      --python /opt/kyojin/.venv/bin/python \
      --pre torch rocm-sdk-core rocm-sdk-devel \
      --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
    && uv pip install \
      --python /opt/kyojin/.venv/bin/python \
      --extra-index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
      -r /opt/kyojin/requirements.txt \
    && /opt/kyojin/.venv/bin/rocm-sdk init

# Build the gfx1151 exllamav3 extension (hipcc compiles C++/HIP kernels).
# rebuild.sh (via build.sh) sources env.sh in build mode itself, picking up
# EXL3_ROCM_SDK / the HIP toolchain / PYTORCH_ROCM_ARCH from the ENV above.
# MAX_JOBS defaults to the builder's core count via nproc; override with
# --build-arg MAX_JOBS=<n> (ninja / PyTorch extension builds respect it).
RUN bash -c 'export MAX_JOBS="${MAX_JOBS:-$(nproc)}" \
    && echo "Building exllamav3 with MAX_JOBS=${MAX_JOBS} (nproc=$(nproc))" \
    && . /opt/kyojin/.venv/bin/activate \
    && . /opt/kyojin/tools/strix_halo/env.sh \
    && cd /opt/kyojin \
    && ./build.sh'

# Kyojin resolves all of its disk caches under $HOME/.cache (dense GEMM tune file,
# startup-history timings, HIP kernel cache). Replace it with a symlink to the runtime
# bind-mount so a fresh container keeps the tuning work of the previous run instead of
# re-tuning (~10 min). rm -rf first: ln -sfn against an existing *directory* would place
# the link inside it (/root/.cache/cache) instead of replacing it.
RUN rm -rf /root/.cache && mkdir -p /cache && ln -s /cache /root/.cache

ENV VENV=/opt/kyojin/.venv \
    MODEL_PATH=/models \
    HOST=0.0.0.0 \
    PORT=8000 \
    CACHE_ROOT=/cache
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30m --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1
# --host 0.0.0.0 is load-bearing: serve.py defaults to 127.0.0.1, which refuses
# connections forwarded from the host publish port. --model-id makes /v1/models
# report (and /v1/chat/completions accept) the HuggingFace repo id.
CMD ["bash", "-c", "source /opt/kyojin/tools/strix_halo/env.sh && exec python /opt/kyojin/tools/glm/serve.py --model /models --host 0.0.0.0 --port 8000 --model-id yamz-labs/GLM-5.3-Flash-EXL3-Yamz -c 262144 --num-draft 2"]
