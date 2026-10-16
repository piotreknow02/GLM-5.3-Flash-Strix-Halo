# GLM-5.3-Flash EXL3 — one-command Podman deploy (AMD Strix Halo 128 GB)

Serves [yamz-labs/GLM-5.3-Flash-EXL3-Yamz](https://huggingface.co/yamz-labs/GLM-5.3-Flash-EXL3-Yamz)
(Kyojin engine built on ExLlamaV3, EXL3 quantisation, gfx1151, MTP speculative decoding) in a
rootless Podman container on an AMD Strix Halo (Ryzen AI Max+ 395).

One command brings it up; one command tears it down.

```bash
cp .env.sample .env   # optional — every var has a default
./start.sh            # build (cached) -> download model -> run -> wait healthy
./stop.sh
```

## What's here

| File            | Role                                                                                                                                                                                            |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Containerfile` | Self-contained runtime: uv Python 3.12, pinned ROCm 7.0+ nightly wheels (torch/triton/rocm-sdk-devel for gfx1151), Kyojin engine cloned from GitHub and compiled. Checksum-pinned git revision. |
| `start.sh`      | One-command entrypoint: preflight → build (cached) → download checkpoint (resumable) → run → health-poll.                                                                                       |
| `stop.sh`       | Stops the container with a 60 s grace period; idempotent.                                                                                                                                       |
| `.env.sample`   | Optional overrides — copy to `.env`.                                                                                                                                                            |

The ~100 GB checkpoint is **not** baked into the image; it is bind-mounted read-only at `/models`. The image is built on first run and cached for subsequent launches.

## Requirements

- **Podman** (rootless) — `podman --version` ≥ 5.x
- **AMD GPU driver** present: `/dev/kfd` and `/dev/dri/renderD128` exist
- Current user in the **`render`** (and ideally `video`) group — check with `id -nG`
- **~130 GB free disk**: ~100 GB weights + ~30 GB image + caches
- **≥128 GB unified RAM** — the default `-c 262144` (256K context) uses most of a 128 GB box; see Troubleshooting if the engine can't allocate
- `curl`, `python3`, and `huggingface-cli` (from `pip install huggingface_hub[cli]`) or `huggingface_hub` Python package on the host for the first-model download

No root/sudo is required for the normal path. Podman runs rootless; GPU access comes from `--group-add keep-groups` (carries your host `render` group into the container) plus SELinux `:z` bind relabeling.

## Quick start

```bash
cp .env.sample .env      # optional
./start.sh
```

First run:

1. Clones and builds the Kyojin engine with ROCm gfx1151 support (~10–20 min).
2. Downloads the ~100 GB checkpoint to `./models` (resumable via `huggingface-cli` or `huggingface_hub`).
3. Starts the container and streams the engine log to `.glm53.log`.
4. Exits "ready" once `/health` returns 200.

Then:

```bash
curl -s http://127.0.0.1:8080/v1/models | jq -r '.data[0].id'
# -> yamz-labs/GLM-5.3-Flash-EXL3-Yamz
```

Tear down:

```bash
./stop.sh
```

Re-running `./start.sh` after the first time skips the build and the download and reaches ready materially faster (Kyojin kernel/GEMM autotune caches live in `./.cache`).

## Configuration

Copy `.env.sample` to `.env` and edit. Precedence (highest first): **exported shell env → `.env` → `start.sh` defaults.**

| Var              | Default                             | Meaning                                                             |
| ---------------- | ----------------------------------- | ------------------------------------------------------------------- |
| `IMAGE`          | `glm53-flash-strix:latest`          | Image tag                                                           |
| `CONTAINER_NAME` | `glm53-flash-strix`                 | Container name                                                      |
| `PORT`           | `8080`                              | Host publish port (container always listens on 8000)                |
| `MODEL_DIR`      | `./models`                          | Where the checkpoint lives / downloads to                           |
| `CACHE_DIR`      | `./.cache`                          | Kyojin/HIP/HF caches (persist kernel/GEMM autotune across restarts) |
| `HF_MODEL`       | `yamz-labs/GLM-5.3-Flash-EXL3-Yamz` | HuggingFace model repo                                              |
| `HF_REVISION`    | *(latest)*                          | Pin a commit SHA for reproducibility                                |
| `LOG_FILE`       | `./.glm53.log`                      | Engine output log                                                   |
| `READY_TIMEOUT`  | `3600`                              | Seconds to wait for the health endpoint                             |
| `MAX_JOBS`       | `$(nproc)`                          | Kernel build parallelism (hipcc/ninja); also `podman build --jobs`  |
| `HF_TOKEN`       | *(unset)*                           | Optional HuggingFace token — higher download rate limits            |

## How it works

The `Containerfile` builds a self-contained image:

1. Ubuntu 24.04 base with system build deps.
2. `uv` for Python management.
3. Kyojin engine git-cloned from GitHub at a pinned commit.
4. ROCm gfx1151 PyTorch + `rocm-sdk-devel` wheel installed into a uv-managed venv (self-contained, no system ROCm), then `rocm-sdk init` expands the devel headers.
5. `build.sh` compiles the exllamav3 C++/HIP extension for gfx1151.
6. At runtime, `env.sh` sets `LD_PRELOAD` (HSA runtime) and `PYTHONPATH`, then `serve.py` serves the model on `0.0.0.0:8000` (host port mapped by `start.sh`) under the HuggingFace repo id, with `-c 262144 --num-draft 2` (256K context). `$HOME/.cache` is a symlink to the `/cache` bind-mount so GEMM tuning and kernel caches survive container recreation.

The image does **not** contain a system ROCm — the wheels are self-contained and the `_rocm_sdk_devel` package provides both the ROCm headers for compilation and the compatible HSA runtime for execution.

## Recommended sampling

From the model card:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "yamz-labs/GLM-5.3-Flash-EXL3-Yamz",
    "messages": [{"role": "user", "content": "Explain what a KV cache is, briefly."}],
    "max_tokens": 256
  }' | jq -r '.choices[0].message'
```

For agent use with reasoning:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "yamz-labs/GLM-5.3-Flash-EXL3-Yamz",
    "messages": [{"role": "user", "content": "Explain why the sky is blue."}],
    "max_tokens": 512,
    "reasoning_effort": "high"
  }'
```

## Troubleshooting

**First launch is slow (5–15 min)** — the engine compiles dense GEMM kernels on the first requests. The progress is printed to the log every 30 seconds. This only happens on a cold cache; subsequent launches warm up fast.

**`Cannot allocate memory` / HSA pin errors at startup** — the user session's `memlock` limit is too low for ROCm memory pinning. `start.sh` only warns about this (it does not block). If the engine actually fails to pin, raise it once (needs sudo):

```bash
sudo mkdir -p /etc/systemd/system/user@.service.d
printf '[Service]\nLimitMEMLOCK=infinity\n' | sudo tee /etc/systemd/system/user@.service.d/10-memlock.conf
sudo systemctl daemon-reload
# then log out and back in
```

**SELinux `Permission denied` on `/models`** — confirm the `:z` suffix on the bind (start.sh sets it). If you relabeled the directory manually, `restorecon -R` it or just re-run `./start.sh`.

**Host OOM under load** — close other apps or reduce the context size by passing a smaller `-c` in the Containerfile's CMD. Default `-c 262144` (256K) leaves little free on a 128 GB box.

**`huggingface-cli: command not found`** — install `huggingface_hub` on the host:

```bash
pip install huggingface_hub[cli]
```

Or ensure `python3`'s environment has `huggingface_hub` installed. start.sh tries both `huggingface-cli` and a Python one-liner using `huggingface_hub`.

**Container exits immediately** — check logs:

```bash
podman logs glm53-flash-strix
```

Common causes: missing GPU driver (`/dev/kfd` not present), `memlock` too low, or the model directory not containing `config.json`.

**No `/dev/kfd` or `/dev/dri`** — the AMD GPU kernel driver is not loaded. Load it:

```bash
sudo modprobe amdgpu
sudo modprobe kfd
```

## Notes

- **No system ROCm in the image.** The wheels are self-contained (`_rocm_sdk_devel` inside the venv); the vendor docs forbid mixing a system ROCm. The container distro (Ubuntu 24.04) is independent of the host distro — only the kernel is shared.
- **`--ipc host` is load-bearing** — the GPU runtime dies at startup without it; no `shm_size` substitutes.
- **Docker instead of Podman** — the same `Containerfile` works under a rootful Docker daemon with `docker run --device /dev/kfd --device /dev/dri --group-add video,render --ipc host --ulimit memlock=-1:-1`. Swap the CLI in `start.sh` if you prefer Docker.

## Model speed reference (AMD Strix Halo, Ryzen AI Max+ 395, 128 GB)

| Metric                                  | Value       |
| --------------------------------------- | ----------- |
| Prefill (3.5K prompt)                   | 580 tok/s   |
| Prefill (14K prompt)                    | 584 tok/s   |
| Prefill (64K prompt)                    | 546 tok/s   |
| Decode (MTP 2 drafts, default sampling) | 26–30 tok/s |
| Decode (MTP 2 drafts, temperature 0)    | ~29 tok/s   |

Speed figures from the [Kyojin README](https://github.com/Yamz-Labs/kyojin) and the [model card](https://huggingface.co/yamz-labs/GLM-5.3-Flash-EXL3-Yamz).

## Licence

The model weights are MIT-licensed (see the Hugo Face model card). The Kyojin engine is MIT-licensed. See the upstream repos for full licence details.

- Base model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) — Copyright (c) 2026 Z.AI Co., Ltd
- Quantisation: [turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3)
- Engine: [Yamz-Labs/kyojin](https://github.com/Yamz-Labs/kyojin) — MIT
- This recipe: authored and published by [piotreknow02](https://github.com/piotreknow02/GLM-5.3-Flash-Strix-Halo) — provided as-is, without warranty
