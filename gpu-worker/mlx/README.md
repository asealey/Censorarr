# Censorarr MLX Worker (Apple Silicon)

An optional transcription worker for Macs with Apple Silicon. It speaks the same HTTP API as the
NVIDIA GPU Worker, so the main Censorarr server needs no changes: enter this Mac's URL and token in
**Settings → Remote GPU worker**. Whisper runs on the Apple GPU through
[`mlx-whisper`](https://github.com/ml-explore/mlx-examples/tree/main/whisper).

Detection, review logic and remuxing stay on the main Censorarr server, exactly as with the CUDA worker.

## Requirements

- Mac with Apple Silicon (M1 or newer)
- `ffmpeg` on `PATH` (`brew install ffmpeg`)
- Python 3.12 (MLX does not publish wheels for every new Python release; [`uv`](https://docs.astral.sh/uv/) makes pinning easy)

The worker must run natively. Docker Desktop on macOS runs a Linux VM with no access to the Apple GPU.

## Install

```bash
cd gpu-worker/mlx
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Run

```bash
cd gpu-worker/mlx
ASR_WORKER_TOKEN='a-long-random-token' .venv/bin/python worker_mlx.py
```

By default it listens on `127.0.0.1:9000`, which is enough when the main Censorarr server runs in
Docker Desktop on the same Mac (use `http://host.docker.internal:9000` as the worker URL).
To serve another machine on your LAN (for example a NAS or Raspberry Pi), listen on all interfaces:

```bash
ASR_HOST=0.0.0.0 ASR_WORKER_TOKEN='a-long-random-token' .venv/bin/python worker_mlx.py
```

Check it:

```bash
curl -H 'X-Censorarr-Token: a-long-random-token' http://127.0.0.1:9000/health
```

A healthy worker reports `"ok": true` and `"mlx_available": true`.

The worker refuses to listen on anything other than localhost without `ASR_WORKER_TOKEN`.

### Run in the background

[`com.censorarr.mlx-worker.plist.example`](com.censorarr.mlx-worker.plist.example) is a LaunchAgent
template that starts the worker at login and restarts it if it exits; the install commands are in
the file. A LaunchAgent only runs while you are logged in, and a sleeping Mac cannot transcribe:
keep `fallback_to_local` on in Censorarr, or schedule processing for times the Mac is awake.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ASR_WORKER_TOKEN` | blank | Shared token; must match the main Censorarr server. Blank disables auth. |
| `ASR_HOST` / `ASR_PORT` | `127.0.0.1` / `9000` | Listen address |
| `ASR_MODEL` | `small.en` | Model used when the request does not name one |
| `ASR_MLX_MODEL` | blank | Force one model regardless of what Censorarr requests |
| `ASR_CHUNK_SECONDS` | `300` | Audio is transcribed in chunks; progress updates after each chunk |
| `ASR_CHUNK_OVERLAP_SECONDS` | `2` | Overlap between chunks (words in the overlap are de-duplicated) |
| `ASR_MAX_UPLOAD_GB` | `2` | Reject larger uploads |
| `ASR_REPEAT_LIMIT` | `8` | Drop a token after this many consecutive repeats (guards against greedy-decoding loops); `0` disables |

### Models

Censorarr's model names map to [mlx-community](https://huggingface.co/mlx-community) conversions,
downloaded to the Hugging Face cache on first use:

| Censorarr model | MLX repo |
|---|---|
| `tiny.en`, `base.en`, `small.en`, `medium.en`, `small`, ... | `mlx-community/whisper-<name>-mlx` |
| `large`, `large-v3` | `mlx-community/whisper-large-v3-mlx` |
| `large-v3-turbo`, `turbo` | `mlx-community/whisper-large-v3-turbo` |
| `distil-large-v3` | `mlx-community/distil-whisper-large-v3` |

Any value containing `/` is used as a Hugging Face repo id directly.

On an M4 (24 GB), `small.en` ran at roughly 17-20x realtime and `large-v3-turbo` at roughly 6-7x,
with `large-v3-turbo` catching slightly more profanity in noisy, overlapping dialogue. `large-v3-turbo`
is a good default when the Mac has the headroom; `medium.en` and `large-v3` were slower without being
more accurate in those tests.

## Differences from the CUDA worker

- `mlx-whisper` decodes greedily and has no VAD, so `beam_size` and `vad_filter` are accepted but ignored (logged per job).
- `language_probability` is reported as `0.0`.
- AI Dialogue (Demucs) is not provided; the worker advertises `dialogue_ai: false`.
