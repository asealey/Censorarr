"""Apple Silicon (MLX) transcription worker for Censorarr.

Speaks the same HTTP contract as ``gpu-worker/app/worker.py`` so the main Censorarr server can
use it as its "Remote GPU worker" with no changes: point the worker URL at this Mac and reuse
the token. Whisper runs through ``mlx_whisper`` on the Apple GPU instead of CUDA/CTranslate2.

Runs natively on macOS (Docker on a Mac has no access to the Apple GPU).
"""
from __future__ import annotations

import asyncio
import base64
import hmac
import multiprocessing as mp
import os
import platform
import re
import tempfile
import threading
import time
import traceback
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request

VERSION = '1.7.0-mlx.1'
DEFAULT_MODEL = os.environ.get('ASR_MODEL', 'small.en')
# Optional: force one model regardless of what the main Censorarr server requests.
MODEL_OVERRIDE = os.environ.get('ASR_MLX_MODEL', '').strip()
TOKEN = os.environ.get('ASR_WORKER_TOKEN', '')
MAX_UPLOAD_GB = float(os.environ.get('ASR_MAX_UPLOAD_GB', '2'))
LOG_LINES = max(250, int(os.environ.get('ASR_LOG_LINES', '4000')))
CHUNK_SECONDS = max(60.0, float(os.environ.get('ASR_CHUNK_SECONDS', '300')))
OVERLAP_SECONDS = max(0.0, min(10.0, float(os.environ.get('ASR_CHUNK_OVERLAP_SECONDS', '2'))))
REPEAT_LIMIT = int(os.environ.get('ASR_REPEAT_LIMIT', '8'))

# Censorarr sends faster-whisper style names (small.en, large-v3, ...). Map them to the
# mlx-community conversions; anything containing "/" is treated as a Hugging Face repo id.
MODEL_ALIASES = {
    'large': 'mlx-community/whisper-large-v3-mlx',
    'large-v3-turbo': 'mlx-community/whisper-large-v3-turbo',
    'turbo': 'mlx-community/whisper-large-v3-turbo',
    'distil-large-v3': 'mlx-community/distil-whisper-large-v3',
}

# Model names a client may request. Hugging Face repo ids and local paths ("/") are accepted only
# from this Mac's own environment (ASR_MODEL / ASR_MLX_MODEL), never from a request.
REQUEST_MODEL_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$')

_log = deque(maxlen=LOG_LINES)
_log_lock = threading.RLock()


def log_line(message: str, level: str = 'INFO') -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {level} {message}"
    with _log_lock:
        _log.append(line)
    print(line, flush=True)


def _engine_log(event_q, message: str, level: str = 'INFO') -> None:
    try:
        event_q.put({'type': 'log', 'level': level, 'message': message})
    except Exception:
        pass


def _fmt_clock(seconds: float | int | None) -> str:
    try:
        total = max(0, int(float(seconds or 0)))
    except (TypeError, ValueError):
        total = 0
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{sec:02d}'


def _drop_repetition_loops(words: list[dict[str, Any]], limit: int) -> tuple[list[dict[str, Any]], int]:
    # Greedy decoding (mlx-whisper has no beam search) occasionally falls into a loop such as
    # "da da da ..." filling a whole 30s window. Keep the first `limit` repeats of a token and
    # drop the rest so a runaway loop cannot become hundreds of words or one giant mute range.
    if limit <= 0:
        return words, 0
    kept: list[dict[str, Any]] = []
    run_token, run_len, dropped = None, 0, 0
    for w in words:
        token = ''.join(ch for ch in str(w.get('word', '')).lower() if ch.isalnum())
        if token and token == run_token:
            run_len += 1
        else:
            run_token, run_len = token, 1
        if run_len > limit:
            dropped += 1
            continue
        kept.append(w)
    return kept, dropped


def resolve_model(name: str | None) -> str:
    name = (MODEL_OVERRIDE or name or DEFAULT_MODEL).strip()
    if '/' in name:
        return name
    return MODEL_ALIASES.get(name, f'mlx-community/whisper-{name}-mlx')


def _engine_main(cmd_q, result_q, event_q) -> None:
    # MLX lives only in this child process so a cancel can terminate it and start a fresh one,
    # exactly like the CUDA worker.
    import mlx_whisper
    from mlx_whisper.audio import SAMPLE_RATE, load_audio

    loaded: set[str] = set()
    _engine_log(event_q, f'MLX ASR engine started pid={os.getpid()}')
    while True:
        cmd = cmd_q.get()
        if not isinstance(cmd, dict):
            continue
        if cmd.get('op') == 'shutdown':
            _engine_log(event_q, 'MLX ASR engine shutting down')
            return
        if cmd.get('op') != 'transcribe':
            continue

        job_id = str(cmd.get('job_id') or '')
        name = str(cmd.get('model') or DEFAULT_MODEL)
        repo = resolve_model(name)
        started = time.time()
        try:
            stage = 'transcribing' if repo in loaded else 'loading-model'
            event_q.put({'type': 'status', 'job_id': job_id, 'stage': stage, 'model': name, 'started': started})
            if repo not in loaded:
                _engine_log(event_q, f'Loading Whisper model {repo} on mlx for job {job_id[:8]} (first use downloads it)')

            audio = load_audio(str(cmd.get('audio_path') or ''))
            total = len(audio)
            duration = total / float(SAMPLE_RATE)
            chunk = int(CHUNK_SECONDS * SAMPLE_RATE)
            overlap = int(OVERLAP_SECONDS * SAMPLE_RATE)
            step = max(1, chunk - overlap)
            chunk_count = 1 if total <= chunk else 1 + (total - chunk + step - 1) // step
            if chunk_count > 1:
                _engine_log(event_q, f'Chunking audio for job {job_id[:8]}: {_fmt_clock(duration)} into {chunk_count} '
                                     f'chunk(s) of up to {_fmt_clock(CHUNK_SECONDS)} with {OVERLAP_SECONDS:.1f}s overlap')

            kwargs = {
                'path_or_hf_repo': repo,
                'word_timestamps': True,
                'condition_on_previous_text': bool(cmd.get('condition_on_previous_text', False)),
                'initial_prompt': cmd.get('prompt') or None,
                'verbose': None,
            }
            if cmd.get('language'):
                kwargs['language'] = cmd['language']

            words: list[dict[str, Any]] = []
            language = cmd.get('language') or ''
            segments_done = 0
            start = 0
            index = 0
            while start < total:
                index += 1
                end = min(total, start + chunk)
                chunk_start = start / float(SAMPLE_RATE)
                if chunk_count > 1:
                    _engine_log(event_q, f'MLX job {job_id[:8]} chunk {index}/{chunk_count}: '
                                         f'{_fmt_clock(chunk_start)}-{_fmt_clock(end / float(SAMPLE_RATE))}')
                result = mlx_whisper.transcribe(audio[start:end], **kwargs)
                if repo not in loaded:
                    loaded.add(repo)
                    event_q.put({'type': 'model-loaded', 'model': name, 'compute': 'mlx'})
                if index == 1:
                    language = result.get('language') or language

                # Give half of the overlap to each adjacent chunk (same rule as the CUDA worker).
                left_owner = chunk_start + (OVERLAP_SECONDS / 2.0 if index > 1 else 0.0)
                right_owner = end / float(SAMPLE_RATE) - (OVERLAP_SECONDS / 2.0 if end < total else 0.0)
                for seg in result.get('segments') or []:
                    segments_done += 1
                    for w in seg.get('words') or []:
                        abs_start = chunk_start + float(w['start'])
                        abs_end = chunk_start + float(w['end'])
                        midpoint = (abs_start + abs_end) / 2.0
                        if midpoint < left_owner or midpoint >= right_owner:
                            continue
                        prob = w.get('probability')
                        words.append({
                            'start': abs_start,
                            'end': abs_end,
                            'word': (w.get('word') or '').strip(),
                            'probability': float(prob) if prob is not None else None,
                        })

                position = end / float(SAMPLE_RATE)
                progress = min(99.9, 100.0 * position / duration) if duration > 0 else None
                event_q.put({
                    'type': 'status', 'job_id': job_id, 'stage': 'transcribing', 'model': name,
                    'started': started, 'progress': round(progress, 1) if progress is not None else None,
                    'position_seconds': round(position, 3), 'duration_seconds': duration,
                    'segments_done': segments_done, 'words_count': len(words),
                })
                if end >= total:
                    break
                start += step

            words, dropped = _drop_repetition_loops(words, REPEAT_LIMIT)
            if dropped:
                _engine_log(event_q, f'MLX job {job_id[:8]}: dropped {dropped} word(s) from a repetition loop', 'WARNING')
            elapsed = round(time.time() - started, 3)
            result_q.put({
                'job_id': job_id,
                'ok': True,
                'data': {
                    'ok': True,
                    'version': VERSION,
                    'job_id': job_id,
                    'model': name,
                    'compute_type': 'mlx',
                    'device': 'mlx',
                    'duration': duration,
                    'language': language,
                    'language_probability': 0.0,
                    'elapsed_seconds': elapsed,
                    'words': words,
                },
            })
            speed = f' ({duration / elapsed:.1f}x realtime)' if elapsed > 0 else ''
            _engine_log(event_q, f'Completed job {job_id[:8]} model={repo} elapsed={elapsed}s words={len(words)}{speed}')
        except BaseException as exc:
            result_q.put({'job_id': job_id, 'ok': False, 'error': f'{type(exc).__name__}: {exc}'})
            _engine_log(event_q, f'Job {job_id[:8]} failed: {type(exc).__name__}: {exc}', 'ERROR')
            _engine_log(event_q, traceback.format_exc(limit=8), 'DEBUG')
        finally:
            try:
                event_q.put({'type': 'status', 'job_id': job_id, 'stage': 'idle'})
            except Exception:
                pass


class EngineManager:
    """Same lifecycle as the CUDA worker's EngineManager: one job at a time, cancel = restart engine."""

    def __init__(self) -> None:
        self.ctx = mp.get_context('spawn')
        self.cmd_q = self.ctx.Queue()
        self.result_q = self.ctx.Queue()
        self.event_q = self.ctx.Queue()
        self.proc: mp.Process | None = None
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)
        self.results: dict[str, dict] = {}
        self.cancelled_jobs: set[str] = set()
        self.current_job: dict[str, Any] | None = None
        self.loaded_models: set[str] = set()
        self.started = False

    def start(self) -> None:
        with self.lock:
            if not self.started:
                self.started = True
                threading.Thread(target=self._result_pump, daemon=True, name='asr-result-pump').start()
                threading.Thread(target=self._event_pump, daemon=True, name='asr-event-pump').start()
            self._start_engine_locked()

    def _start_engine_locked(self) -> None:
        if self.proc and self.proc.is_alive():
            return
        self.proc = self.ctx.Process(
            target=_engine_main, args=(self.cmd_q, self.result_q, self.event_q),
            daemon=True, name='censorarr-mlx-engine',
        )
        self.proc.start()
        log_line(f'MLX ASR engine process launched pid={self.proc.pid}')

    def shutdown(self) -> None:
        with self.lock:
            p = self.proc
            if not p:
                return
            if p.is_alive():
                try: self.cmd_q.put({'op': 'shutdown'})
                except Exception: pass
                p.join(timeout=3)
            if p.is_alive():
                p.terminate(); p.join(timeout=2)
            self.proc = None

    def _result_pump(self) -> None:
        while True:
            try:
                item = self.result_q.get()
            except Exception:
                time.sleep(.1); continue
            if not isinstance(item, dict):
                continue
            jid = str(item.get('job_id') or '')
            with self.cond:
                if jid in self.cancelled_jobs:
                    continue
                self.results[jid] = item
                self.cond.notify_all()

    def _event_pump(self) -> None:
        while True:
            try:
                ev = self.event_q.get()
            except Exception:
                time.sleep(.1); continue
            if not isinstance(ev, dict):
                continue
            typ = ev.get('type')
            if typ == 'log':
                log_line(str(ev.get('message', '')), str(ev.get('level', 'INFO')))
            elif typ == 'model-loaded':
                with self.lock:
                    self.loaded_models.add(f"{ev.get('model')}/{ev.get('compute')}")
            elif typ == 'status':
                jid = str(ev.get('job_id') or '')
                with self.lock:
                    if self.current_job and str(self.current_job.get('job_id')) == jid:
                        if ev.get('stage') == 'idle':
                            self.current_job['stage'] = 'finishing'
                        else:
                            self.current_job.update({
                                k: v for k, v in ev.items()
                                if k in {'stage', 'model', 'started', 'progress', 'position_seconds',
                                         'duration_seconds', 'segments_done', 'words_count'}
                            })

    def run(self, job_id: str, audio_path: Path, params: dict[str, Any]) -> dict:
        with self.cond:
            if self.current_job is not None:
                raise RuntimeError(f"GPU worker is busy with job {self.current_job.get('job_id')}")
            self._start_engine_locked()
            self.current_job = {
                'job_id': job_id, 'stage': 'queued', 'model': params.get('model'),
                'started': time.time(), 'audio_bytes': audio_path.stat().st_size,
                'progress': 0.0, 'position_seconds': 0.0, 'duration_seconds': None,
                'segments_done': 0, 'words_count': 0,
            }
            self.results.pop(job_id, None)
            self.cancelled_jobs.discard(job_id)
            self.cmd_q.put({'op': 'transcribe', 'job_id': job_id, 'audio_path': str(audio_path), **params})
            while True:
                item = self.results.pop(job_id, None)
                if item is not None:
                    if self.current_job and self.current_job.get('job_id') == job_id:
                        self.current_job = None
                    if item.get('cancelled'):
                        self.cancelled_jobs.discard(job_id)
                        raise InterruptedError('Remote GPU job cancelled')
                    if not item.get('ok'):
                        raise RuntimeError(str(item.get('error') or 'ASR engine failed'))
                    data = dict(item.get('data') or {})
                    data['gpu'] = gpu_info()
                    return data
                p = self.proc
                if not p or not p.is_alive():
                    self.current_job = None
                    raise RuntimeError('ASR engine process exited unexpectedly')
                self.cond.wait(timeout=.5)

    def cancel(self, job_id: str) -> dict:
        with self.cond:
            cur = self.current_job
            if not cur or str(cur.get('job_id')) != job_id:
                return {'ok': False, 'cancelled': False, 'job_id': job_id, 'message': 'Job is not currently running'}
            log_line(f'Cancelling MLX job {job_id[:8]} by terminating ASR engine', 'WARNING')
            p = self.proc
            if p and p.is_alive():
                p.terminate(); p.join(timeout=3)
                if p.is_alive():
                    p.kill(); p.join(timeout=2)
            self.proc = None
            self.loaded_models.clear()
            self.cancelled_jobs.add(job_id)
            self.results[job_id] = {'job_id': job_id, 'ok': False, 'cancelled': True}
            self.current_job = None
            self._start_engine_locked()
            self.cond.notify_all()
            return {'ok': True, 'cancelled': True, 'job_id': job_id, 'message': 'MLX job cancelled; ASR engine restarted'}

    def status(self) -> dict:
        with self.lock:
            p = self.proc
            cur = dict(self.current_job) if self.current_job else None
            if cur and cur.get('started'):
                cur['elapsed_seconds'] = round(time.time() - float(cur['started']), 1)
                try:
                    pct = float(cur.get('progress'))
                    if 0.1 <= pct < 100.0:
                        cur['eta_seconds'] = round(cur['elapsed_seconds'] * (100.0 - pct) / pct, 1)
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
            return {
                'engine_alive': bool(p and p.is_alive()),
                'engine_pid': p.pid if p and p.is_alive() else None,
                'current_job': cur,
                'loaded_models': sorted(self.loaded_models),
            }


engine = EngineManager()


def gpu_info() -> dict:
    # cuda_devices stays 0 so nothing in Censorarr mistakes this for an NVIDIA worker;
    # "features" tells AI Dialogue (Demucs) that it is not available here.
    out: dict[str, Any] = {
        'cuda_devices': 0, 'mlx_available': False, 'platform': f'{platform.system()} {platform.machine()}',
        'features': {'transcription': True, 'dialogue_ai': False},
    }
    try:
        import mlx.core as mx
        out['mlx_available'] = bool(mx.metal.is_available())
        out['mlx_version'] = mx.__version__
        info = mx.device_info() if hasattr(mx, 'device_info') else mx.metal.device_info()
        out['gpu_name'] = info.get('device_name') or info.get('architecture')
        if info.get('memory_size'):
            out['gpu_memory_gb'] = round(int(info['memory_size']) / 1024**3, 1)
    except Exception:
        pass
    return out


def auth(request: Request) -> None:
    supplied = request.headers.get('X-Censorarr-Token', '')
    if TOKEN and not hmac.compare_digest(supplied.encode('utf-8'), TOKEN.encode('utf-8')):
        raise HTTPException(401, 'Invalid worker token')


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not TOKEN:
        log_line('ASR_WORKER_TOKEN is blank; any client on the network can use this worker', 'WARNING')
    engine.start()
    log_line(f'Censorarr MLX worker {VERSION} ready; default_model={resolve_model(DEFAULT_MODEL)}'
             f'{" (override)" if MODEL_OVERRIDE else ""}')
    yield
    engine.shutdown()


app = FastAPI(title='Censorarr MLX ASR Worker', version=VERSION, lifespan=lifespan)


@app.get('/health')
def health(request: Request):
    auth(request); g = gpu_info(); s = engine.status()
    return {
        'ok': g['mlx_available'], 'version': VERSION, 'device': 'mlx', 'default_model': DEFAULT_MODEL,
        'compute_type': 'mlx', 'loaded_models': s['loaded_models'], **g,
        'engine_alive': s['engine_alive'], 'engine_pid': s['engine_pid'],
    }


@app.get('/status')
def status(request: Request):
    auth(request); g = gpu_info(); s = engine.status()
    return {'ok': g['mlx_available'], 'version': VERSION, 'default_model': DEFAULT_MODEL,
            'compute_type': 'mlx', **g, **s}


@app.get('/logs')
def logs(request: Request, lines: int = Query(default=250, ge=10, le=5000)):
    auth(request)
    with _log_lock:
        data = list(_log)[-lines:]
    return {'ok': True, 'lines': data}


@app.post('/logs/clear')
def clear_logs(request: Request):
    auth(request)
    with _log_lock: _log.clear()
    log_line('MLX worker log cleared')
    return {'ok': True}


@app.delete('/jobs/{job_id}')
def cancel_job(job_id: str, request: Request):
    auth(request)
    return engine.cancel(job_id)


@app.post('/transcribe')
async def transcribe(
    request: Request,
    job_id: str = Query(default=''),
    model: str = Query(default=''),
    language: str = Query(default='en'),
    beam_size: int = Query(default=5, ge=1, le=10),
    vad_filter: int = Query(default=0),
    condition_on_previous_text: int = Query(default=0),
):
    auth(request)
    if not gpu_info()['mlx_available']:
        raise HTTPException(503, 'No Apple GPU (Metal) visible to the MLX worker')
    clen = request.headers.get('content-length')
    try: clen_bytes = int(clen) if clen else 0
    except ValueError: raise HTTPException(400, 'Invalid Content-Length')
    if clen_bytes > int(MAX_UPLOAD_GB * 1024**3):
        raise HTTPException(413, 'Audio upload exceeds ASR_MAX_UPLOAD_GB')
    prompt = ''
    if request.headers.get('X-Censorarr-Prompt-B64'):
        try: prompt = base64.b64decode(request.headers['X-Censorarr-Prompt-B64']).decode('utf-8')
        except Exception: raise HTTPException(400, 'Invalid prompt header')
    name = (model or DEFAULT_MODEL).strip()
    if model and not MODEL_OVERRIDE and not REQUEST_MODEL_RE.match(name):
        raise HTTPException(400, 'Unsupported model name; set ASR_MLX_MODEL on the worker to use a custom repo')
    jid = (job_id or request.headers.get('X-Censorarr-Job-ID') or uuid.uuid4().hex).strip()
    total = 0
    fd, tmp = tempfile.mkstemp(prefix='censorarr-asr-', suffix='.wav'); os.close(fd); p = Path(tmp)
    try:
        with p.open('wb') as f:
            async for chunk in request.stream():
                total += len(chunk)
                if total > int(MAX_UPLOAD_GB * 1024**3):
                    raise HTTPException(413, 'Audio upload too large')
                f.write(chunk)
        log_line(f'Received job {jid[:8]} model={name} bytes={total}')
        # mlx_whisper decodes greedily and has no VAD; accept the options so requests don't fail.
        if beam_size > 1 or vad_filter:
            log_line(f'Job {jid[:8]}: beam_size={beam_size} vad_filter={bool(vad_filter)} not supported by MLX; '
                     'using greedy decoding without VAD')
        params = {
            'model': name, 'language': language or None,
            'condition_on_previous_text': bool(condition_on_previous_text),
            'prompt': prompt or None,
        }
        try:
            return await asyncio.to_thread(engine.run, jid, p, params)
        except InterruptedError:
            raise HTTPException(499, 'GPU transcription cancelled')
        except RuntimeError as exc:
            if 'busy with job' in str(exc):
                raise HTTPException(409, str(exc))
            raise HTTPException(500, str(exc))
    finally:
        try: p.unlink()
        except OSError: pass


if __name__ == '__main__':
    import uvicorn
    host = os.environ.get('ASR_HOST', '127.0.0.1')
    if host not in {'127.0.0.1', 'localhost', '::1'} and not TOKEN:
        raise SystemExit(f'Refusing to listen on {host} without ASR_WORKER_TOKEN; set a token or use ASR_HOST=127.0.0.1')
    uvicorn.run(app, host=host, port=int(os.environ.get('ASR_PORT', '9000')), workers=1)
