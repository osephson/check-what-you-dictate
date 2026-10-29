"""Local dictation practice. One file at a time, on this machine."""

from __future__ import annotations

import gc
import logging
import os
import re
import sys

# The Xet download path can stall on this network. Use normal HTTPS instead.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import tempfile
import threading
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

MODEL_NAME = "large-v3-turbo"
BATCH_SIZE = 8
MAX_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac"}
SENTENCE_END = re.compile(r"[.!?…][\"'”’)\]]*$")
SENTENCE_SPLIT = re.compile(r'(?<=[.!?…])(?:["\'”’)\]]+)?(?:\s+|$)')
ABBREVIATIONS = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "jr.",
    "sr.",
    "prof.",
    "st.",
    "vs.",
    "etc.",
    "inc.",
    "ltd.",
}
MAX_CHUNK_SECONDS = 12.0
MAX_CHUNK_WORDS = 22
MIN_CHUNK_SECONDS = 0.25

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("transcriber")


def prepare_windows_cuda_dlls() -> None:
    """Let pip-installed CUDA 12 libraries load on Windows."""
    if sys.platform != "win32":
        return

    import importlib.util

    dll_dirs: list[str] = []
    for package in ("nvidia.cublas", "nvidia.cudnn"):
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            continue
        for location in spec.submodule_search_locations:
            bin_dir = os.path.abspath(os.path.join(location, "bin"))
            if os.path.isdir(bin_dir):
                dll_dirs.append(bin_dir)

    for bin_dir in dll_dirs:
        os.add_dll_directory(bin_dir)
    if dll_dirs:
        os.environ["PATH"] = os.pathsep.join(dll_dirs) + os.pathsep + os.environ.get("PATH", "")


prepare_windows_cuda_dlls()


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_sentence_end(token: str) -> bool:
    cleaned = token.strip()
    if not cleaned or not SENTENCE_END.search(cleaned):
        return False
    letters = re.sub(r"[^A-Za-z.]", "", cleaned).lower()
    if letters in ABBREVIATIONS:
        return False
    if re.fullmatch(r"(?:[a-z]\.){2,}", letters):
        return False
    return True


def sentence_chunks(segments) -> list[dict]:
    """Turn Whisper output into playable sentence-like pieces."""
    words = []
    fallback = []
    for segment in segments:
        segment_words = getattr(segment, "words", None)
        if segment_words:
            words.extend(segment_words)
        elif segment.text and segment.text.strip():
            fallback.append(segment)
    if words:
        return _chunks_from_words(words)
    return _chunks_from_segments(fallback)


def _chunks_from_words(words) -> list[dict]:
    chunks: list[dict] = []
    buf: list = []

    def word_count() -> int:
        return sum(1 for item in buf if item.word.strip())

    def flush(end_index: int | None = None) -> None:
        take = buf if end_index is None else buf[: end_index + 1]
        if not take:
            return
        text = _clean_text("".join(item.word for item in take))
        start = float(take[0].start)
        end = float(take[-1].end)
        if text and end - start >= MIN_CHUNK_SECONDS:
            chunks.append(
                {
                    "start": round(start, 2),
                    "end": round(max(end, start + MIN_CHUNK_SECONDS), 2),
                    "text": text,
                }
            )
        if end_index is None:
            buf.clear()
        else:
            del buf[: end_index + 1]

    for word in words:
        if not getattr(word, "word", None) or word.start is None or word.end is None:
            continue
        buf.append(word)
        duration = float(buf[-1].end) - float(buf[0].start)
        ended = _is_sentence_end(word.word)
        too_long = duration >= MAX_CHUNK_SECONDS or word_count() >= MAX_CHUNK_WORDS
        if ended:
            flush()
            continue
        while too_long and buf:
            flush(_split_index(buf))
            if not buf:
                break
            duration = float(buf[-1].end) - float(buf[0].start)
            too_long = duration >= MAX_CHUNK_SECONDS or word_count() >= MAX_CHUNK_WORDS
    flush()
    return chunks


def _split_index(buf) -> int:
    last = len(buf) - 1
    for index in range(last, 2, -1):
        token = buf[index].word.strip()
        if token.endswith((",", ";", ":")):
            return index
    return max(last // 2, 0)


def _chunks_from_segments(segments) -> list[dict]:
    pieces: list[tuple[str, float, float]] = []
    buf_text: list[str] = []
    start: float | None = None
    end: float | None = None

    def flush() -> None:
        nonlocal start, end
        text = _clean_text(" ".join(buf_text))
        if text and start is not None and end is not None and end - start >= MIN_CHUNK_SECONDS:
            pieces.append((text, start, end))
        buf_text.clear()
        start = None
        end = None

    for segment in segments:
        text = _clean_text(segment.text)
        if not text:
            continue
        parts = [part.strip() for part in SENTENCE_SPLIT.split(text) if part.strip()]
        if not parts:
            continue
        span = max(float(segment.end) - float(segment.start), 0.01)
        offset = 0.0
        for part in parts:
            part_start = float(segment.start) + span * (offset / max(len(text), 1))
            offset += len(part) + 1
            part_end = float(segment.start) + span * min(offset / max(len(text), 1), 1.0)
            if start is None:
                start = part_start
            buf_text.append(part)
            end = part_end
            joined = _clean_text(" ".join(buf_text))
            duration = (end - start) if start is not None else 0.0
            if _is_sentence_end(part) or duration >= MAX_CHUNK_SECONDS:
                flush()
    flush()
    return [
        {"start": round(item[1], 2), "end": round(item[2], 2), "text": item[0]}
        for item in pieces
    ]


class Transcriber:
    def __init__(self) -> None:
        self._pipeline = None
        self._busy = threading.Lock()
        self.ready = False
        self.error: str | None = None
        self.device: str | None = None
        self.compute_type: str | None = None

    def load(self) -> None:
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        attempts = (("cuda", "float16"), ("cpu", "int8"))
        for device, compute_type in attempts:
            for try_index in range(3):
                model = None
                pipeline = None
                try:
                    log.info(
                        "Loading %s on %s (%s), attempt %s",
                        MODEL_NAME,
                        device,
                        compute_type,
                        try_index + 1,
                    )
                    model = WhisperModel(MODEL_NAME, device=device, compute_type=compute_type)
                    pipeline = BatchedInferencePipeline(model=model)
                    self._warmup(pipeline)
                except Exception:
                    log.exception("Could not start on %s (%s)", device, compute_type)
                    del pipeline
                    del model
                    gc.collect()
                    if try_index < 2:
                        time.sleep(3)
                    continue

                self._pipeline = pipeline
                self.device = device
                self.compute_type = compute_type
                self.ready = True
                log.info("Model ready on %s (%s)", device, compute_type)
                return

        self.error = "The transcription model could not be loaded. See the server log for details."

    def _warmup(self, pipeline) -> None:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with wave.open(path, "w") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x00" * 16000)
            segments, _info = pipeline.transcribe(
                path,
                batch_size=BATCH_SIZE,
                vad_filter=True,
            )
            for _segment in segments:
                pass
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def transcribe(self, path: str) -> dict:
        if self.error:
            raise HTTPException(status_code=503, detail=self.error)
        if not self.ready or self._pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="The transcription model is still loading. The first run downloads about 1.6 GB.",
            )
        if not self._busy.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="A transcription is already running.")

        try:
            segments, info = self._pipeline.transcribe(
                path,
                batch_size=BATCH_SIZE,
                vad_filter=True,
                word_timestamps=True,
            )
            chunks = sentence_chunks(list(segments))
            if not chunks:
                raise HTTPException(
                    status_code=400,
                    detail="No speech was detected in this file.",
                )
            return {
                "chunks": chunks,
                "language": info.language,
                "language_probability": round(float(info.language_probability), 4),
                "duration_seconds": round(float(info.duration), 2),
                "device": self.device,
                "compute_type": self.compute_type,
            }
        except HTTPException:
            raise
        except Exception:
            log.exception("Transcription failed")
            raise HTTPException(
                status_code=500,
                detail="Transcription failed. Try another file.",
            ) from None
        finally:
            self._busy.release()


transcriber = Transcriber()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=transcriber.load, name="model-load", daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "ready": transcriber.ready,
        "model": MODEL_NAME,
        "device": transcriber.device,
        "compute_type": transcriber.compute_type,
        "error": transcriber.error,
    }


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="This file type is not supported. Use mp3, wav, m4a, webm, ogg, or flac.",
        )

    data = await _read_limited(file)
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        Path(path).write_bytes(data)
        return transcriber.transcribe(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


async def _read_limited(file: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_BYTES:
            raise HTTPException(status_code=400, detail="This file is over 25 MB.")
        chunks.append(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="The file is empty.")
    return b"".join(chunks)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
