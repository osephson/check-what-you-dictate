"""Local audio transcription. One file at a time, on this machine."""

from __future__ import annotations

import gc
import logging
import os
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
            )
            lines = [segment.text.strip() for segment in segments if segment.text.strip()]
            return {
                "text": "\n".join(lines),
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
