# Transcriber

Upload one audio file in the browser and get a plain-text transcript. Transcription runs on this machine with Whisper `large-v3-turbo`.

## Run

From this folder:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

The first start downloads the model (about 1.6 GB) into the Hugging Face cache. Later starts use that copy.

The app tries the GPU first (CUDA, float16). If the CUDA libraries do not load, it uses the CPU (int8) and the page says which one is active.

## Limits

- One file at a time
- 25 MB maximum
- Formats: mp3, wav, m4a, webm, ogg, flac
- The audio file is deleted when the request finishes. The transcript stays in the browser until you copy it, download it, or clear it.
