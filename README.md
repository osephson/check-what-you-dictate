# Dictation

Upload one audio clip, then hear it sentence by sentence. Type what you heard. The app marks missed and extra words. Transcription runs on this machine with Whisper `large-v3-turbo`.

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

## Practice

- Prepare splits the clip into sentence-like pieces and hides the text
- Jump to sentence N with the number box
- The play bar shows position in the current sentence
- Play starts that sentence from the beginning; Stop pauses it. It does not continue into the next sentence. Speed is 0.75x / 1x / 1.25x
- Check compares your typing to the transcript, ignoring punctuation and capitalization
- Reveal shows the answer if you are stuck
- Skip drops noise or overlapping speech
- Retry missed walks through the sentences you did not get exactly

Shortcuts (practice screen):

- Ctrl+Space play or stop
- `[` / `]` jump 3 seconds and play
- Ctrl+Up / Ctrl+Down change speed
- Ctrl+Enter check, or go to the next sentence after a check
- Alt+Left / Alt+Right do not use the browser back or forward buttons

## Limits

- One file at a time
- 25 MB maximum
- Formats: mp3, wav, m4a, webm, ogg, flac
- Audio stays in the browser for this session. The uploaded copy is deleted on the server when preparation finishes.
