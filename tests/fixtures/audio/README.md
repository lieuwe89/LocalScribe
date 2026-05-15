# Audio fixtures

These wav files are required for integration tests. They are *not* committed
to git (binary churn). Regenerate locally:

```bash
python tests/fixtures/generate_fixtures.py
```

Required system tools:
- `espeak-ng` (Linux: `apt install espeak-ng`; macOS: `brew install espeak-ng`)
- `ffmpeg`

Produces:
- `nl_2speakers_10s.wav` — Dutch, two synthetic speakers
- `en_2speakers_10s.wav` — English, two synthetic speakers
- `en_silence_then_speech.wav` — 3s silence then English speech
