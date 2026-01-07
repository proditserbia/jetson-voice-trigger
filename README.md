# Jetson Voice Trigger (Offline) — VAD + faster-whisper (CUDA with CPU fallback)

A low-latency, offline voice-trigger / hotphrase runner designed for **NVIDIA Jetson Orin (JetPack 6 / CUDA 12.x)**, but also works on x86 Linux and CPU-only ARM boards.

## Features
- WebRTC VAD (20ms frames) → short speech segments for low latency
- `faster-whisper` (CTranslate2) **CUDA** on Jetson; **auto fallback** to CPU
- Fuzzy phrase matching → run shell commands
- UDP control + remote triggers:
  - `CTRL:PAUSE`, `CTRL:RESUME`
  - `TRIGGER:<phrase>` (must exist in triggers)
  - `CMD:<shell command>` (optional; can be disabled)
- Optional **model prefetch** and **warmup** to avoid a slow first trigger

## Quick start (Jetson Orin)

### 1) Clone + install
```bash
git clone <YOUR_REPO_URL>
cd jetson-voice-trigger
chmod +x scripts/install_orin.sh
./scripts/install_orin.sh
```

### 2) Self-test
```bash
source .venv/bin/activate
python self_test.py --model tiny.en
```

### 3) Run
```bash
source .venv/bin/activate
python -m voice_trigger.app --asr_device auto --model medium.en --compute float16 --lang en
```

## Triggers configuration

You can either edit defaults in `src/voice_trigger/config.py`, or provide an external JSON file:

**triggers.json**
```json
{
  "open browser": "xdg-open https://www.wikipedia.org",
  "say hello": "bash -lc 'notify-send \"Trigger\" \"Hello\"'"
}
```

Run with:
```bash
python -m voice_trigger.app --triggers triggers.json
```

## UDP control

Enable UDP listener:
```bash
python -m voice_trigger.app --udp_in --udp_host 0.0.0.0 --udp_port 9999 --udp_token SECRET
```

Messages (prepend `SECRET:` when token is used):
- `CTRL:PAUSE`
- `CTRL:RESUME`
- `TRIGGER:<phrase>`
- `CMD:<shell command>` (disabled by default; enable with `--allow_udp_cmd`)

## Performance tips (Jetson)
- For **medium/large**: `--compute float16` and `--asr_device cuda` (or `auto`)
- Use `--prefetch-model` to download weights ahead of time
- Use `--warmup-sec 1.0` to warm kernels/cache

Example:
```bash
python -m voice_trigger.app --asr_device auto --model medium.en --compute float16 --prefetch-model --warmup-sec 1.0
```

## Troubleshooting

### “CTranslate2 package was not compiled with CUDA support”
On Jetson/AArch64, you typically need to **build CTranslate2 from source with CUDA**:
```bash
./scripts/install_orin.sh
```

### No microphone / PortAudio errors
Install:
- `libportaudio2 portaudio19-dev libasound2-dev`

List devices:
```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

## Security notes
- UDP `CMD:` executes arbitrary shell commands → **disabled by default**
- If enabling it, always use `--udp_token`

## License
MIT
