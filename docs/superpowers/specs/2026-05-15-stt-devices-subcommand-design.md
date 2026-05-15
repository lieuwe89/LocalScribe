# `stt devices` Subcommand — Design

**Date:** 2026-05-15
**Status:** Approved (spec)
**Scope:** Single CLI subcommand. Extends the existing CLI from the MVP without touching pipeline, ASR, or diarization.
**Parent spec:** `docs/superpowers/specs/2026-05-14-speech-to-text-cli-design.md`

---

## 1. Purpose

Users who want to record meetings (Zoom/Teams/Meet) need to capture **system audio** (the other participants), not just the microphone. That requires a loopback or aggregate input device — but those devices have opaque names like `"BlackHole 2ch"`, `"VoiceMeeter Output"`, `"Monitor of Built-in Audio Analog Stereo"`. Users today must guess the right string to pass to `stt record --device`.

`stt devices` solves this: it lists every audio input the OS reports, with index, name, channel count, default-input marker, and (best-effort) loopback/monitor hint. The user picks one and feeds it to `stt record`.

Out of scope:
- Selecting / setting a default device persistently (use `~/.config/speechtotext/config.toml` for that — separate change).
- Creating aggregate devices programmatically (macOS Audio MIDI Setup or Linux `pactl` is the right tool).
- Output devices (we never write audio, only read).

---

## 2. CLI Surface

```
stt devices [--json] [--all]

  (default)        # list input devices in a table, human-readable
  --json           # emit one JSON object per device, for scripting
  --all            # include output devices and duplex devices too (rare; useful for debugging)
```

Examples:

```
$ stt devices
  idx  name                                  channels  default  hint
    0  MacBook Pro Microphone                       1   *        mic
    1  BlackHole 2ch                                2            loopback
    2  Aggregate (Mic + BlackHole)                  3            mic+loopback
    4  External USB Mic                             1            mic

$ stt devices --json | jq '.[] | select(.hint=="loopback")'
{"index": 1, "name": "BlackHole 2ch", "channels": 2, "default": false, "hint": "loopback"}
```

**Exit code:** `0` if at least one input device is enumerated; `1` if `sounddevice` reports zero inputs (almost always a permission / driver bug — emit a one-line hint pointing to `stt doctor`).

---

## 3. Output Schema (`--json`)

```json
[
  {
    "index": 0,
    "name": "MacBook Pro Microphone",
    "channels": 1,
    "sample_rate": 48000.0,
    "default": true,
    "hint": "mic"
  },
  {
    "index": 1,
    "name": "BlackHole 2ch",
    "channels": 2,
    "sample_rate": 48000.0,
    "default": false,
    "hint": "loopback"
  }
]
```

`hint` is one of `"mic"`, `"loopback"`, `"mic+loopback"`, `"unknown"`. Heuristic, not authoritative — see §5.

---

## 4. Module Layout

Only one file added, one modified:

```
speechtotext/
├── devices.py           # NEW — pure enumeration + classification, no Typer
└── cli.py               # MODIFIED — wire a `devices` subcommand calling devices.py
tests/
└── test_devices.py      # NEW — unit tests using mocked sounddevice
```

Keeping the enumeration out of `cli.py` (the way `pipeline.py` is kept out of `cli.py`) means a future UI can call `speechtotext.devices.list_inputs()` directly.

---

## 5. Interface

```python
# speechtotext/devices.py

from dataclasses import dataclass
from typing import Literal

Hint = Literal["mic", "loopback", "mic+loopback", "unknown"]

@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    channels: int                 # max input channels
    sample_rate: float            # default sample rate reported by host
    default: bool                 # True if this is the OS default input
    hint: Hint

def list_inputs(include_all: bool = False) -> list[AudioDevice]: ...
def classify(name: str) -> Hint: ...
```

`classify()` is the hint heuristic — case-insensitive substring match on the device name:

| Match | Hint |
|---|---|
| `"blackhole"`, `"loopback"`, `"voicemeeter"`, `"soundflower"`, `".monitor"`, `"monitor of"`, `"stereo mix"` | `loopback` |
| `"aggregate"` AND (any of the loopback strings appears OR channels > 2) | `mic+loopback` |
| `"mic"`, `"microphone"`, `"input"`, `"line in"`, `"airpods"`, `"headset"` | `mic` |
| else | `unknown` |

Heuristic is intentionally dumb — the table just helps users spot the right entry. The real source of truth is what they pick.

---

## 6. Implementation Sketch

```python
# speechtotext/devices.py
import sounddevice as sd

def list_inputs(include_all: bool = False) -> list[AudioDevice]:
    raw = sd.query_devices()
    default_idx = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
    out: list[AudioDevice] = []
    for i, d in enumerate(raw):
        if d["max_input_channels"] == 0 and not include_all:
            continue
        out.append(
            AudioDevice(
                index=i,
                name=d["name"],
                channels=d["max_input_channels"],
                sample_rate=float(d.get("default_samplerate", 0.0)),
                default=(i == default_idx),
                hint=classify(d["name"]),
            )
        )
    return out
```

CLI wiring in `cli.py`:

```python
@app.command()
def devices(
    json_output: Annotated[bool, typer.Option("--json")] = False,
    include_all: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    inputs = list_inputs(include_all=include_all)
    if not inputs:
        typer.echo("no audio inputs detected; run `stt doctor`", err=True)
        raise typer.Exit(code=1)
    if json_output:
        typer.echo(json.dumps([asdict(d) for d in inputs], indent=2))
        return
    # Rich table render
    ...
```

Rich rendering: one line per device, columns `idx`, `name`, `channels`, `default`, `hint`. Use `rich.table.Table` to match the existing CLI aesthetic.

---

## 7. Error Handling

| Failure | Handling |
|---|---|
| `sounddevice` import fails | Already handled — CLI itself wouldn't load. `stt doctor` flags. |
| `sd.query_devices()` raises (PortAudio init error) | Catch, print `"audio host init failed: <err>; run stt doctor"`, exit 2. |
| No inputs found | Exit 1 with hint pointing at `stt doctor`. |
| Hostapi-specific weirdness (Windows WDM-KS vs WASAPI duplicates) | Show all entries; `--all` reveals duplicates the user might want to disambiguate. |

---

## 8. Tests

```
tests/test_devices.py
```

Mock `sounddevice.query_devices` and `sounddevice.default.device`. No real audio host needed.

Cases:

1. `test_lists_only_input_devices_by_default` — fixture has 1 input + 1 output; result has only the input.
2. `test_include_all_returns_everything` — same fixture, `include_all=True` returns both.
3. `test_default_flag_set_on_default_device` — sd.default.device = 2; only device with index 2 has `default=True`.
4. `test_hint_classification` — parametrize `classify()` over known names:
   - `"MacBook Pro Microphone"` → `mic`
   - `"BlackHole 2ch"` → `loopback`
   - `"Monitor of Built-in Audio Analog Stereo"` → `loopback`
   - `"Aggregate (Mic + BlackHole)"` → `mic+loopback`
   - `"Some Weird DAC Thing"` → `unknown`
5. `test_cli_table_output` — invoke `app, ["devices"]`, assert each device name appears in stdout.
6. `test_cli_json_output` — invoke `app, ["devices", "--json"]`, parse stdout as JSON, assert schema.
7. `test_cli_exit_1_when_no_inputs` — mock empty input list, assert exit code 1 and stderr hint.

CLI tests use `typer.testing.CliRunner` with `mix_stderr=False` to inspect stderr separately.

---

## 9. Documentation Touch-Ups

Add to `README.md` under "Usage":

```markdown
### Recording meetings (system audio)

`stt record` captures the **default input** (your mic). To capture the other
side of an online meeting, you need a loopback device:

- **macOS:** install [BlackHole](https://existential.audio/blackhole/), then
  create an Aggregate Device in Audio MIDI Setup that combines your mic with
  BlackHole. Find its name with `stt devices`, then:
  `stt record --device "Aggregate (Mic + BlackHole)"`.
- **Linux (PulseAudio):** every output has a `.monitor` source. List with
  `stt devices` (look for `hint: loopback`) and pass the name.
- **Simplest:** use the meeting tool's built-in recorder and then
  `stt transcribe meeting.mp4`.
```

Also: the existing `stt record` subcommand becomes meaningfully more discoverable once `stt devices` is documented next to it.

---

## 10. What This Does NOT Change

- No new dependencies. `sounddevice` is already pulled in (Task 12).
- No change to `record_to_wav()` signature; it already accepts `device: str | int | None`.
- No change to config schema.
- No change to the sidecar JSON v1 schema.
- No change to the Pipeline interface.

It's purely additive.

---

## 11. Effort Estimate

- `devices.py` + `classify()`: ~40 LOC
- CLI wiring: ~20 LOC
- Tests: ~80 LOC
- README touch-up: ~15 lines

One implementation task. Follows the same TDD pattern as the original Tasks 1–17. Should land in a single commit.

---

## 12. Open Question for Implementation Phase

- Whether to surface `hostapi` in the table on Windows. The plan above hides it for simplicity; if dual WASAPI/WDM listings confuse Windows users in practice, expose `hostapi` as an extra column (and JSON field) without changing other behavior.
