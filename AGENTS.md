# AGENTS.md

## Project

LocalLexis is a local-first speech-to-text project. It contains:

- A Python package and CLI named `speechtotext` / `stt`.
- A FastAPI sidecar under `speechtotext/api`.
- A Tauri 2 desktop app under `ui`, with a Vite + React frontend and Rust shell.

The product invariant is local-first privacy: audio, models, transcripts, and indexes stay on the user's machine. Avoid introducing network calls, telemetry, or cloud dependencies unless the user explicitly asks for them.

## Working Style

- Read the nearby implementation before editing. Prefer the patterns already used in the Python package, API routes, React components, and Tauri shell.
- Keep source changes scoped to the user's request. Do not clean up unrelated files or revert work you did not make.
- Treat transcript JSON files as canonical when working on library/index behavior; the SQLite/FTS index is rebuildable.
- For desktop work, remember the runtime boundary: frontend talks to the FastAPI sidecar over localhost REST/SSE, and the Tauri shell owns sidecar lifecycle.
- For UI work, keep the manuscript-themed LocalLexis interface consistent with the existing screens and design handoff.

## Common Commands

Run Python commands from the repository root:

```bash
pytest -m "not integration"
pytest -m integration
stt doctor
```

Run frontend commands from the `ui` package:

```bash
pnpm test
pnpm tsc --noEmit
pnpm build
pnpm tauri dev
```

Run the Tauri smoke tests from the repository root:

```bash
cargo test --manifest-path /Users/lieuwejongsma/SpeechToText/ui/src-tauri/Cargo.toml --release
```

Prefer the narrowest validation that proves the change. Broaden to the full fast Python suite, frontend tests, type-checking, or Tauri smoke tests when touching shared behavior or cross-boundary flows.

## Versioning And Pushes

Before any requested `git push`, bump the project version first, commit the bump, and then push. If the project uses tags for releases, create the tag before pushing and push with tags.

Current version sources include:

- `pyproject.toml`
- `ui/package.json`
- `ui/src-tauri/Cargo.toml`
- Tauri config/version files if present

Keep related version files aligned unless the user explicitly asks for a partial bump.

## Output Files

User-facing generated artifacts that are not source code or project configuration should be written to `~/Documents/Codex-output/`. Create that directory when needed.

## Notes

- The repo may contain user or generated changes before you start. Check status and do not overwrite them.
- Use `rg` / `rg --files` for searches.
- The integration tests can be slow because they may use real models; prefer fast tests unless model behavior is directly affected.
