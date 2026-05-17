# LocalLexis sidecar packaging

Build the FastAPI sidecar as a standalone binary:

    pip install -e ".[api,packaging]"
    pyinstaller packaging/locallexis-sidecar.spec --clean

Output: `dist/locallexis-sidecar` (Linux/macOS) or `dist/locallexis-sidecar.exe` (Windows).

Note: pyannote + torch are heavy. Expect 600–900 MB.
Run a smoke test:

    ./dist/locallexis-sidecar &
    curl http://127.0.0.1:<port>/health    # port comes from the JSON handshake
