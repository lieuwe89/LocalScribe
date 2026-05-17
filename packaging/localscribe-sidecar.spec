# PyInstaller spec for the LocalScribe FastAPI sidecar.
# Build:   pyinstaller packaging/localscribe-sidecar.spec --clean
# Output:  dist/localscribe-sidecar (or .exe on Windows)
# The Tauri shell ships this binary alongside the app.
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

# pyannote.audio loads its model subpackages dynamically (importlib-based
# plugin discovery for the diarization pipeline). Only sweep the models tree
# — submodules of `pipelines` or `tasks` transitively import speechbrain and
# pytorch_lightning, which bloat the binary and make startup hang.
pyannote_audio_models = collect_submodules('pyannote.audio.models')

a = Analysis(
    ['../speechtotext/api/__main__.py'],
    pathex=[],
    # torchcodec ships native libtorchcodec_*.dylib + libtorchcodec_pybind_*.so
    # alongside its Python code; pyannote.audio.core.io imports AudioDecoder
    # from torchcodec and silently disables itself if the dylibs are missing.
    binaries=collect_dynamic_libs('torchcodec'),
    datas=(
        collect_data_files('pyannote.audio')    # config.yaml + ancillary
        + collect_data_files('faster_whisper')  # silero_vad onnx + assets
        + collect_data_files('torchcodec')      # version.py + metadata
    ),
    hiddenimports=[
        'speechtotext.asr.faster_whisper',
        'speechtotext.diarize.pyannote',
        'uvicorn.logging',
        *pyannote_audio_models,
        # Pipelines we use, named explicitly so we don't haul in every recipe:
        'pyannote.audio.pipelines.speaker_diarization',
        'pyannote.audio.pipelines.utils.hook',
        'torchcodec',
        'torchcodec.decoders',
    ],
    hookspath=[],
    excludes=['matplotlib', 'tkinter'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas,
    name='localscribe-sidecar',
    debug=False,
    strip=False,
    upx=False,
    console=True,
)
