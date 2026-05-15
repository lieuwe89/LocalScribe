import speechtotext

def test_package_importable():
    assert hasattr(speechtotext, "__version__")
    assert isinstance(speechtotext.__version__, str)
