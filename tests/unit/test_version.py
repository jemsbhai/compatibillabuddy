"""Verify package is importable and version is set."""


def test_package_importable():
    """The package should be importable after install."""
    import compatibillabuddy

    assert hasattr(compatibillabuddy, "__version__")


def test_version_format():
    """Version string should be valid semver (MAJOR.MINOR.PATCH)."""
    from compatibillabuddy import __version__

    parts = __version__.split(".")
    assert len(parts) == 3, f"Expected 3 version parts, got {len(parts)}: {__version__}"
    assert all(p.isdigit() for p in parts), f"Non-numeric version parts in: {__version__}"


def test_version_matches_pyproject():
    """Version in __init__ should match pyproject.toml."""
    from compatibillabuddy import __version__

    assert __version__ == "0.1.0"
