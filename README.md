# 🐾 Compatibillabuddy

**Hardware-aware dependency compatibility framework for Python ML stacks.**

[![PyPI version](https://img.shields.io/pypi/v/compatibillabuddy)](https://pypi.org/project/compatibillabuddy/)
[![Python](https://img.shields.io/pypi/pyversions/compatibillabuddy)](https://pypi.org/project/compatibillabuddy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

Compatibillabuddy diagnoses and repairs dependency conflicts in Python ML environments — especially the ones that install fine but crash at runtime because of GPU drivers, CUDA mismatches, or ABI breaks.

## The Problem

```
$ pip install torch numpy pandas scikit-learn
# ✅ Successfully installed ...

$ python -c "import torch; print(torch.cuda.is_available())"
# ❌ False — wrong CUDA version for your driver

$ python -c "import sklearn"
# ❌ ImportError: numpy ABI incompatibility
```

Traditional resolvers (`pip`, `uv`, `poetry`) solve **version constraints** — but ML environments fail because of **hardware mismatches**, **ABI breaks**, and **runtime incompatibilities** that metadata alone can't capture.

## What Compatibillabuddy Does

- 🔍 **Probes your hardware** — GPU, CUDA driver, CPU features, OS — without importing heavy ML packages
- 🩺 **Diagnoses your environment** — detects known-bad combinations from a curated knowledge base
- 💡 **Explains what's wrong** — human-readable reports with exact fix commands
- 🤖 **AI Agent** — Gemini-powered assistant that can diagnose and repair your environment interactively
- 🔒 **Standards-first** — emits `pylock.toml` (PEP 751), aligns with PEP 817 wheel variants

## Installation

```bash
pip install compatibillabuddy
```

With AI agent support:

```bash
pip install "compatibillabuddy[agent]"
```

## Quick Start

```bash
# Diagnose your current environment
compatibuddy doctor

# Get a detailed explanation of issues
compatibuddy explain

# Let the AI agent fix it (requires Gemini API key)
compatibuddy agent doctor
```

## Development

```bash
git clone https://github.com/jemsbhai/compatibillabuddy.git
cd compatibillabuddy
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).

## Author

**Muntaser Syed** — [muntaser@ieee.org](mailto:muntaser@ieee.org)
