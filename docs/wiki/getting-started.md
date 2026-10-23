# Getting Started & Quickstart Guide

This guide covers local environment setup, zero-dependency test verification, and executing the interactive demonstration harness.

---

## 🛠️ Prerequisites

* **Python 3.10+** (Tested on Python 3.10, 3.11, 3.12, 3.13, 3.14).
* **Zero External Dependencies:** Built entirely with standard library modules (`math`, `time`, `dataclasses`, `hashlib`, `unittest`). No `pip install` required for core execution.

---

## ⚡ 30-Second Quickstart

```bash
# 1. Clone the monorepo
git clone https://github.com/AAH20/apex-compute-energy-futures.git
cd apex-compute-energy-futures

# 2. Run the master demonstration harness (< 2 ms)
python3 demo.py
```

---

## 🧪 Running the Unit Test Suite

Execute the complete 14-test suite via `unittest`:

```bash
# Using Makefile
make test

# Or directly with Python
python3 -m unittest discover -s tests
```

Expected output:
```
..............
----------------------------------------------------------------------
Ran 14 tests in 0.001s

OK
```

---

## 🖥️ Command-Line Interface (CLI)

The repository provides a unified CLI via `apex_compute_futures/cli.py`:

```bash
# Display help and subcommands
python3 -m apex_compute_futures.cli --help

# Calculate forward compute term structure
python3 -m apex_compute_futures.cli forward-curve --spot 2.45 --region US_EAST_VA

# Simulate Synthetic Inertia AMM response to an ERCOT grid trip
python3 -m apex_compute_futures.cli inertia-amm --freq 59.80

# List available on-demand ecosystem integrations
python3 -m apex_compute_futures.cli integrations
```
