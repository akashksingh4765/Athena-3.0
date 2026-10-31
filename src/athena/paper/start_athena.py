"""
Athena 3.0
One-Click Paper Trading Launcher V1

Double-clicking the desktop shortcut will:

1. Locate Athena 3.0
2. Load all five registered models
3. Require 5/5 models to be ready
4. Start Streamlit
5. Open the Athena dashboard in the browser

Safety:
    - Paper trading only.
    - No Shoonya connection.
    - No real orders.
"""

from __future__ import annotations

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path


# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DASHBOARD = (
    PROJECT_ROOT
    / "src"
    / "athena"
    / "paper"
    / "paper_trading_dashboard_v1.py"
)

MODEL_MANAGER = (
    PROJECT_ROOT
    / "src"
    / "athena"
    / "paper"
    / "model_manager_v1.py"
)

PYTHON = sys.executable

STREAMLIT_URL = "http://localhost:8501"


# ============================================================================
# CONSOLE HELPERS
# ============================================================================

def banner():

    print()
    print("=" * 72)
    print("ATHENA 3.0")
    print("PAPER TRADING STARTUP")
    print("=" * 72)
    print()


def fail(message):

    print()
    print("=" * 72)
    print("ATHENA STARTUP FAILED")
    print("=" * 72)
    print()
    print(message)
    print()
    print("Press ENTER to close this window.")
    input()

    raise SystemExit(1)


# ============================================================================
# CHECK FILES
# ============================================================================

def check_files():

    if not PROJECT_ROOT.exists():
        fail(
            f"Athena project root was not found:\n"
            f"{PROJECT_ROOT}"
        )

    if not MODEL_MANAGER.exists():
        fail(
            f"Model manager was not found:\n"
            f"{MODEL_MANAGER}"
        )

    if not DASHBOARD.exists():
        fail(
            f"Dashboard was not found:\n"
            f"{DASHBOARD}"
        )


# ============================================================================
# LOAD MODELS
# ============================================================================

def load_models():

    banner()

    print(
        f"Project:\n{PROJECT_ROOT}"
    )

    print()
    print("Loading Athena models...")
    print()

    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
            / "src"
        )
    )

    from athena.paper.model_manager_v1 import (
        load_all_models,
        summarize_results,
    )

    results = load_all_models()

    for name, result in results.items():

        if result.loaded:

            print(
                f"[READY ] "
                f"{name:12} "
                f"{result.load_time_seconds:7.2f}s"
            )

        else:

            print(
                f"[FAILED] "
                f"{name:12}"
            )

            print(
                f"         {result.error}"
            )

    summary = summarize_results(
        results
    )

    print()
    print("-" * 72)

    print(
        f"Models ready: "
        f"{summary['loaded_models']}/"
        f"{summary['total_models']}"
    )

    print("-" * 72)

    if not summary["all_loaded"]:

        fail(
            "Not all required models loaded successfully.\n"
            "Athena will NOT start the trading dashboard."
        )

    print()
    print("ALL 5 MODELS READY")
    print()


# ============================================================================
# START STREAMLIT
# ============================================================================

def start_dashboard():

    print(
        "Starting Athena dashboard..."
    )

    print()

    command = [
        PYTHON,
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD),
        "--server.headless=true",
        "--server.port=8501",
        "--browser.gatherUsageStats=false",
    ]

    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )

    print(
        f"Dashboard process started "
        f"(PID {process.pid})"
    )

    return process


# ============================================================================
# WAIT FOR STREAMLIT
# ============================================================================

def wait_for_dashboard(
    timeout_seconds=30,
):

    print(
        "Waiting for dashboard..."
    )

    start = time.time()

    while (
        time.time() - start
        < timeout_seconds
    ):

        try:

            import urllib.request

            with urllib.request.urlopen(
                STREAMLIT_URL,
                timeout=1,
            ):

                print(
                    "Dashboard is online."
                )

                return True

        except Exception:

            time.sleep(0.5)

    return False


# ============================================================================
# OPEN BROWSER
# ============================================================================

def open_browser():

    print(
        "Opening Athena dashboard..."
    )

    webbrowser.open(
        STREAMLIT_URL
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    check_files()

    load_models()

    process = start_dashboard()

    if wait_for_dashboard():

        open_browser()

        print()
        print("=" * 72)
        print("ATHENA 3.0 IS READY")
        print("=" * 72)
        print()
        print(
            "Paper trading dashboard:"
        )
        print(
            STREAMLIT_URL
        )
        print()
        print(
            "The dashboard process is running."
        )
        print(
            "You may close this launcher window."
        )
        print()

    else:

        try:
            process.terminate()
        except Exception:
            pass

        fail(
            "The Streamlit dashboard did not become "
            "available within 30 seconds."
        )


if __name__ == "__main__":
    main()