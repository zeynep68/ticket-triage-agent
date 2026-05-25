"""Entry point for `uv run triage-ui` - spawns streamlit on the home page."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    home = Path(__file__).parent / "home.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(home)],
        check=False,
    )
