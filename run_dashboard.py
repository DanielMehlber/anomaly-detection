"""Launch the Streamlit dashboard from the project root."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    app = root / "src" / "dashboard" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app)],
        cwd=root,
        check=True,
    )


if __name__ == "__main__":
    main()
