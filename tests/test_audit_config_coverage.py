import subprocess
import sys
from pathlib import Path


def test_audit_config_coverage():
    root = Path(__file__).resolve().parents[1]
    script = root / "tools" / "audit_config_coverage.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(f"audit failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")

