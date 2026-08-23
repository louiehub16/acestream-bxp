import subprocess, sqlite3, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

def test_py_compile():
    for f in ["dispatcher/dispatch.py", "sidecar/worker_sidecar.py"]:
        assert subprocess.run([sys.executable, "-m", "py_compile", str(ROOT / f)]).returncode == 0

def test_worker_js():
    assert subprocess.run(["node", "--check", str(ROOT / "queue-worker/src/index.js")]).returncode == 0

def test_entrypoint():
    assert subprocess.run(["bash", "-n", str(ROOT / "docker/entrypoint.sh")]).returncode == 0

def test_yaml():
    import yaml
    with open(ROOT / ".github/workflows/docker-build.yml") as fh:
        assert yaml.safe_load(fh) is not None


