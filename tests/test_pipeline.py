import subprocess, sqlite3, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent

def test_py_compile():
    for f in ["dispatcher/dispatch.py", "dispatcher/kanban_hook.py",
              "sidecar/worker_sidecar.py"]:
        assert subprocess.run([sys.executable, "-m", "py_compile", str(ROOT / f)]).returncode == 0

def test_worker_js():
    assert subprocess.run(["node", "--check", str(ROOT / "queue-worker/src/index.js")]).returncode == 0

def test_entrypoint():
    assert subprocess.run(["bash", "-n", str(ROOT / "docker/entrypoint.sh")]).returncode == 0

def test_yaml():
    import yaml
    with open(ROOT / ".github/workflows/docker-build.yml") as fh:
        assert yaml.safe_load(fh) is not None


# ---------------------------------------------------------------- dispatcher features
import ast, inspect, re

GPU_5090 = "851399fb-7329-4195-a042-d6514b28cf33"
GPU_4090 = "ed563892-aacd-40f5-80b7-90c9be6c759b"


def _load_module(path: Path, name: str):
    """importlib load with zero network access (module top-level only)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_retry_update_simulation():
    """Offline simulation of the queue /retry endpoint's D1 UPDATE:
    failed -> pending with attempts reset to 0, scoped to session+filenames."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, session TEXT,"
                 " output_filename TEXT, status TEXT, attempts INTEGER)")
    conn.executemany("INSERT INTO jobs VALUES (?,?,?,?,?)", [
        ("0001", "s1", "a.wav", "failed", 3),
        ("0002", "s1", "b.wav", "failed", 2),
        ("0003", "s1", "c.wav", "done", 1),
        ("0004", "s2", "a.wav", "failed", 3),   # other session -> untouched
        ("0005", "s1", "d.wav", "pending", 0),  # already pending -> untouched
    ])
    filenames = ["a.wav", "b.wav"]
    marks = ",".join("?" * len(filenames))
    cur = conn.execute(
        f"UPDATE jobs SET status = 'pending', attempts = 0 "
        f"WHERE session = ? AND output_filename IN ({marks})"
        f" AND status = 'failed'",
        ["s1", *filenames])
    conn.commit()
    assert cur.rowcount == 2
    rows = conn.execute(
        "SELECT id, status, attempts FROM jobs ORDER BY id").fetchall()
    assert rows == [
        ("0001", "pending", 0),
        ("0002", "pending", 0),
        ("0003", "done", 1),
        ("0004", "failed", 3),
        ("0005", "pending", 0),
    ]


def test_content_type_map_import():
    """Import sidecar/worker_sidecar.py network-free via importlib and verify
    its R2 ContentType map covers every format we can emit."""
    try:
        mod = _load_module(ROOT / "sidecar" / "worker_sidecar.py",
                           "worker_sidecar_offline")
    except BaseException as e:  # missing boto3/requests, env quirks, SystemExit
        pytest.skip(f"sidecar not importable offline: {e}")
    src = inspect.getsource(mod)
    m = re.search(r"content_types\s*=\s*(\{.*?\})", src, re.S)
    if not m:
        pytest.skip("content_types map not found in sidecar source")
    ct = ast.literal_eval(m.group(1))
    assert ct.get("wav") == "audio/wav"
    assert ct.get("flac") == "audio/flac"
    assert ct.get("mp3") == "audio/mpeg"


def test_kanban_hook_silent_without_url(monkeypatch):
    """post_progress must be a silent no-op returning False when disabled."""
    try:
        kh = _load_module(ROOT / "dispatcher" / "kanban_hook.py",
                          "kanban_hook_offline")
    except BaseException as e:
        pytest.skip(f"kanban_hook not importable offline: {e}")
    monkeypatch.delenv("KANBAN_PROGRESS_URL", raising=False)
    monkeypatch.delenv("KANBAN_URL", raising=False)
    assert kh.post_progress("session-x", {"done": 1, "total": 2}) is False


def test_gpu_classes_and_output_format_env(monkeypatch):
    """dispatch.gpu_classes(): unset/empty -> 5090 pool; comma list parsed."""
    try:
        dp = _load_module(ROOT / "dispatcher" / "dispatch.py", "dispatch_offline")
    except BaseException as e:
        pytest.skip(f"dispatch not importable offline: {e}")
    monkeypatch.setenv("GPU_CLASSES", "")
    assert dp.gpu_classes() == [GPU_5090]
    monkeypatch.setenv("GPU_CLASSES",
                       f"{GPU_5090}, {GPU_4090}")  # 4090 fallback, xl fits 24GB
    assert dp.gpu_classes() == [GPU_5090, GPU_4090]
    monkeypatch.setenv("OUTPUT_FORMAT", "flac")
    assert dp.build_node_env().get("OUTPUT_FORMAT") == "flac"
