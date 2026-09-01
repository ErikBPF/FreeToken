import hashlib
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "scripts/freeze-rocm-candidate.sh"


def run(*args, cwd: Path):
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)


def test_source_snapshot_is_deterministic_and_excludes_ignored_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    assert run("git", "init", "-q", cwd=repo).returncode == 0
    (repo / ".gitignore").write_text("ignored.txt\n")
    (repo / "tracked.txt").write_text("tracked\n")
    (repo / "untracked.txt").write_text("untracked\n")
    (repo / "ignored.txt").write_text("secret\n")
    assert run("git", "add", ".gitignore", "tracked.txt", cwd=repo).returncode == 0
    assert run(
        "git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "base", cwd=repo,
    ).returncode == 0

    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    for output in (first, second):
        result = run("bash", str(FREEZE), str(output), cwd=repo)
        assert result.returncode == 0, result.stderr

    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    with tarfile.open(first) as archive:
        names = {name.removeprefix("./") for name in archive.getnames()}
    assert {".freetoken-source.json", ".gitignore", "tracked.txt", "untracked.txt"} <= names
    assert "ignored.txt" not in names
    assert not any(name.startswith(".git/") for name in names)
