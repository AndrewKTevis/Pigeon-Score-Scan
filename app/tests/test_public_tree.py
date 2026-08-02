from __future__ import annotations

from pathlib import Path

from app.tools import check_public_tree


def test_public_tree_accepts_clean_source(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Example\n", encoding="utf-8")
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "steps:\n  - uses: actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803\n",
        encoding="utf-8",
    )

    assert check_public_tree.audit(tmp_path) == []


def test_public_tree_rejects_private_and_generated_state(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "job.json").write_text("{}\n", encoding="utf-8")
    environment = tmp_path / "app" / ".venv"
    environment.mkdir(parents=True)
    (tmp_path / "launcher.exe").write_bytes(b"generated")
    fake_token = "gh" + "p_" + "12345678901234567890"
    separator = chr(92)
    windows_path = separator.join(("C:", "Users", "example", "Desktop"))
    escaped_windows_path = windows_path.replace(separator, separator * 2)
    wsl_path = (
        separator * 2
        + separator.join(("wsl.localhost", "Ubuntu", "home", "example", "model"))
    )
    (tmp_path / "notes.txt").write_text(
        (
            f"local={windows_path}\n"
            f'json="{escaped_windows_path}"\n'
            f"wsl={wsl_path}\n"
            f"secret={fake_token}\n"
        ),
        encoding="utf-8",
    )

    problems = check_public_tree.audit(tmp_path)

    assert any(problem.startswith("forbidden root:") for problem in problems)
    assert any("generated directory: app/.venv" in problem for problem in problems)
    assert any("generated/private file: launcher.exe" in problem for problem in problems)
    assert any("Windows user path: notes.txt" in problem for problem in problems)
    assert any("WSL user UNC path: notes.txt" in problem for problem in problems)
    assert any("GitHub token: notes.txt" in problem for problem in problems)


def test_public_tree_rejects_unpinned_actions(tmp_path: Path) -> None:
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("steps:\n  - uses: actions/checkout@v6\n", encoding="utf-8")

    assert check_public_tree.audit(tmp_path) == [
        "GitHub Action is not pinned to a commit: .github/workflows/ci.yml"
    ]
