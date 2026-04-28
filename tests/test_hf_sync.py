"""Conflict-prompt tests for utils.hf_sync.pull.

We do not exercise the live HF backend in unit tests — instead we stub
HfApi to feed a controlled list of remote files + hashes, and stub
`hf_hub_download` to return content from an in-memory fake remote. This
keeps the conflict-resolution logic under test without network or auth.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

# Project convention: tests adjust sys.path so `utils...` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import hf_sync  # noqa: E402
from utils.hf_sync import (  # noqa: E402
    CONFLICT_ALL_OVERWRITE,
    CONFLICT_ALL_SKIP,
    CONFLICT_OVERWRITE,
    CONFLICT_SKIP,
    pull,
)


# --- minimal stand-ins for HF data classes ---


@dataclass
class _LfsInfo:
    sha256: str


@dataclass
class _RepoFile:
    path: str
    blob_id: Optional[str] = None
    lfs: Optional[_LfsInfo] = None


def _git_blob_sha1(content: bytes) -> str:
    h = hashlib.sha1()
    h.update(f"blob {len(content)}\0".encode())
    h.update(content)
    return h.hexdigest()


class _StubApi:
    """Stands in for `huggingface_hub.HfApi`. Driven by an in-memory remote dict."""

    def __init__(self, remote: dict[str, bytes]):
        self.remote = remote

    def list_repo_files(self, repo_id, repo_type):
        return list(self.remote.keys())

    def get_paths_info(self, repo_id, paths, repo_type, expand=True):
        return [
            _RepoFile(path=p, blob_id=_git_blob_sha1(self.remote[p]))
            for p in paths
            if p in self.remote
        ]


@pytest.fixture
def fake_remote(tmp_path):
    """A tiny remote with three files under prefix `prefix/`."""
    return {
        "prefix/a.txt": b"alpha-remote",
        "prefix/sub/b.txt": b"bravo-remote",
        "prefix/sub/c.txt": b"charlie-remote",
    }


def _stub_hf_hub_download(remote):
    """Build a stand-in for `hf_hub_download` that writes from `remote`."""

    def _impl(repo_id, filename, repo_type, local_dir):
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(remote[filename])
        return str(path)

    return _impl


def test_pull_downloads_missing_files(tmp_path, fake_remote):
    api = _StubApi(fake_remote)
    with patch.object(hf_sync, "hf_hub_download", _stub_hf_hub_download(fake_remote)):
        stats = pull(
            local_dir=tmp_path, repo_id="x/y", remote_prefix="prefix", api=api,
        )
    assert stats.downloaded == 3
    assert stats.skipped_match == 0
    assert (tmp_path / "a.txt").read_bytes() == b"alpha-remote"
    assert (tmp_path / "sub/b.txt").read_bytes() == b"bravo-remote"


def test_pull_skips_matching_local(tmp_path, fake_remote):
    # Pre-populate local with the same content as the remote.
    (tmp_path / "a.txt").write_bytes(b"alpha-remote")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/b.txt").write_bytes(b"bravo-remote")
    api = _StubApi(fake_remote)
    with patch.object(hf_sync, "hf_hub_download", _stub_hf_hub_download(fake_remote)):
        stats = pull(
            local_dir=tmp_path, repo_id="x/y", remote_prefix="prefix", api=api,
        )
    # Two matches; one missing -> downloaded.
    assert stats.skipped_match == 2
    assert stats.downloaded == 1
    assert stats.overwritten == 0


def test_pull_conflict_skip(tmp_path, fake_remote):
    """Local file differs from remote, user chooses skip — local is kept."""
    (tmp_path / "a.txt").write_bytes(b"alpha-LOCAL-EDIT")
    api = _StubApi(fake_remote)
    answers = iter([CONFLICT_SKIP, CONFLICT_SKIP, CONFLICT_SKIP])
    with patch.object(hf_sync, "hf_hub_download", _stub_hf_hub_download(fake_remote)):
        stats = pull(
            local_dir=tmp_path, repo_id="x/y", remote_prefix="prefix", api=api,
            on_conflict=lambda _p: next(answers),
        )
    assert stats.skipped_conflict == 1
    assert stats.downloaded == 2  # b and c had no local copy
    assert (tmp_path / "a.txt").read_bytes() == b"alpha-LOCAL-EDIT"


def test_pull_conflict_overwrite(tmp_path, fake_remote):
    """Local file differs, user chooses overwrite — local replaced."""
    (tmp_path / "a.txt").write_bytes(b"alpha-LOCAL-EDIT")
    api = _StubApi(fake_remote)
    answers = iter([CONFLICT_OVERWRITE])
    with patch.object(hf_sync, "hf_hub_download", _stub_hf_hub_download(fake_remote)):
        stats = pull(
            local_dir=tmp_path, repo_id="x/y", remote_prefix="prefix", api=api,
            on_conflict=lambda _p: next(answers),
        )
    assert stats.overwritten == 1
    assert (tmp_path / "a.txt").read_bytes() == b"alpha-remote"


def test_pull_all_overwrite_is_sticky(tmp_path, fake_remote):
    """First answer 'A'll-overwrite' applies to remaining conflicts without re-prompt."""
    (tmp_path / "a.txt").write_bytes(b"alpha-LOCAL")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/b.txt").write_bytes(b"bravo-LOCAL")
    (tmp_path / "sub/c.txt").write_bytes(b"charlie-LOCAL")
    api = _StubApi(fake_remote)

    prompt_calls = []

    def _prompt(rel):
        prompt_calls.append(rel)
        return CONFLICT_ALL_OVERWRITE

    with patch.object(hf_sync, "hf_hub_download", _stub_hf_hub_download(fake_remote)):
        stats = pull(
            local_dir=tmp_path, repo_id="x/y", remote_prefix="prefix", api=api,
            on_conflict=_prompt,
        )
    assert stats.overwritten == 3
    # Prompt should have been called exactly once — the rest are sticky.
    assert len(prompt_calls) == 1
    assert (tmp_path / "a.txt").read_bytes() == b"alpha-remote"
    assert (tmp_path / "sub/b.txt").read_bytes() == b"bravo-remote"
    assert (tmp_path / "sub/c.txt").read_bytes() == b"charlie-remote"


def test_pull_all_skip_is_sticky(tmp_path, fake_remote):
    (tmp_path / "a.txt").write_bytes(b"alpha-LOCAL")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/b.txt").write_bytes(b"bravo-LOCAL")
    (tmp_path / "sub/c.txt").write_bytes(b"charlie-LOCAL")
    api = _StubApi(fake_remote)

    prompt_calls = []

    def _prompt(rel):
        prompt_calls.append(rel)
        return CONFLICT_ALL_SKIP

    with patch.object(hf_sync, "hf_hub_download", _stub_hf_hub_download(fake_remote)):
        stats = pull(
            local_dir=tmp_path, repo_id="x/y", remote_prefix="prefix", api=api,
            on_conflict=_prompt,
        )
    assert stats.skipped_conflict == 3
    assert len(prompt_calls) == 1
    assert (tmp_path / "a.txt").read_bytes() == b"alpha-LOCAL"
    assert (tmp_path / "sub/b.txt").read_bytes() == b"bravo-LOCAL"
    assert (tmp_path / "sub/c.txt").read_bytes() == b"charlie-LOCAL"
