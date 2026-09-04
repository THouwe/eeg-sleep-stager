"""Download the Sleep-EDF Cassette subset and build manifest.csv (S2).

The Sleep Cassette recordings are named ``SC4ssNXX-PSG.edf`` and
``SC4ssNXX-Hypnogram.edf`` where ``ss`` is the subject number, ``N`` the night
(1 or 2) and ``XX`` a two-character suffix (a fixed letter plus the scorer id
for hypnograms). The subject+night prefix ``SC4ssN`` (the first six characters)
is shared by a recording's PSG and Hypnogram, so it is used to pair them.

The dataset is open access — no PhysioNet login or token is required.

Pure helpers (`parse_recording_stem`, `discover_recordings`) do the filename
logic and are unit-tested; the network functions stream files to disk and skip
anything already present.
"""

from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import requests

from .config import Config

# A recording file stem looks like "SC4001E0" (PSG) / "SC4001EC" (Hypnogram):
# SC4 + 2-digit subject + 1-digit night + 2-char suffix.
_STEM_RE = re.compile(r"^SC4(?P<subject>\d{2})(?P<night>\d)[A-Za-z0-9]{2}$")

# Matches the file names in a PhysioNet directory listing.
_FILE_RE = re.compile(r"SC4\d{3}[A-Za-z0-9]{2}-(?:PSG|Hypnogram)\.edf")

_PSG_SUFFIX = "-PSG.edf"
_HYP_SUFFIX = "-Hypnogram.edf"

MANIFEST_COLUMNS = ["subject_id", "night", "psg_path", "hypnogram_path"]


@dataclass(frozen=True)
class Recording:
    """One subject-night: a PSG file paired with its Hypnogram."""

    subject_id: str          # e.g. "SC00" (subject 00)
    night: int               # 1 or 2
    psg_name: str            # e.g. "SC4001E0-PSG.edf"
    hypnogram_name: str      # e.g. "SC4001EC-Hypnogram.edf"

    @property
    def key(self) -> str:
        """subject+night prefix shared by both files, e.g. 'SC4001'."""
        return self.psg_name[:6]


def parse_recording_stem(stem: str) -> Optional[tuple[str, int]]:
    """Parse a 6+ char file stem into (subject_id, night), or None if it doesn't match.

    >>> parse_recording_stem("SC4001E0")
    ('SC00', 1)
    >>> parse_recording_stem("SC4451E0")
    ('SC45', 1)
    >>> parse_recording_stem("not-a-stem") is None
    True
    """
    m = _STEM_RE.match(stem)
    if m is None:
        return None
    return f"SC{m.group('subject')}", int(m.group("night"))


def discover_recordings(file_names: Iterable[str]) -> list[Recording]:
    """Pair PSG and Hypnogram files into Recordings, sorted by subject then night.

    Files whose PSG has no matching Hypnogram (or vice versa) are skipped.
    """
    psg: dict[str, str] = {}
    hyp: dict[str, str] = {}
    for name in file_names:
        if name.endswith(_PSG_SUFFIX):
            psg[name[:6]] = name
        elif name.endswith(_HYP_SUFFIX):
            hyp[name[:6]] = name

    recordings: list[Recording] = []
    for key in psg:
        if key not in hyp:
            continue
        parsed = parse_recording_stem(psg[key][:8])
        if parsed is None:
            continue
        subject_id, night = parsed
        recordings.append(
            Recording(
                subject_id=subject_id,
                night=night,
                psg_name=psg[key],
                hypnogram_name=hyp[key],
            )
        )
    recordings.sort(key=lambda r: (r.subject_id, r.night))
    return recordings


def limit_subjects(recordings: list[Recording], limit: Optional[int]) -> list[Recording]:
    """Keep only recordings from the first ``limit`` distinct subjects (all nights)."""
    if limit is None:
        return recordings
    keep: list[str] = []
    for r in recordings:
        if r.subject_id not in keep:
            if len(keep) >= limit:
                continue
            keep.append(r.subject_id)
    kept = set(keep)
    return [r for r in recordings if r.subject_id in kept]


def list_remote_files(base_url: str, session: Optional[requests.Session] = None) -> list[str]:
    """Fetch the PhysioNet directory index and return the .edf file names it lists."""
    sess = session or requests.Session()
    resp = sess.get(base_url, timeout=60)
    resp.raise_for_status()
    # De-duplicate while preserving order.
    seen: dict[str, None] = {}
    for name in _FILE_RE.findall(resp.text):
        seen.setdefault(name, None)
    return list(seen)


def download_file(
    url: str,
    dest: Path,
    session: Optional[requests.Session] = None,
    chunk_size: int = 1 << 20,
    max_retries: int = 5,
) -> Path:
    """Stream ``url`` to ``dest``. Skips the download if ``dest`` already exists.

    Downloads to a ``.part`` temp file and renames on success so an interrupted
    download never leaves a truncated file that looks complete.

    PhysioNet occasionally drops long-lived HTTPS connections mid-transfer
    (``IncompleteRead`` / ``ChunkedEncodingError``). To survive that we retry up
    to ``max_retries`` times and *resume* from the bytes already written to the
    ``.part`` file via an HTTP Range request, so a failure near the end of a
    large PSG file does not restart the download from zero.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    sess = session or requests.Session()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        # Resume from whatever we already have on disk.
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with sess.get(url, stream=True, timeout=120, headers=headers) as resp:
                # 206 = server honoured the Range and is resuming.
                # 200 = server ignored it (or we asked from 0): restart the file.
                if have and resp.status_code == 200:
                    have = 0
                resp.raise_for_status()
                mode = "ab" if have else "wb"
                with tmp.open(mode) as fh:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            fh.write(chunk)
            tmp.replace(dest)
            return dest
        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as err:
            last_err = err
            if attempt < max_retries:
                wait = min(30, 2 ** attempt)  # exponential backoff, capped
                got = tmp.stat().st_size if tmp.exists() else 0
                print(
                    f"    download interrupted ({type(err).__name__}); "
                    f"retry {attempt}/{max_retries - 1} in {wait}s "
                    f"(have {got} bytes) ..."
                )
                time.sleep(wait)

    raise RuntimeError(
        f"Failed to download {url} after {max_retries} attempts: {last_err}"
    )


def write_manifest(recordings: list[Recording], raw_dir: Path, manifest_path: Path) -> Path:
    """Write manifest.csv (one row per recording) with absolute file paths."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for r in recordings:
            writer.writerow(
                {
                    "subject_id": r.subject_id,
                    "night": r.night,
                    "psg_path": str((raw_dir / r.psg_name).resolve()),
                    "hypnogram_path": str((raw_dir / r.hypnogram_name).resolve()),
                }
            )
    return manifest_path


def ingest(cfg: Config, limit: Optional[int] = None) -> Path:
    """Download recordings, verify PSG+Hypnogram pairing, write manifest.csv.

    Args:
        cfg: Validated run configuration (paths + dataset source URL).
        limit: If given, only the first N distinct subjects are pulled.

    Returns:
        Path to the written manifest.csv.
    """
    base_url = cfg.dataset.source_url
    if not base_url.endswith("/"):
        base_url += "/"

    session = requests.Session()
    print(f"Listing recordings at {base_url} ...")
    file_names = list_remote_files(base_url, session=session)
    recordings = discover_recordings(file_names)
    if not recordings:
        raise RuntimeError(
            f"No paired PSG/Hypnogram recordings found at {base_url}. "
            "The listing may have changed or the URL is wrong."
        )

    recordings = limit_subjects(recordings, limit)
    n_subjects = len({r.subject_id for r in recordings})
    print(f"Selected {len(recordings)} recordings across {n_subjects} subjects.")

    raw_dir = cfg.paths.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(recordings, 1):
        for name in (r.psg_name, r.hypnogram_name):
            dest = raw_dir / name
            existed = dest.exists()
            download_file(base_url + name, dest, session=session)
            status = "have" if existed else "got "
            print(f"  [{i}/{len(recordings)}] {status} {name}")

    manifest_path = write_manifest(recordings, raw_dir, cfg.paths.manifest_path)
    print(f"Wrote manifest: {manifest_path} ({len(recordings)} rows)")
    return manifest_path
