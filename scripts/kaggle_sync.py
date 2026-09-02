"""
kaggle_sync.py — thin wrapper around the Kaggle Python API for VoxGuard's
                 GPU notebook workflow.

ONE-TIME SETUP (do this before running any function in this file)
─────────────────────────────────────────────────────────────────
1. Log in to https://www.kaggle.com and go to:
       Account → Settings → API → "Create New Token"
   This downloads a file called kaggle.json containing your API credentials.

2. Place the file at:
       ~/.kaggle/kaggle.json          (macOS / Linux)
       C:\\Users\\<you>\\.kaggle\\kaggle.json   (Windows)

3. Restrict permissions so only your user can read it:
       chmod 600 ~/.kaggle/kaggle.json     (macOS / Linux)
       On Windows the file is not world-readable by default; no extra step needed.

4. Verify the setup by running:
       python scripts/kaggle_sync.py --check

See the project's "Local Hardware Profile & Free-Service Workflow" section
for the full one-time Kaggle account setup steps this script depends on.

──────────────────────────────────────────────────────────────────────────
Usage examples
──────────────────────────────────────────────────────────────────────────
  # Verify Kaggle auth token
  python scripts/kaggle_sync.py --check

  # Create a new private dataset from a local directory
  python scripts/kaggle_sync.py upload \\
      --local-dir data/processed/embeddings \\
      --slug my-username/voxguard-embeddings \\
      --title "VoxGuard embeddings"

  # Push a new version of an existing dataset
  python scripts/kaggle_sync.py upload \\
      --local-dir data/processed/embeddings \\
      --slug my-username/voxguard-embeddings \\
      --title "VoxGuard embeddings v2" \\
      --update

  # Download outputs from a completed Kaggle notebook run
  python scripts/kaggle_sync.py download \\
      --kernel my-username/voxguard-phase1-embeddings \\
      --output-dir models/phase1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Kaggle Datasets practical size ceiling is ~20 GB; warn at 19 GB.
KAGGLE_DATASET_SIZE_WARNING_GB: float = 19.0
_BYTES_PER_GB: int = 1024 ** 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_kaggle_api():
    """Import and return the kaggle.api singleton, raising clearly if unavailable."""
    try:
        import kaggle.api as api
    except ImportError:
        print(
            "ERROR: 'kaggle' package not found.\n"
            "       Run:  pip install 'kaggle>=1.6,<2'",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        api.authenticate()
    except Exception as exc:
        print(
            f"ERROR: Kaggle authentication failed: {exc}\n"
            "\n"
            "Make sure ~/.kaggle/kaggle.json (or %USERPROFILE%\\.kaggle\\kaggle.json\n"
            "on Windows) exists and contains valid API credentials.\n"
            "\n"
            "One-time setup:\n"
            "  1. Visit https://www.kaggle.com → Account → Settings → API\n"
            "     → 'Create New Token' to download kaggle.json.\n"
            "  2. Move it to ~/.kaggle/kaggle.json\n"
            "  3. chmod 600 ~/.kaggle/kaggle.json   (macOS / Linux)\n"
            "  4. python scripts/kaggle_sync.py --check",
            file=sys.stderr,
        )
        sys.exit(1)

    return api


def _get_dir_size_bytes(directory: Path) -> int:
    """Return the total size of all files under *directory* in bytes."""
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())


def _write_dataset_metadata(local_dir: Path, slug: str, title: str) -> Path:
    """Write a dataset-metadata.json into *local_dir* and return its path.

    The Kaggle API's ``datasets create`` command requires this file to be
    present in the upload directory.  If one already exists it is
    overwritten so the title stays in sync.
    """
    # slug is expected as "owner/dataset-name"; metadata only needs the id part.
    dataset_id = slug.split("/")[-1] if "/" in slug else slug

    metadata = {
        "title": title,
        "id": slug,
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_path = local_dir / "dataset-metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    return meta_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upload_dataset(
    local_dir: str,
    dataset_slug: str,
    title: str,
    update: bool = False,
) -> None:
    """Upload or version a Kaggle Dataset from a local directory.

    Parameters
    ----------
    local_dir:
        Path to the local directory whose contents will be uploaded.
    dataset_slug:
        Kaggle dataset identifier in ``"owner/dataset-name"`` format,
        e.g. ``"alice/voxguard-embeddings"``.
    title:
        Human-readable title for the dataset (or version message when
        *update* is True).
    update:
        If ``False`` (default), creates a brand-new private dataset via
        ``kaggle datasets create``.
        If ``True``, pushes a new version via ``kaggle datasets version``.

    Raises
    ------
    FileNotFoundError
        If *local_dir* does not exist.
    """
    api = _require_kaggle_api()

    dir_path = Path(local_dir).resolve()
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Local directory not found: {dir_path}")

    # Size warning — Kaggle Datasets have a practical ~20 GB ceiling.
    size_bytes = _get_dir_size_bytes(dir_path)
    size_gb = size_bytes / _BYTES_PER_GB
    if size_gb > KAGGLE_DATASET_SIZE_WARNING_GB:
        print(
            f"WARNING: '{dir_path}' is {size_gb:.1f} GB, which exceeds the "
            f"{KAGGLE_DATASET_SIZE_WARNING_GB:.0f} GB warning threshold.\n"
            "         Kaggle Datasets have a practical ceiling near 20 GB; "
            "the upload may be rejected or time out.",
            file=sys.stderr,
        )

    owner = dataset_slug.split("/")[0] if "/" in dataset_slug else None

    if not update:
        # Create a new dataset — requires dataset-metadata.json in the dir.
        meta_path = _write_dataset_metadata(dir_path, dataset_slug, title)
        print(f"Creating new Kaggle Dataset '{dataset_slug}' from {dir_path} …")
        try:
            api.dataset_create_new(
                folder=str(dir_path),
                public=False,
                quiet=False,
                convert_to_csv=False,
                dir_mode="zip",
            )
        finally:
            # Clean up the metadata file we injected so it does not litter
            # the source directory after the upload.
            if meta_path.exists():
                meta_path.unlink()
    else:
        # Push a new version of an existing dataset.
        meta_path = _write_dataset_metadata(dir_path, dataset_slug, title)
        print(
            f"Pushing new version of '{dataset_slug}' from {dir_path} …\n"
            f"Version message: {title!r}"
        )
        try:
            api.dataset_create_version(
                folder=str(dir_path),
                version_notes=title,
                quiet=False,
                convert_to_csv=False,
                delete_old_versions=False,
                dir_mode="zip",
            )
        finally:
            if meta_path.exists():
                meta_path.unlink()

    dataset_url = f"https://www.kaggle.com/datasets/{dataset_slug}"
    print(f"Done.  Dataset available at: {dataset_url}")


def download_output(kernel_slug: str, output_dir: str) -> None:
    """Download the output files of a completed Kaggle Notebook run.

    Uses ``kaggle kernels output`` to pull all output files (CSV, PT
    checkpoints, numpy arrays, etc.) produced during the notebook's last
    run into *output_dir*.  Typical use case: pull back embeddings or
    model checkpoints generated by a GPU notebook run on Kaggle.

    Parameters
    ----------
    kernel_slug:
        Kaggle kernel (notebook) identifier in ``"owner/kernel-name"`` format,
        e.g. ``"alice/voxguard-phase1-embeddings"``.
    output_dir:
        Local directory to write the downloaded files into.  Created
        automatically if it does not exist.

    Raises
    ------
    RuntimeError
        If the Kaggle API reports that the kernel has no output or the
        run has not yet completed.
    """
    api = _require_kaggle_api()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading outputs of kernel '{kernel_slug}' → {out_path} …")
    try:
        api.kernels_output(kernel_slug, path=str(out_path))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download kernel output for '{kernel_slug}': {exc}\n"
            "\n"
            "Common causes:\n"
            "  • The notebook run has not completed yet (check the notebook status\n"
            "    on https://www.kaggle.com).\n"
            "  • The kernel slug is wrong — expected 'owner/kernel-name' format.\n"
            "  • The notebook produced no output files in the /kaggle/working dir."
        ) from exc

    downloaded = list(out_path.iterdir())
    if downloaded:
        print(f"Downloaded {len(downloaded)} file(s) to {out_path}:")
        for f in sorted(downloaded):
            print(f"  {f.name}")
    else:
        print(
            f"WARNING: output directory {out_path} is empty — the kernel may "
            "have produced no output files.",
            file=sys.stderr,
        )


def check_auth() -> None:
    """Verify that the Kaggle API token is present and valid.

    Calls a lightweight, read-only Kaggle API endpoint (list competitions)
    and prints the authenticated username on success.  Safe to run at any
    time as a quick sanity check; does not upload or download anything.
    """
    api = _require_kaggle_api()

    # competitions_list is a cheap read-only call that validates the token
    # without touching any user-owned resources.
    try:
        api.competitions_list()
    except Exception as exc:
        print(f"ERROR: Kaggle token check failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Read the token file directly to surface the username — the API object
    # doesn't expose a "who am I?" method in all versions.
    token_path = Path.home() / ".kaggle" / "kaggle.json"
    username = "<unknown>"
    if token_path.exists():
        try:
            creds = json.loads(token_path.read_text())
            username = creds.get("username", username)
        except Exception:
            pass

    print(f"Kaggle authentication OK — logged in as: {username}")
    print(f"Token location: {token_path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaggle_sync.py",
        description="VoxGuard Kaggle dataset/notebook sync helper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Verify that the Kaggle API token (~/.kaggle/kaggle.json) is present "
            "and valid without uploading or downloading anything."
        ),
    )

    sub = parser.add_subparsers(dest="command")

    # ── upload subcommand ────────────────────────────────────────────────────
    upload_p = sub.add_parser(
        "upload",
        help="Upload a local directory as a new Kaggle Dataset or push a new version.",
    )
    upload_p.add_argument(
        "--local-dir",
        required=True,
        metavar="PATH",
        help="Local directory whose contents will be uploaded.",
    )
    upload_p.add_argument(
        "--slug",
        required=True,
        metavar="OWNER/NAME",
        help="Kaggle dataset slug, e.g. 'alice/voxguard-embeddings'.",
    )
    upload_p.add_argument(
        "--title",
        required=True,
        metavar="TITLE",
        help="Dataset title (new dataset) or version message (--update).",
    )
    upload_p.add_argument(
        "--update",
        action="store_true",
        default=False,
        help="Push a new version instead of creating a new dataset.",
    )

    # ── download subcommand ──────────────────────────────────────────────────
    download_p = sub.add_parser(
        "download",
        help="Download output files from a completed Kaggle Notebook run.",
    )
    download_p.add_argument(
        "--kernel",
        required=True,
        metavar="OWNER/KERNEL",
        help="Kaggle kernel slug, e.g. 'alice/voxguard-phase1-embeddings'.",
    )
    download_p.add_argument(
        "--output-dir",
        required=True,
        metavar="PATH",
        help="Local directory to write the downloaded files into.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.check:
        check_auth()
        return

    if args.command == "upload":
        upload_dataset(
            local_dir=args.local_dir,
            dataset_slug=args.slug,
            title=args.title,
            update=args.update,
        )
    elif args.command == "download":
        download_output(
            kernel_slug=args.kernel,
            output_dir=args.output_dir,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
