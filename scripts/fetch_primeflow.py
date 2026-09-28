"""Fetch files from the PrimeFlow VCC dataset repo into ``vcc_data/primeflow_data``.

The HF *blob* URL in a browser is an HTML page, not the file -- the download needs the
``resolve`` endpoint, which ``hf_hub_download`` uses for you. It also resumes partial
downloads and verifies the hash, both of which matter for a multi-GB h5ad.
"""

import argparse
import gzip
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

REPO = "altoslabs/primeflow-vcc-datasets"
DEST = Path(__file__).resolve().parents[1].parent / "vcc_data" / "primeflow_data"


def fetch(filename: str, repo: str, dest: Path, decompress: bool = True) -> Path:
    """Download one file from the dataset repo, optionally gunzipping it."""
    dest.mkdir(parents=True, exist_ok=True)

    # local_dir= puts the real file in dest rather than a symlink into
    # ~/.cache/huggingface, so the data survives a cache clear and is readable by
    # anndata directly.
    path = Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            local_dir=dest,
            token=os.environ.get("HF_TOKEN"),
        )
    )

    if not (decompress and path.suffix == ".gz"):
        return path

    out = path.with_suffix("")
    if out.exists():
        print(f"{out.name} already present, skipping decompress")
        return out

    print(f"decompressing -> {out.name}")
    with gzip.open(path, "rb") as fh_in, open(out, "wb") as fh_out:
        shutil.copyfileobj(fh_in, fh_out, length=32 * 1024 * 1024)

    return out


def main() -> None:
    """Parse arguments and fetch each requested file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        help="filenames within the repo, e.g. replogle22_k562_preprocessed.h5ad.gz",
    )
    parser.add_argument(
        "--repo", default=REPO, help=f"HF dataset repo (default: {REPO})"
    )
    parser.add_argument(
        "--dest", type=Path, default=DEST, help=f"output dir (default: {DEST})"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list what the repo contains and exit, without downloading",
    )
    parser.add_argument(
        "--no-decompress",
        action="store_true",
        help="keep .gz files as-is instead of gunzipping them",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")

    if args.list:
        for name in sorted(
            list_repo_files(args.repo, repo_type="dataset", token=token)
        ):
            print(name)
        return

    # Fail here rather than after a long download, so a typo costs nothing.
    if not args.files:
        parser.error("give at least one filename, or --list to see what is available")

    available = set(list_repo_files(args.repo, repo_type="dataset", token=token))
    missing = [f for f in args.files if f not in available]
    if missing:
        parser.error(
            f"not in {args.repo}: {', '.join(missing)}\nrun --list to see options"
        )

    for name in args.files:
        result = fetch(name, args.repo, args.dest, decompress=not args.no_decompress)
        print(f"{result}  ({result.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
