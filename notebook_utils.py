"""Local notebook paths, downloads and PyTorch device selection."""

import os
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile


REPO_ROOT = Path(__file__).resolve().parent
DATASET_DIR = REPO_ROOT / "dataset"


def _link_dataset(assignment_dir, filename):
    """Keep the course's original paths pointing at the shared dataset."""
    target = DATASET_DIR / filename
    link = Path(assignment_dir) / "cs231n" / "datasets" / filename
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists() and not link.is_symlink():
        link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=target.is_dir())


def download_file(url, destination):
    """Download once; interrupted downloads never replace a cached file."""
    destination = Path(destination)
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    print(f"Downloading {url}\n  -> {destination}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        temporary.replace(destination)
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not download {url}. Check your network/proxy settings, "
            f"or download the file manually to {destination} and rerun this cell."
        ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def ensure_dataset(assignment_dir, name):
    """Fetch the course CIFAR-10 or COCO archive only when files are missing."""
    datasets_dir = DATASET_DIR
    datasets_dir.mkdir(parents=True, exist_ok=True)
    if name == "cifar10":
        directory = datasets_dir / "cifar-10-batches-py"
        required = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch", "batches.meta"]
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        archive_name = "cifar-10-python.tar.gz"
    elif name == "coco":
        directory = datasets_dir / "coco_captioning"
        required = [
            "coco2014_captions.h5", "coco2014_vocab.json",
            "train2014_vgg16_fc7_pca.h5", "val2014_vgg16_fc7_pca.h5",
            "train2014_urls.txt", "val2014_urls.txt",
        ]
        url = "https://cs231n.stanford.edu/coco_captioning.zip"
        archive_name = "coco_captioning.zip"
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if not all((directory / filename).is_file() for filename in required):
        # Reuse an archive downloaded with the previous per-assignment layout.
        legacy_archive = Path(assignment_dir) / "cs231n" / "datasets" / archive_name
        archive = (
            legacy_archive if legacy_archive.is_file()
            else download_file(url, datasets_dir / archive_name)
        )
        print(f"Extracting {archive.name} ...", flush=True)
        if name == "cifar10":
            with tarfile.open(archive) as source:
                source.extractall(datasets_dir, filter="data")
        else:
            with zipfile.ZipFile(archive) as source:
                for member in source.infolist():
                    if not (datasets_dir / member.filename).resolve().is_relative_to(datasets_dir.resolve()):
                        raise ValueError(f"Unsafe archive path: {member.filename}")
                source.extractall(datasets_dir)
        missing = [filename for filename in required if not (directory / filename).is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete {name} dataset at {directory}: {missing}")
    _link_dataset(assignment_dir, directory.name)
    return directory


def setup_assignment(number, datasets=()):
    """Select the correct cs231n package and make relative notebook paths local."""
    assignment_dir = REPO_ROOT / f"assignment{number}"
    # Submission ZIPs contain one assignment plus this helper at their top level.
    if not assignment_dir.is_dir() and (REPO_ROOT / "cs231n").is_dir():
        assignment_dir = REPO_ROOT
    if not (assignment_dir / "cs231n").is_dir():
        raise FileNotFoundError(f"Assignment directory not found: {assignment_dir}")
    for filename in ("cifar-10-batches-py", "imagenet_val_25.npz", "coco_captioning"):
        if (DATASET_DIR / filename).exists():
            _link_dataset(assignment_dir, filename)
    if number == 3 and (DATASET_DIR / "cifar-10-batches-py").is_dir():
        data_link = assignment_dir / "data" / "cifar-10-batches-py"
        data_link.parent.mkdir(parents=True, exist_ok=True)
        if not data_link.exists() and not data_link.is_symlink():
            data_link.symlink_to(
                os.path.relpath(DATASET_DIR / "cifar-10-batches-py", data_link.parent),
                target_is_directory=True,
            )
    loaded = sys.modules.get("cs231n")
    if loaded is not None and Path(loaded.__file__).resolve().parent != assignment_dir / "cs231n":
        raise RuntimeError(
            "Another assignment's cs231n package is already loaded. "
            "Restart the notebook kernel before switching assignments."
        )
    path = str(assignment_dir)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
    os.chdir(assignment_dir)
    print(f"Using local assignment directory: {assignment_dir}")
    for name in datasets:
        ensure_dataset(assignment_dir, name)
    return assignment_dir


def get_device():
    """Use CUDA when available, otherwise CPU; allow an explicit MPS override."""
    import torch

    requested = os.environ.get("CS231N_DEVICE", "auto").lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Set CS231N_DEVICE=cpu or install a CUDA PyTorch build.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this Python environment. Set CS231N_DEVICE=cpu.")
    return device
