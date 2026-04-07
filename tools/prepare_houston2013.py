from __future__ import annotations

from pathlib import Path
from typing import Iterable
import sys
import urllib.error
import urllib.request

import numpy as np
from scipy.io import savemat, loadmat


VA_URLS: tuple[str, ...] = (
    "https://github.com/songyz2019/rs-fusion-datasets-dist/releases/download/v1.0.0/2013_IEEE_GRSS_DF_Contest_Samples_VA.txt",
    "https://pastebin.com/raw/FJyu5SQX",
)


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")


def _download_first_available(dest: Path, urls: Iterable[str]) -> None:
    opener = urllib.request.build_opener()
    opener.addheaders = [
        (
            "User-Agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
        )
    ]
    urllib.request.install_opener(opener)

    errors: list[str] = []
    for url in urls:
        try:
            print(f"Downloading validation labels from {url}")
            urllib.request.urlretrieve(url, dest)
            print(f"Saved validation labels to {dest}")
            return
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
            errors.append(f"{url}: {exc}")

    raise RuntimeError(
        "Unable to generate validation indices because validation label text "
        "could not be downloaded.\n"
        "Please manually place 2013_IEEE_GRSS_DF_Contest_Samples_VA.txt into "
        f"{dest.parent}\n"
        + "\n".join(errors)
    )


def _ensure_validation_txt(raw_dir: Path) -> Path:
    va_txt = raw_dir / "2013_IEEE_GRSS_DF_Contest_Samples_VA.txt"
    if va_txt.is_file():
        return va_txt

    print("Validation label text not found. Attempting to download public mirror...")
    _download_first_available(va_txt, VA_URLS)
    return va_txt


def _load_hsi_and_lidar(raw_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import rasterio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "rasterio is required to read the 144-band Houston2013 CASI TIFF. "
            "Please install rasterio, then rerun this script."
        ) from exc

    casi_path = raw_dir / "2013_IEEE_GRSS_DF_Contest_CASI.tif"
    lidar_path = raw_dir / "2013_IEEE_GRSS_DF_Contest_LiDAR.tif"
    _require_file(casi_path)
    _require_file(lidar_path)

    with rasterio.open(casi_path) as src:
        hsi = src.read().transpose(1, 2, 0)
    with rasterio.open(lidar_path) as src:
        lidar = src.read(1)

    if hsi.shape != (349, 1905, 144):
        raise ValueError(f"Unexpected HSI shape: {hsi.shape}")
    if lidar.shape != (349, 1905):
        raise ValueError(f"Unexpected LiDAR shape: {lidar.shape}")
    return hsi, lidar


def _parse_roi_txt(path: Path, shape: tuple[int, int]) -> np.ndarray:
    label_map = np.zeros(shape, dtype=np.uint8)
    class_id = 0
    block_lines: list[str] = []

    def flush_block() -> None:
        nonlocal class_id, block_lines, label_map
        if not block_lines:
            return
        class_id += 1
        for line in block_lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            x = int(parts[1])
            y = int(parts[2])
            row = y - 1
            col = x - 1
            if 0 <= row < shape[0] and 0 <= col < shape[1]:
                label_map[row, col] = class_id
        block_lines = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(";"):
                continue
            if line.startswith("1 ") and block_lines:
                flush_block()
            block_lines.append(line)
        flush_block()

    return label_map


def _coords_from_label_map(label_map: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(label_map)
    return np.stack([rows, cols], axis=1).astype(np.int16)


def _validate_outputs(base_dir: Path) -> None:
    hsi = loadmat(base_dir / "houston_hsi.mat")["houston_hsi"]
    lidar = loadmat(base_dir / "houston_lidar.mat")["houston_lidar"]
    gt = loadmat(base_dir / "houston_gt.mat")["houston_gt"]
    index = loadmat(base_dir / "houston_index.mat")

    train_idx = index["houston_train"]
    test_idx = index["houston_test"]
    all_idx = index["houston_all"]

    assert hsi.shape == (349, 1905, 144), hsi.shape
    assert lidar.shape == (349, 1905), lidar.shape
    assert gt.shape == (349, 1905), gt.shape
    assert train_idx.shape[1] == 2, train_idx.shape
    assert test_idx.shape[1] == 2, test_idx.shape
    assert all_idx.shape[1] == 2, all_idx.shape

    combined = np.concatenate([train_idx, test_idx], axis=0)
    combined = np.unique(combined, axis=0)
    all_unique = np.unique(all_idx, axis=0)
    if combined.shape != all_unique.shape or not np.array_equal(combined, all_unique):
        raise AssertionError("houston_all does not match union of train/test coordinates")


def main() -> int:
    base_dir = Path(__file__).resolve().parents[1] / "data" / "Houston2013"
    raw_dir = base_dir / "2013_DFTC"
    macos_dir = base_dir / "__MACOSX"

    if macos_dir.exists():
        print(f"Ignoring macOS metadata directory: {macos_dir}")

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Missing raw dataset directory: {raw_dir}")

    tr_txt = raw_dir / "2013_IEEE_GRSS_DF_Contest_Samples_TR.txt"
    _require_file(tr_txt)
    va_txt = _ensure_validation_txt(raw_dir)

    hsi, lidar = _load_hsi_and_lidar(raw_dir)
    train_gt = _parse_roi_txt(tr_txt, (349, 1905))
    test_gt = _parse_roi_txt(va_txt, (349, 1905))
    gt = train_gt.copy()
    gt[test_gt > 0] = test_gt[test_gt > 0]

    train_index = _coords_from_label_map(train_gt)
    test_index = _coords_from_label_map(test_gt)
    all_index = _coords_from_label_map(gt)

    savemat(base_dir / "houston_hsi.mat", {"houston_hsi": hsi})
    savemat(base_dir / "houston_lidar.mat", {"houston_lidar": lidar})
    savemat(base_dir / "houston_gt.mat", {"houston_gt": gt})
    savemat(
        base_dir / "houston_index.mat",
        {
            "houston_train": train_index,
            "houston_test": test_index,
            "houston_all": all_index,
        },
    )

    _validate_outputs(base_dir)

    print("Houston2013 conversion completed successfully.")
    print(f"Train samples: {train_index.shape[0]}")
    print(f"Test samples: {test_index.shape[0]}")
    print(f"All labeled samples: {all_index.shape[0]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - one-shot script
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
