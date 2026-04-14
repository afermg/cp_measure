"""
Compare cp_measure against CellProfiler using benchmark images and masks.

Usage
-----
    python benchmark.py [--limit N] [--output-dir DIR]

Reads images/masks from /datastore/alan/cp_measure/benchmark_upload/,
extracts archives if needed, runs cp_measure featurize() on all (gene, site)
pairs, and produces:

  - cpm_cellprofiler_joint.parquet   — per-cell median values side-by-side
  - benchmark_summary.parquet        — per-feature Pearson r and R²
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from skimage.segmentation import relabel_sequential

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BENCH_DIR = Path("/datastore/alan/cp_measure/benchmark_upload")
IMAGES_TAR = BENCH_DIR / "images.tar.gz"
MASKS_TAR = BENCH_DIR / "masks.tar.gz"
IMAGES_DIR = BENCH_DIR / "images"
MASKS_DIR = BENCH_DIR / "masks"
CP_PARQUET = BENCH_DIR / "cellprofiler_data.parquet"

CHANNELS = ["DNA", "AGP", "ER", "Mito", "RNA"]
OBJECTS = ["Nuclei", "Cells"]  # order must match mask order: [DNA mask, AGP mask]
MASK_CHANNELS = {"Nuclei": "DNA", "Cells": "AGP"}


# ---------------------------------------------------------------------------
# Archive helpers
# ---------------------------------------------------------------------------


def extract_if_missing(archive: Path, dest: Path) -> None:
    if dest.exists():
        return
    print(f"Extracting {archive.name} → {dest.parent} ...", flush=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest.parent)
    print("  Done.", flush=True)


# ---------------------------------------------------------------------------
# File index
# ---------------------------------------------------------------------------


def build_image_index(images_dir: Path) -> dict[tuple[str, str, str], Path]:
    """Map (gene, site, channel) → filepath for all .tif files."""
    index: dict[tuple[str, str, str], Path] = {}
    for p in images_dir.glob("*.tif"):
        parts = p.stem.split("__")[0].split("_")  # <gene>_<site>_<channel>
        if len(parts) >= 3:
            gene, site, channel = parts[0], parts[1], parts[2]
            index[(gene, site, channel)] = p
    return index


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_image(path: Path) -> np.ndarray:
    """Load a TIFF image and normalise to [0, 1] float32."""
    import tifffile

    img = tifffile.imread(str(path))
    img = img.astype(np.float32)
    if img.max() > 0:
        dtype_max = np.iinfo(np.uint16).max if img.max() > 1 else 1.0
        img = img / dtype_max
    return img


def load_mask(path: Path) -> np.ndarray:
    """Load a mask .npz file and return a contiguous-label integer array."""
    data = np.load(path)
    mask = data[list(data.keys())[0]]
    mask, _, _ = relabel_sequential(mask)
    return mask.astype(np.int32)


# ---------------------------------------------------------------------------
# Feature name parsing
# ---------------------------------------------------------------------------


def parse_column(col: str) -> tuple[str, str] | None:
    """
    Parse a featurize() column name into (cpm_id, channel).

    Returns None for correlation columns between two channels
    (e.g. "Correlation_Pearson__DNA__AGP") since these don't have
    a direct 1:1 match in cellprofiler_data.parquet.
    """
    parts = col.split("__")
    if len(parts) == 1:
        return col, ""  # shape feature, no channel
    if len(parts) == 2:
        return parts[0], parts[1]  # e.g. "Intensity_MeanIntensity", "DNA"
    # 3 parts → pairwise correlation → skip
    return None


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------


def run_featurize(
    gene: str,
    site: str,
    image_index: dict,
    masks_dir: Path,
    config: dict,
) -> pa.Table | None:
    """
    Run featurize() for one (gene, site) pair.

    Returns a long-format Table with columns:
        gene, site, object, cpm_id, channel, cp_measure
    or None if any required files are missing.
    """
    from cp_measure.featurizer import featurize

    # Load 5-channel image
    channel_images = []
    for ch in CHANNELS:
        key = (gene, site, ch)
        if key not in image_index:
            return None
        channel_images.append(load_image(image_index[key]))
    image = np.stack(channel_images)  # (5, H, W)

    # Load masks: Nuclei from DNA, Cells from AGP
    masks_list = []
    for obj in OBJECTS:
        mask_ch = MASK_CHANNELS[obj]
        mask_path = masks_dir / f"{gene}_{site}_{mask_ch}.npz"
        if not mask_path.exists():
            return None
        masks_list.append(load_mask(mask_path))
    masks = np.stack(masks_list)  # (2, H, W)

    data, columns, rows = featurize(image, masks, config, image_id=f"{gene}_{site}")

    objects_arr = np.array([r[1] for r in rows])
    unique_objects = np.unique(objects_arr)

    # Parse columns and compute median per object in long format
    result: dict[str, list] = {
        "gene": [], "site": [], "object": [], "cpm_id": [], "channel": [], "cp_measure": [],
    }
    for col_idx, col in enumerate(columns):
        parsed = parse_column(col)
        if parsed is None:
            continue
        cpm_id, channel = parsed
        for obj_name in unique_objects:
            values = data[objects_arr == obj_name, col_idx]
            values = values[~np.isnan(values)]
            if len(values) == 0:
                continue
            result["gene"].append(gene)
            result["site"].append(site)
            result["object"].append(obj_name)
            result["cpm_id"].append(cpm_id)
            result["channel"].append(channel)
            result["cp_measure"].append(float(np.median(values)))

    if not result["gene"]:
        return None
    return pa.table(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


ALL_FEATURES = {
    "intensity",
    "texture",
    "granularity",
    "sizeshape",
    "zernike",
    "feret",
    "radial_distribution",
    "radial_zernikes",
    "correlation_pearson",
    "correlation_costes",
    "correlation_manders_fold",
    "correlation_rwc",
}


def build_config(features: set[str]) -> dict:
    from cp_measure.featurizer import make_featurizer_config

    return make_featurizer_config(
        CHANNELS,
        objects=OBJECTS,
        intensity="intensity" in features,
        texture="texture" in features,
        texture_params={"scale": 15},
        granularity="granularity" in features,
        granularity_params={"granular_spectrum_length": 16},
        sizeshape="sizeshape" in features,
        zernike="zernike" in features,
        feret="feret" in features,
        radial_distribution="radial_distribution" in features,
        radial_zernikes="radial_zernikes" in features,
        correlation_pearson="correlation_pearson" in features,
        correlation_costes="correlation_costes" in features,
        correlation_manders_fold="correlation_manders_fold" in features,
        correlation_rwc="correlation_rwc" in features,
    )


def main(limit: int | None, output_dir: Path, features: set[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: extract archives
    extract_if_missing(IMAGES_TAR, IMAGES_DIR)
    extract_if_missing(MASKS_TAR, MASKS_DIR)

    # Step 2: load CellProfiler ground truth
    print("Loading CellProfiler data ...", flush=True)
    cp_data = pq.read_table(CP_PARQUET)
    key_cols = ["gene", "site", "object", "channel", "cpm_id"]
    cp_consensus = cp_data.group_by(key_cols).aggregate(
        [("CellProfiler", "approximate_median")]
    )
    cp_consensus = cp_consensus.rename_columns(
        [c.replace("_approximate_median", "") for c in cp_consensus.column_names]
    )

    # Step 3: build image index
    print("Building image index ...", flush=True)
    image_index = build_image_index(IMAGES_DIR)

    # Unique (gene, site) pairs from CP data
    pairs_table = (
        cp_data.select(["gene", "site"])
        .group_by(["gene", "site"]).aggregate([])
        .sort_by([("gene", "ascending"), ("site", "ascending")])
    )
    pairs = list(zip(
        pairs_table.column("gene").to_pylist(),
        pairs_table.column("site").to_pylist(),
    ))
    if limit is not None:
        pairs = pairs[:limit]
    print(f"Processing {len(pairs)} (gene, site) pairs ...", flush=True)
    print(f"Features enabled: {sorted(features)}", flush=True)

    # Step 4: configure featurizer (texture scale=15 to match CellProfiler)
    config = build_config(features)

    # Step 5: run featurize for each pair
    all_frames: list[pa.Table] = []
    n_ok = 0
    n_fail = 0
    for i, (gene, site) in enumerate(pairs):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i + 1}/{len(pairs)}] {gene}_{site}", flush=True)
        result = run_featurize(gene, site, image_index, MASKS_DIR, config)
        if result is None:
            n_fail += 1
        else:
            all_frames.append(result)
            n_ok += 1

    print(f"  Completed: {n_ok} ok, {n_fail} missing files", flush=True)

    if not all_frames:
        print("ERROR: no data produced", file=sys.stderr)
        sys.exit(1)

    cpm_long = pa.concat_tables(all_frames)

    # Step 6: join with CellProfiler data
    print("Joining with CellProfiler data ...", flush=True)
    joint = cpm_long.join(cp_consensus, keys=key_cols, join_type="inner")

    joint_path = output_dir / "cpm_cellprofiler_joint.parquet"
    pq.write_table(joint, joint_path)
    print(f"Joint table saved: {joint_path} ({len(joint):,} rows)", flush=True)

    # Step 7: summary statistics
    print("Computing summary statistics ...", flush=True)
    valid = pc.and_(
        pc.and_(
            pc.invert(pc.is_nan(joint.column("cp_measure"))),
            pc.invert(pc.is_nan(joint.column("CellProfiler"))),
        ),
        pc.and_(
            pc.is_finite(joint.column("cp_measure")),
            pc.is_finite(joint.column("CellProfiler")),
        ),
    )
    filtered = joint.filter(valid)

    cpm_ids = sorted(pc.unique(filtered.column("cpm_id")).to_pylist())
    summary_data: dict[str, list] = {"cpm_id": [], "pearson_r": [], "n": [], "r2": []}
    for cpm_id in cpm_ids:
        group = filtered.filter(pc.equal(filtered.column("cpm_id"), cpm_id))
        cp_vals = group.column("CellProfiler").to_numpy()
        cpm_vals = group.column("cp_measure").to_numpy()
        n = len(cp_vals)
        r = float(np.corrcoef(cp_vals, cpm_vals)[0, 1]) if n > 1 else float("nan")
        summary_data["cpm_id"].append(cpm_id)
        summary_data["pearson_r"].append(r)
        summary_data["n"].append(n)
        summary_data["r2"].append(r**2)

    summary = pa.table(summary_data)
    summary_path = output_dir / "benchmark_summary.parquet"
    pq.write_table(summary, summary_path)
    print(f"Summary table saved: {summary_path}", flush=True)

    # Print summary
    print("\n=== Feature similarity summary ===")
    sorted_summary = summary.sort_by([("r2", "descending")])
    print(f"  {'cpm_id':40s}  {'pearson_r':>10s}  {'r2':>10s}  {'n':>6s}")
    for i in range(len(sorted_summary)):
        cid = sorted_summary.column("cpm_id")[i].as_py()
        pr = sorted_summary.column("pearson_r")[i].as_py()
        r2 = sorted_summary.column("r2")[i].as_py()
        n = sorted_summary.column("n")[i].as_py()
        print(f"  {cid:40s}  {pr:10.4f}  {r2:10.4f}  {n:6d}")

    # High-level stats
    r2_arr = summary.column("r2").to_numpy()
    r2_vals = r2_arr[~np.isnan(r2_arr)]
    print(f"\nFeatures matched: {len(summary)}")
    print(f"Median R²: {float(np.median(r2_vals)):.4f}")
    print(f"Features with R² > 0.95: {int((r2_vals > 0.95).sum())}/{len(r2_vals)}")
    print(f"Features with R² > 0.90: {int((r2_vals > 0.90).sum())}/{len(r2_vals)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N (gene, site) pairs (for quick testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory to write output parquet files (default: benchmarks/)",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=sorted(ALL_FEATURES),
        choices=sorted(ALL_FEATURES),
        metavar="FEATURE",
        help=(
            f"Feature groups to compute. Choices: {sorted(ALL_FEATURES)}. "
            "Defaults to all features."
        ),
    )
    args = parser.parse_args()
    main(limit=args.limit, output_dir=args.output_dir, features=set(args.features))
