from pathlib import Path

import numpy as np
import polars as pl
from cp_measure.bulk import (
    get_core_measurements,
    get_correlation_measurements,
    # get_multimask_measurements,
)
from skimage.io import imread

MEASUREMENTS = get_core_measurements()
MEASUREMENTS_2 = get_correlation_measurements()


def read_labels(mask_path: Path):
    if str(mask_path).endswith("npz"):
        labels = np.load(mask_path)["arr_0"]
    else:
        labels = imread(mask_path)
    return labels


def apply_measurements(
    mask_path: Path, img_path: Path, object_name: str = None
) -> pl.DataFrame:
    gene, site, channel = img_path.stem.split("_")[:3]
    if object_name is None:
        object_name = mask_path.stem.split("_")[2]

    labels = read_labels(mask_path)

    img = imread(img_path)
    if img.ndim == 3:
        img = img.max(axis=0)
    # We know that the input data is int16
    img = (img / 65535).astype(np.float32)
    d = {}
    for meas_name, meas_f in MEASUREMENTS.items():
        if meas_name == "texture":
            # This has a specific parameter in the data
            measurements = meas_f(labels, img, scale=15)
        else:
            measurements = meas_f(labels, img)
        # Unpack output dictionaries
        for k, v in measurements.items():
            d[k] = v
            d["object"] = object_name
            d["gene"] = gene
            d["site"] = site
            d["channel"] = channel

    df = pl.from_dict(d)

    return df


def get_keys(fpath: Path, n: int = 2) -> tuple[str]:
    return tuple(fpath.stem.split("_")[:n])


def apply_measurements_2(
    mask_path: Path, pixels1_path: Path, pixels2_path: Path, object_name: str
) -> pl.DataFrame:
    labels = read_labels(mask_path)
    pixels1 = imread(pixels1_path)
    pixels2 = imread(pixels2_path)

    d = {}
    for meas_name, meas_f in MEASUREMENTS_2.items():
        measurements = meas_f(labels, pixels1, pixels2)
        # Unpack output dictionaries
        breakpoint()
        for k, v in measurements.items():
            d[k] = v
            d["object"] = object_name
            gene, site, channel1 = pixels1_path.stem.split("_")[:3]
            d["gene"] = gene
            d["site"] = site
            channel2 = pixels2_path.stem.split("_")[2]
            d["channel"] = (channel1, channel2)

    df = pl.from_dict(d)

    return df
