"""Shared fixtures for featurizer tests."""

import numpy as np
import pytest

CELL_PAINTING_CHANNELS = ["DNA", "ER", "RNA", "AGP", "Mito"]

# All feature flags set to False — tests override only what they need.
ALL_OFF = dict(
    intensity=False,
    texture=False,
    granularity=False,
    radial_distribution=False,
    radial_zernikes=False,
    sizeshape=False,
    zernike=False,
    feret=False,
    correlation_pearson=False,
    correlation_costes=False,
    correlation_manders_fold=False,
    correlation_rwc=False,
)


def get_rng():
    """Return a fresh RNG with a fixed seed for reproducible test data.

    Each caller gets an independent generator so fixture output does not
    depend on execution order.
    """
    return np.random.default_rng(42)


SIZE_2D = 64
SIZE_3D = 32
DEPTH_3D = 8


def _stamp_objects_2d(mask_2d, n_objects=2):
    """Stamp non-overlapping square objects into a 2D spatial slice."""
    size = mask_2d.shape[-1]
    step = size // (n_objects + 1)
    obj_size = max(step // 2, 8)
    for i in range(n_objects):
        r = step * (i + 1) - obj_size // 2
        c = step * (i + 1) - obj_size // 2
        mask_2d[r : r + obj_size, c : c + obj_size] = i + 1


def _stamp_objects_3d(mask_3d, n_objects=2):
    """Stamp non-overlapping cubes into a 3D spatial volume."""
    size = mask_3d.shape[-1]
    depth = mask_3d.shape[0]
    step = size // (n_objects + 1)
    obj_size = max(step // 2, 8)
    z0, z1 = depth // 4, 3 * depth // 4
    for i in range(n_objects):
        r = step * (i + 1) - obj_size // 2
        c = step * (i + 1) - obj_size // 2
        mask_3d[z0:z1, r : r + obj_size, c : c + obj_size] = i + 1


# -- 2D fixtures -------------------------------------------------------------


@pytest.fixture()
def image_2d_1ch():
    """Random 2D image ``(1, H, W)``."""
    return get_rng().random((1, SIZE_2D, SIZE_2D))


@pytest.fixture()
def image_2d_2ch():
    """Random 2D image ``(2, H, W)``."""
    return get_rng().random((2, SIZE_2D, SIZE_2D))


@pytest.fixture()
def mask_2d():
    """Integer mask ``(1, H, W)`` with 2 objects."""
    mask = np.zeros((1, SIZE_2D, SIZE_2D), dtype=np.int32)
    _stamp_objects_2d(mask[0])
    return mask


@pytest.fixture()
def masks_2d_multi():
    """Two-type 2D masks ``(2, H, W)``: nuclei (2 labels) + cells (3 labels)."""
    nuclei = np.zeros((SIZE_2D, SIZE_2D), dtype=np.int32)
    nuclei[5:15, 5:15] = 1
    nuclei[30:40, 30:40] = 2
    cells = np.zeros((SIZE_2D, SIZE_2D), dtype=np.int32)
    cells[3:18, 3:18] = 1
    cells[28:45, 28:45] = 2
    cells[50:60, 50:60] = 3
    masks = np.stack([nuclei, cells], axis=0)
    assert masks[0].max() == 2 and masks[1].max() == 3  # sanity check
    return masks


# -- 3D fixtures -------------------------------------------------------------


@pytest.fixture()
def image_3d_1ch():
    """Random 3D volumetric image ``(1, Z, H, W)``."""
    return get_rng().random((1, DEPTH_3D, SIZE_3D, SIZE_3D))


@pytest.fixture()
def image_3d_2ch():
    """Random 3D volumetric image ``(2, Z, H, W)``."""
    return get_rng().random((2, DEPTH_3D, SIZE_3D, SIZE_3D))


@pytest.fixture()
def mask_3d():
    """Integer mask ``(1, Z, H, W)`` with 2 objects."""
    mask = np.zeros((1, DEPTH_3D, SIZE_3D, SIZE_3D), dtype=np.int32)
    _stamp_objects_3d(mask[0])
    return mask


@pytest.fixture()
def masks_3d_multi():
    """Two-type 3D masks ``(2, Z, H, W)``: nuclei (1 label) + cells (2 labels)."""
    nuclei = np.zeros((DEPTH_3D, SIZE_3D, SIZE_3D), dtype=np.int32)
    nuclei[2:6, 5:15, 5:15] = 1
    cells = np.zeros((DEPTH_3D, SIZE_3D, SIZE_3D), dtype=np.int32)
    cells[2:6, 3:18, 3:18] = 1
    cells[2:6, 20:28, 20:28] = 2
    masks = np.stack([nuclei, cells], axis=0)
    assert masks[0].max() == 1 and masks[1].max() == 2  # sanity check
    return masks
