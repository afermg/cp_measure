"""Output format converters for :func:`cp_measure.featurizer.featurize`.

Each converter lazily imports its optional dependency and raises a helpful
:class:`ImportError` when the package is missing.
"""

from __future__ import annotations

import json

import numpy as np


def _lazy_import(module_name: str, extra: str):
    """Import *module_name* or raise with install instructions."""
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError:
        raise ImportError(
            f"{module_name} is required for return_as='{extra}'. "
            f"Install it with: pip install cp_measure[{extra}]"
        ) from None


def _unpack_rows(rows: list[tuple]) -> tuple[list, list, list]:
    """Extract image_id, object_type, label lists from row tuples."""
    if not rows:
        return [], [], []
    image_ids, object_types, labels = zip(*rows)
    return list(image_ids), list(object_types), list(labels)


def _to_pandas(*, data, columns, rows, **_kwargs):
    pd = _lazy_import("pandas", "pandas")

    df = pd.DataFrame(data, columns=columns)
    image_ids, object_types, labels = _unpack_rows(rows)
    df.insert(0, "image_id", image_ids)
    df.insert(1, "object_type", object_types)
    df.insert(2, "label", labels)
    return df


def _to_pyarrow(*, data, columns, rows, col_meta, config, channels, objects, is_3d, **_kwargs):
    pa = _lazy_import("pyarrow", "pyarrow")

    image_ids, object_types, labels = _unpack_rows(rows)
    arrays = {
        "image_id": image_ids,
        "object_type": object_types,
        "label": labels,
    }
    for i, col in enumerate(columns):
        arrays[col] = data[:, i]

    table = pa.table(arrays)

    fields = []
    for i, field in enumerate(table.schema):
        if i < 3:
            fields.append(field)
        else:
            meta = col_meta[i - 3]
            fields.append(
                field.with_metadata(
                    {k: str(v).encode() for k, v in meta.items() if v is not None}
                )
            )
    schema = pa.schema(
        fields,
        metadata={
            b"cp_measure_config": json.dumps(config).encode(),
            b"channels": json.dumps(channels).encode(),
            b"objects": json.dumps(objects).encode(),
            b"is_3d": json.dumps(is_3d).encode(),
        },
    )
    return table.cast(schema)


def _to_anndata(*, data, columns, rows, col_meta, config, channels, objects, is_3d, **_kwargs):
    ad = _lazy_import("anndata", "anndata")
    pd = _lazy_import("pandas", "anndata")

    obs = pd.DataFrame(rows, columns=["image_id", "object_type", "label"])
    if rows[0][0] is not None:
        obs.index = [f"{r[0]}_{r[1]}_{r[2]}" for r in rows]
    else:
        obs.index = [f"{r[1]}_{r[2]}" for r in rows]
    obs.index = obs.index.astype(str)

    var = pd.DataFrame(col_meta)
    var.index = columns

    uns = {
        "config": config,
        "channels": channels,
        "objects": objects,
        "is_3d": is_3d,
    }

    return ad.AnnData(X=data.astype(np.float32, copy=False), obs=obs, var=var, uns=uns)


_CONVERTERS = {
    "pandas": _to_pandas,
    "pyarrow": _to_pyarrow,
    "anndata": _to_anndata,
}


def convert(fmt: str, **kwargs):
    """Dispatch to the appropriate converter."""
    return _CONVERTERS[fmt](**kwargs)
