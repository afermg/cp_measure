"""Regenerate the float32 granularity fixture in CellProfiler 4.2.8."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np


def main():
    import cellprofiler_core.preferences

    cellprofiler_core.preferences.set_headless()
    from cellprofiler.modules.measuregranularity import MeasureGranularity
    from cellprofiler_core.image import Image, ImageSetList
    from cellprofiler_core.measurement import Measurements
    from cellprofiler_core.object import Objects, ObjectSet
    from cellprofiler_core.pipeline import Pipeline
    from cellprofiler_core.workspace import Workspace

    y, x = np.indices((64, 80))
    raw = ((73 * y + 157 * x + ((13 * x + 29 * y) % 31) * 101) % 40000 + 1000).astype(
        np.uint16
    )
    pixels = raw.astype(np.float32) / np.float32(65535)
    labels = np.zeros(raw.shape, np.int32)
    labels[8:28, 8:28] = 1
    labels[32:58, 37:68] = 2
    cases = []
    for sub, background in ((0.25, 0.25), (1.0, 0.5), (0.5, 1.0)):
        module = MeasureGranularity()
        module.set_module_num(1)
        module.images_list.value = "A"
        module.objects_list.value = "Objects"
        module.wants_objects.value = True
        module.subsample_size.value = sub
        module.image_sample_size.value = background
        module.element_size.value = 2
        module.granular_spectrum_length.value = 6
        pipeline = Pipeline()
        pipeline.add_module(module)
        image_list = ImageSetList()
        images = image_list.get_image_set(0)
        images.add("A", Image(pixels, convert=False))
        objects = Objects()
        objects.segmented = labels
        object_set = ObjectSet()
        object_set.add_objects(objects, "Objects")
        measurements = Measurements()
        try:
            module.run(
                Workspace(
                    pipeline, module, images, object_set, measurements, image_list
                )
            )
            cases.append(
                dict(
                    subsample_size=sub,
                    image_sample_size=background,
                    expected={
                        f"Granularity_{n}": np.asarray(
                            measurements.get_current_measurement(
                                "Objects", f"Granularity_{n}_A"
                            )
                        ).tolist()
                        for n in range(1, 7)
                    },
                )
            )
        finally:
            measurements.close()
    payload = {
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("CellProfiler", "numpy", "scipy", "scikit-image")
        },
        "pixels_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        "cases": cases,
    }
    (
        Path(__file__).parent / "data/granularity_float32_cellprofiler_reference.json"
    ).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
