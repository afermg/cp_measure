"""Regenerate tiny references with CellProfiler 4.2.8 (optional test dependency).

Run this script in a CP 4.2.8 environment, not the cp_measure test environment.
The original references used cpm_benchmark's Nix-pinned legacy environment.
"""

import importlib.metadata
import json
from pathlib import Path

import numpy as np


def main():
    import cellprofiler_core.preferences

    cellprofiler_core.preferences.set_headless()
    from cellprofiler.modules.measurecolocalization import MeasureColocalization
    from cellprofiler_core.image import Image, ImageSetList
    from cellprofiler_core.measurement import Measurements
    from cellprofiler_core.object import Objects, ObjectSet
    from cellprofiler_core.pipeline import Pipeline
    from cellprofiler_core.workspace import Workspace

    cases = [
        dict(
            name="float32_threshold_boundary",
            first=[[3, 20], [10, 0]],
            second=[[20, 3], [10, 0]],
            labels=[[1, 1], [1, 1]],
        ),
        dict(
            name="two_object_rank_scope",
            first=[[65535, 10, 20, 20, 40], [65535, 50, 60, 60, 90]],
            second=[[0, 40, 20, 10, 20], [0, 90, 60, 50, 60]],
            labels=[[0, 1, 1, 1, 1], [0, 2, 2, 2, 2]],
        ),
    ]
    for case in cases:
        module = MeasureColocalization()
        module.set_module_num(1)
        module.images_list.value = "A, B"
        module.objects_list.value = "Objects"
        module.images_or_objects.value = "Within objects"
        module.thr.value = 15
        module.do_all.value = False
        module.do_corr_and_slope.value = True
        module.do_manders.value = True
        module.do_rwc.value = True
        module.do_overlap.value = True
        module.do_costes.value = False
        pipeline = Pipeline()
        pipeline.add_module(module)
        image_list = ImageSetList()
        images = image_list.get_image_set(0)
        for channel, key in (("A", "first"), ("B", "second")):
            pixels = np.asarray(case[key], np.float32) / np.float32(65535)
            images.add(channel, Image(pixels, convert=False))
        objects = Objects()
        objects.segmented = np.asarray(case["labels"], np.int32)
        object_set = ObjectSet()
        object_set.add_objects(objects, "Objects")
        measurements = Measurements()
        try:
            module.run(
                Workspace(
                    pipeline, module, images, object_set, measurements, image_list
                )
            )
            case["expected"] = {
                name: np.asarray(
                    measurements.get_current_measurement("Objects", name)
                ).tolist()
                for name in measurements.get_feature_names("Objects")
                if name.startswith("Correlation_")
            }
        finally:
            measurements.close()
    payload = {
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("CellProfiler", "cellprofiler-core", "numpy", "scipy")
        },
        "normalization": "float32(raw) / float32(65535)",
        "threshold_percent": 15,
        "cases": cases,
    }
    path = Path(__file__).parent / "data/colocalization_cellprofiler_reference.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
