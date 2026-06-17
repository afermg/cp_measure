"""Profile all 13 feature functions on real masks (post-rebase: main #69-73 + radial).
Identifies which functions still have numpy-only speedup headroom."""

import time
import numpy as np
from cp_measure import bulk

DATA = "/ictstr01/groups/ml01/workspace/ttreis/projects/cp_measure_3tier_bench/data"


def best(fn, reps=3):
    fn()
    t = float("inf")
    for _ in range(reps):
        s = time.perf_counter(); fn(); t = min(t, time.perf_counter() - s)
    return t * 1e3


def main():
    d = np.load(f"{DATA}/large.npz")
    m = d["mask_int"]; px = d["pixels"].astype(float); px2 = d["pixels_2"].astype(float)
    core = bulk.get_core_measurements()
    corr = bulk.get_correlation_measurements()
    print(f"large 1080^2 / {len(np.unique(m))-1} objects\n{'function':>22}{'ms':>10}")
    print("-" * 34)
    rows = []
    for name, fn in core.items():
        try:
            t = best(lambda: fn(m, px))
            rows.append((name, t))
        except Exception as e:
            rows.append((name, f"ERR {type(e).__name__}: {str(e)[:40]}"))
    for name, fn in corr.items():
        try:
            t = best(lambda: fn(px, px2, m))
            rows.append((name, t))
        except Exception as e:
            try:
                t = best(lambda: fn(m, px, px2))
                rows.append((name + "(m,px,px2)", t))
            except Exception as e2:
                rows.append((name, f"ERR {type(e2).__name__}: {str(e2)[:40]}"))
    rows_num = [(n, t) for n, t in rows if isinstance(t, float)]
    rows_err = [(n, t) for n, t in rows if not isinstance(t, float)]
    for name, t in sorted(rows_num, key=lambda r: -r[1]):
        print(f"{name:>22}{t:>10.1f}")
    for name, t in rows_err:
        print(f"{name:>22}  {t}")
    total = sum(t for _, t in rows_num)
    print("-" * 34)
    print(f"{'TOTAL':>22}{total:>10.1f}")


if __name__ == "__main__":
    main()
