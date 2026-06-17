"""DEEP investigation: speed up get_feret with numpy + existing deps only.

Current main get_feret(masks):
    ijv = masks_to_ijv(masks)                      # nonzero + argsort (PR #71)
    indices = unique(ijv[:,2])>0
    chulls, counts = convex_hull_ijv(ijv, indices) # COMPILED C, cost ~ #input points
    min_f, max_f = feret_diameter(chulls, counts, indices)  # pure-py, ~#hull points

Levers (numpy + existing deps only):
  A. convex_hull(masks, indices)            -> centrosome's own entry point; reduces via outline()
  B. convex_hull_ijv(masks_to_ijv(outline(masks)), indices)  -> explicit boundary ijv
  C. drop the argsort in masks_to_ijv on the hull path (C impl accepts unsorted ijv)
  D. skip the second unique() (indices already known from a single pass)

We measure component breakdown + each candidate end-to-end, and assert BIT-EXACT feret
vs current main on real Cell-Painting masks (tiny/small/large/m2160/m4320) + a touching-object
synthetic (the case where 4-conn erosion would be WRONG but outline is right).
"""

import time

import numpy as np
import centrosome.cpmorphology as cm


def masks_to_ijv(masks):
    """main's PR #71 version (our branch still has the old per-label loop; use main's
    so the baseline reflects current main, which feret work would build on)."""
    i, j = np.nonzero(masks)
    v = masks[i, j]
    order = np.argsort(v, kind="stable")
    return np.column_stack((i[order], j[order], v[order])).astype(int, copy=False)

DATA = "/ictstr01/groups/ml01/workspace/ttreis/projects/cp_measure_3tier_bench/data"


def load(name):
    return np.load(f"{DATA}/{name}.npz")["mask_int"]


def touching_synthetic(size=512, n=8):
    """Adjacent objects that SHARE borders -> 4-conn fg erosion drops shared edges,
    centrosome.outline keeps them. Exercises the correctness difference."""
    labels = np.zeros((size, size), np.int32)
    step = size // n
    for a in range(n):
        for b in range(n):
            labels[a * step:(a + 1) * step, b * step:(b + 1) * step] = a * n + b + 1
    return labels


def best(fn, reps=3):
    fn()
    t = float("inf")
    for _ in range(reps):
        s = time.perf_counter(); fn(); t = min(t, time.perf_counter() - s)
    return t * 1e3


# --- implementations ---
def feret_main(masks):
    ijv = masks_to_ijv(masks)
    idx = np.unique(ijv[:, 2]); idx = idx[idx > 0]
    ch, cc = cm.convex_hull_ijv(ijv, idx)
    return cm.feret_diameter(ch, cc, idx)


def feret_A(masks):
    """centrosome.convex_hull does outline() reduction internally."""
    idx = np.unique(masks); idx = idx[idx > 0]
    ch, cc = cm.convex_hull(masks, idx)
    return cm.feret_diameter(ch, cc, idx)


def feret_B(masks):
    """explicit outline boundary -> ijv -> hull."""
    out = cm.outline(masks)
    ijv = masks_to_ijv(out)
    idx = np.unique(masks); idx = idx[idx > 0]
    ch, cc = cm.convex_hull_ijv(ijv, idx)
    return cm.feret_diameter(ch, cc, idx)


def feret_B_noargsort(masks):
    """B, but build boundary ijv WITHOUT argsort (C hull accepts unsorted)."""
    out = cm.outline(masks)
    i, j = np.nonzero(out)
    v = out[i, j]
    ijv = np.column_stack((i, j, v)).astype(int, copy=False)
    idx = np.unique(masks); idx = idx[idx > 0]
    ch, cc = cm.convex_hull_ijv(ijv, idx)
    return cm.feret_diameter(ch, cc, idx)


def eq(a, b):
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))


def profile_components(masks, label):
    ijv = masks_to_ijv(masks)
    idx = np.unique(ijv[:, 2]); idx = idx[idx > 0]
    ch, cc = cm.convex_hull_ijv(ijv, idx)
    t_ijv = best(lambda: masks_to_ijv(masks))
    t_hull = best(lambda: cm.convex_hull_ijv(ijv, idx))
    t_fer = best(lambda: cm.feret_diameter(ch, cc, idx))
    t_out = best(lambda: cm.outline(masks))
    t_total = best(lambda: feret_main(masks))
    print(f"\n[{label}] shape={masks.shape} nobj={len(idx)}  full ijv pts={len(ijv)}")
    print(f"  components: masks_to_ijv={t_ijv:.2f}  convex_hull_ijv={t_hull:.2f}  "
          f"feret_diameter={t_fer:.2f}  (outline={t_out:.2f})  total={t_total:.2f} ms")
    return idx


def main():
    cases = [("tiny", load("tiny")), ("small", load("small")), ("large", load("large")),
             ("m2160", load("m2160")), ("touching8", touching_synthetic())]
    # m4320 (4320^2/2272obj) measured separately once — full-ijv baseline is pathologically slow.

    print("=" * 78)
    print("COMPONENT BREAKDOWN (current main get_feret)")
    print("=" * 78)
    for name, m in cases:
        profile_components(m, name)

    print("\n" + "=" * 78)
    print("CANDIDATE COMPARISON (bit-exactness + end-to-end speed vs main)")
    print("=" * 78)
    print(f"{'case':>10}{'main ms':>9}{'A ms':>8}{'B ms':>8}{'B-nas ms':>10}"
          f"{'A=':>4}{'B=':>4}{'Bn=':>4}{'A x':>7}{'B x':>7}{'Bn x':>7}")
    for name, m in cases:
        ref = feret_main(m)
        rA, rB, rBn = feret_A(m), feret_B(m), feret_B_noargsort(m)
        eA, eB, eBn = eq(ref, rA), eq(ref, rB), eq(ref, rBn)
        tm = best(lambda: feret_main(m))
        tA = best(lambda: feret_A(m))
        tB = best(lambda: feret_B(m))
        tBn = best(lambda: feret_B_noargsort(m))
        print(f"{name:>10}{tm:>9.2f}{tA:>8.2f}{tB:>8.2f}{tBn:>10.2f}"
              f"{str(eA):>4}{str(eB):>4}{str(eBn):>4}"
              f"{tm/tA:>6.2f}x{tm/tB:>6.2f}x{tm/tBn:>6.2f}x")


if __name__ == "__main__":
    main()
