import time, numpy as np, skimage.measure
import centrosome.cpmorphology as cm
from cp_measure.utils import masks_to_ijv
d = np.load("/ictstr01/groups/ml01/workspace/ttreis/projects/cp_measure_3tier_bench/data/large.npz")
mask = d["mask_int"].astype(np.int32)
ul = np.unique(mask); ul = ul[ul>0]; n=len(ul)

# skimage reference (per-region)
ref = skimage.measure.regionprops_table(mask, properties=["area_convex"])["area_convex"]

def area_convex_centrosome_full():
    ijv = masks_to_ijv(mask)
    ch, counts = cm.convex_hull_ijv(ijv, ul)
    filled = cm.fill_convex_hulls(ch, counts)         # (M,3): label,i,j of interior points
    return np.bincount(filled[:,0], minlength=int(ul.max())+1)[ul].astype(float)

def area_convex_calc():
    return cm.calculate_convex_hull_areas(mask, ul)

# correctness
ac = area_convex_centrosome_full()
print(f"centrosome fill_convex_hulls: max|diff vs skimage| = {np.max(np.abs(ac-ref)):.1f}  mean|diff| = {np.mean(np.abs(ac-ref)):.3f}")
print(f"  exact matches: {(ac==ref).sum()}/{n};  within 1px: {(np.abs(ac-ref)<=1).sum()}/{n}; within 3px: {(np.abs(ac-ref)<=3).sum()}/{n}")
try:
    acc = area_convex_calc()
    print(f"calculate_convex_hull_areas: max|diff vs skimage| = {np.max(np.abs(acc-ref)):.1f}  (likely polygon area, not pixel count)")
except Exception as e:
    print(f"calculate_convex_hull_areas error: {e}")

def bench(f,reps=4):
    f(); return min((lambda:(a:=time.perf_counter(),f(),time.perf_counter()-a)[-1])() for _ in range(reps))*1e3
print(f"\nskimage area_convex (per-region) = {bench(lambda: skimage.measure.regionprops_table(mask, properties=['area_convex'])):7.1f} ms")
print(f"centrosome batched (full ijv)    = {bench(area_convex_centrosome_full):7.1f} ms")
