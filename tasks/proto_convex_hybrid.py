import time, numpy as np, skimage.measure
import centrosome.cpmorphology as cm
from cp_measure.utils import masks_to_ijv
d = np.load("/ictstr01/groups/ml01/workspace/ttreis/projects/cp_measure_3tier_bench/data/large.npz")
mask = d["mask_int"].astype(np.int32)
ul = np.unique(mask); ul = ul[ul>0]; n=len(ul)
ref = skimage.measure.regionprops_table(mask, properties=["area_convex","bbox"])
refac = ref["area_convex"]

def hybrid_area_convex():
    ijv = masks_to_ijv(mask)
    ch, counts = cm.convex_hull_ijv(ijv, ul)        # batched hull vertices (label,i,j)
    out = np.zeros(n)
    off = 0
    # per-object: rasterize hull with skimage's exact grid_points_in_poly
    for k in range(n):
        c = counts[k]
        pts = ch[off:off+c, 1:3].astype(float)       # (i,j) vertices for this object
        off += c
        if c < 3:
            out[k] = c  # degenerate (point/line): skimage hull image = the pixels themselves
            continue
        rmin, cmin = pts[:,0].min(), pts[:,1].min()
        rmax, cmax = pts[:,0].max(), pts[:,1].max()
        shape = (int(rmax-rmin)+1, int(cmax-cmin)+1)
        local = pts - [rmin, cmin]
        out[k] = skimage.measure.grid_points_in_poly(shape, local).sum()
    return out

ac = hybrid_area_convex()
diff = np.abs(ac-refac)
print(f"hybrid (centrosome hull + skimage grid_points_in_poly):")
print(f"  exact matches {(diff==0).sum()}/{n}; within 1px {(diff<=1).sum()}/{n}; within 3px {(diff<=3).sum()}/{n}")
print(f"  max|diff| {diff.max():.0f}  mean|diff| {diff.mean():.2f}")
def bench(f,reps=4):
    f(); return min((lambda:(a:=time.perf_counter(),f(),time.perf_counter()-a)[-1])() for _ in range(reps))*1e3
print(f"  speed: {bench(hybrid_area_convex):.1f} ms  (skimage ~90ms)")
