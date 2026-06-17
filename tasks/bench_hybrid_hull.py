import time, numpy as np, skimage.measure
from skimage.morphology.convex_hull import _offsets_diamond
from skimage.measure import grid_points_in_poly
from scipy.spatial import ConvexHull
d = np.load("/ictstr01/groups/ml01/workspace/ttreis/projects/cp_measure_3tier_bench/data/large.npz")
mask = d["mask_int"].astype(np.int32)
ul = np.unique(mask); ul=ul[ul>0]; n=len(ul)
imgs = skimage.measure.regionprops_table(mask, properties=["image"])["image"]
def bench(f,r=4):
    f(); return np.median([(lambda:(t:=time.perf_counter(),f(),(time.perf_counter()-t)*1e3)[-1])() for _ in range(r)])

# baseline: skimage area_convex (full convex_hull_image per region)
print(f"skimage area_convex (per-region)          = {bench(lambda: skimage.measure.regionprops_table(mask, properties=['area_convex'])):6.1f} ms")

# precompute hull vertices per object (scipy, to isolate the raster floor)
hulls=[]
for img in imgs:
    c=np.transpose(np.nonzero(img)).astype(float)
    pts=(c[:,None,:]+_offsets_diamond(2)).reshape(-1,2)
    try: h=ConvexHull(pts); hulls.append(h.points[h.vertices])
    except: hulls.append(pts)
def raster_only():
    return [int((grid_points_in_poly(imgs[k].shape, hulls[k], binarize=False)>=1).sum()) for k in range(n)]
print(f"grid_points_in_poly only (raster floor)   = {bench(raster_only):6.1f} ms  <- irreducible (keep skimage)")

# hull construction floor (python monotone-chain proxy; numba would be ~5-10x faster)
def mono(pts):
    pts=pts[np.lexsort((pts[:,1],pts[:,0]))]
    def cr(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lo=[]
    for p in pts:
        while len(lo)>=2 and cr(lo[-2],lo[-1],p)<=0: lo.pop()
        lo.append(p)
    up=[]
    for p in pts[::-1]:
        while len(up)>=2 and cr(up[-2],up[-1],p)<=0: up.pop()
        up.append(p)
    return np.array(lo[:-1]+up[:-1])
def full_hybrid_python():
    out=[]
    for img in imgs:
        c=np.transpose(np.nonzero(img)).astype(float)
        pts=(c[:,None,:]+_offsets_diamond(2)).reshape(-1,2)
        v=mono(pts)
        out.append(int((grid_points_in_poly(img.shape,v,binarize=False)>=1).sum()) if len(v)>=3 else int(img.sum()))
    return out
print(f"hybrid (PYTHON monotone-chain + raster)   = {bench(full_hybrid_python):6.1f} ms  <- numba hull would slash the hull part")
