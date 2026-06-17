import numpy as np, skimage.measure
from skimage.morphology.convex_hull import _offsets_diamond
from scipy.spatial import ConvexHull
d = np.load("/ictstr01/groups/ml01/workspace/ttreis/projects/cp_measure_3tier_bench/data/large.npz")
mask = d["mask_int"].astype(np.int32)
ul = np.unique(mask); ul=ul[ul>0]; n=len(ul)
ref = skimage.measure.regionprops_table(mask, properties=["area_convex","image"])
refac = ref["area_convex"]; imgs = ref["image"]

def crossing_inside(verts, H, W):
    """Even-odd ray-cast point-in-poly for all grid points (numba-replicable)."""
    yy,xx = np.mgrid[0:H,0:W]
    py = yy.ravel().astype(float); px = xx.ravel().astype(float)
    inside = np.zeros(py.shape, bool)
    m = len(verts); j = m-1
    for i in range(m):
        yi,xi = verts[i]; yj,xj = verts[j]
        cond = ((xi>px) != (xj>px))
        # y of edge at px
        with np.errstate(divide='ignore', invalid='ignore'):
            yint = (yj-yi)*(px-xi)/(xj-xi) + yi
        inside ^= cond & (py < yint)
        j = i
    return inside.sum()

def my_area(region_img):
    coords = np.transpose(np.nonzero(region_img)).astype(float)
    if len(coords)==0: return 0
    pts = (coords[:,None,:] + _offsets_diamond(2)).reshape(-1,2)
    try: hull = ConvexHull(pts)
    except Exception: return int(region_img.sum())
    verts = hull.points[hull.vertices]
    H,W = region_img.shape
    return crossing_inside(verts, H, W)

diffs = np.array([my_area(imgs[k]) - refac[k] for k in range(n)])
print(f"numpy crossing-test vs skimage area_convex: exact {(diffs==0).sum()}/{n}, "
      f"within1 {(np.abs(diffs)<=1).sum()}/{n}, within3 {(np.abs(diffs)<=3).sum()}/{n}, "
      f"max {np.abs(diffs).max()}, mean {np.abs(diffs).mean():.2f}")
