import numpy as np, skimage.measure
from skimage.morphology.convex_hull import _offsets_diamond
from skimage.measure import grid_points_in_poly
from scipy.spatial import ConvexHull
d = np.load("/ictstr01/groups/ml01/workspace/ttreis/projects/cp_measure_3tier_bench/data/large.npz")
mask = d["mask_int"].astype(np.int32)
ul = np.unique(mask); ul=ul[ul>0]; n=len(ul)
ref = skimage.measure.regionprops_table(mask, properties=["area_convex","image"])
refac = ref["area_convex"]; imgs = ref["image"]

def monotone_chain(pts):
    """Andrew's monotone chain convex hull (numba-able). Returns hull vertices CCW."""
    pts = pts[np.lexsort((pts[:,1], pts[:,0]))]
    # remove exact dups
    keep = np.ones(len(pts), bool); keep[1:] = np.any(np.diff(pts,axis=0)!=0,axis=1); pts=pts[keep]
    if len(pts) < 3: return pts
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in pts:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0: lower.pop()
        lower.append(p)
    upper=[]
    for p in pts[::-1]:
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0: upper.pop()
        upper.append(p)
    return np.array(lower[:-1]+upper[:-1])

def area_via(img, hull_fn):
    coords = np.transpose(np.nonzero(img)).astype(float)
    if len(coords)==0: return 0
    pts = (coords[:,None,:]+_offsets_diamond(2)).reshape(-1,2)
    verts = hull_fn(pts)
    if len(verts)<3: return int(img.sum())
    return int((grid_points_in_poly(img.shape, verts, binarize=False) >= 1).sum())

# A) sanity: skimage's own ConvexHull vertices + grid_points_in_poly  (should be exact)
def scipy_hull(pts):
    try: h=ConvexHull(pts); return h.points[h.vertices]
    except Exception: return pts[:0]
dA = np.array([area_via(imgs[k], scipy_hull) - refac[k] for k in range(n)])
# B) hybrid: monotone-chain hull + grid_points_in_poly
dB = np.array([area_via(imgs[k], monotone_chain) - refac[k] for k in range(n)])
print(f"A) scipy ConvexHull + grid_points_in_poly : exact {(dA==0).sum()}/{n}, max|diff| {np.abs(dA).max()}")
print(f"B) monotone-chain  + grid_points_in_poly  : exact {(dB==0).sum()}/{n}, max|diff| {np.abs(dB).max()}")
