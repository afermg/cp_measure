import time, numpy as np, scipy.ndimage, skimage.measure
from cp_measure.core.measureobjectsizeshape import get_sizeshape

d = np.load("/ictstr01/groups/ml01/workspace/ttreis/projects/cp_measure_3tier_bench/data/large.npz")
mask = d["mask_int"].astype(np.int32); pixels = d["pixels"].astype(np.float32)
nobj = int((np.unique(mask)>0).sum())
print(f"large tile: {mask.shape}, {nobj} objects")

def bench(fn, reps=5):
    fn(); return min((lambda:(a:=time.perf_counter(),fn(),time.perf_counter()-a)[-1])() for _ in range(reps))*1e3

print(f"\nFULL get_sizeshape            = {bench(lambda: get_sizeshape(mask, pixels)):7.1f} ms")

# regionprops with property subsets to isolate each primitive's marginal cost
MOMENTS = ["image","area","area_bbox","equivalent_diameter_area","bbox","centroid",
           "extent","axis_major_length","axis_minor_length","eccentricity","orientation",
           "inertia_tensor","inertia_tensor_eigvals","moments","moments_hu",
           "moments_central","moments_normalized","euler_number","area_filled"]
def rp(props): return lambda: skimage.measure.regionprops_table(mask, pixels, properties=props)
t_mom   = bench(rp(MOMENTS))
t_conv  = bench(rp(MOMENTS+["area_convex","solidity"]))
t_perim = bench(rp(MOMENTS+["area_convex","solidity","perimeter","perimeter_crofton"]))
print(f"\nregionprops moments+euler+filled = {t_mom:7.1f} ms   (reducible bulk)")
print(f"  + area_convex + solidity       = {t_conv:7.1f} ms   (+{t_conv-t_mom:.1f} = CONVEX HULL)")
print(f"  + perimeter + perimeter_crofton= {t_perim:7.1f} ms   (+{t_perim-t_conv:.1f} = PERIMETER)")

# isolate euler_number and area_filled within the moments group
base = ["image","area","bbox","centroid","moments","moments_central"]
t_base = bench(rp(base))
t_euler = bench(rp(base+["euler_number"]))
t_filled = bench(rp(base+["area_filled"]))
print(f"\n  base moments only              = {t_base:7.1f} ms")
print(f"  + euler_number                 = {t_euler:7.1f} ms   (+{t_euler-t_base:.1f})")
print(f"  + area_filled                  = {t_filled:7.1f} ms   (+{t_filled-t_base:.1f})")

# EDT radius loop
def edt_loop():
    props = skimage.measure.regionprops_table(mask, pixels, properties=["image"])
    mx=np.zeros(nobj); mn=np.zeros(nobj); md=np.zeros(nobj)
    for i, im in enumerate(props["image"]):
        im = np.pad(im, 1); dist = scipy.ndimage.distance_transform_edt(im)
        inside = dist[im]; mx[i]=inside.max(); mn[i]=inside.mean(); md[i]=np.median(inside)
    return mx
print(f"\nEDT radius loop (incl. regionprops['image']) = {bench(edt_loop):7.1f} ms")
# just the EDT calls, given the crops
props = skimage.measure.regionprops_table(mask, pixels, properties=["image"])
crops = list(props["image"])
def edt_only():
    for im in crops:
        im2 = np.pad(im,1); dist=scipy.ndimage.distance_transform_edt(im2); _=dist[im2]
print(f"  EDT calls only (142 crops)     = {bench(edt_only):7.1f} ms")
