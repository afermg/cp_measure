import time, numpy as np, skimage.measure
from cp_measure.primitives._moments import spatial_moments_2d
d = np.load("/ictstr01/groups/ml01/workspace/ttreis/projects/cp_measure_3tier_bench/data/large.npz")
mask = d["mask_int"].astype(np.int32)
ul = np.unique(mask); ul=ul[ul>0]; n=len(ul)
ref = skimage.measure.regionprops_table(mask, properties=["inertia_tensor","inertia_tensor_eigvals"])

_, central, _, _ = spatial_moments_2d(mask)
mu00 = central[:,0,0]
a = central[:,2,0]/mu00   # mu20/mu00
b = central[:,1,1]/mu00   # mu11/mu00
c = central[:,0,2]/mu00   # mu02/mu00
# skimage inertia_tensor 2D convention: [[c, -b], [-b, a]]  (verify empirically)
it00, it01, it10, it11 = c, -b, -b, a
print("inertia_tensor diffs vs skimage:")
for name,val,key in [("0_0",it00,"inertia_tensor-0-0"),("0_1",it01,"inertia_tensor-0-1"),
                     ("1_0",it10,"inertia_tensor-1-0"),("1_1",it11,"inertia_tensor-1-1")]:
    print(f"  {name}: max|diff| {np.max(np.abs(val-ref[key])):.2e}")
# eigenvalues of [[c,-b],[-b,a]] (symmetric), descending
tr=(a+c)/2; disc=np.sqrt(((c-a)/2)**2 + b**2)
ev0 = tr+disc; ev1 = tr-disc   # descending
print("eigvals diffs vs skimage:")
print(f"  eig0: max|diff| {np.max(np.abs(ev0-ref['inertia_tensor_eigvals-0'])):.2e}")
print(f"  eig1: max|diff| {np.max(np.abs(ev1-ref['inertia_tensor_eigvals-1'])):.2e}")

# Does removing inertia from regionprops kill the einsum (moments_central)?
def bench(props,reps=5):
    f=lambda: skimage.measure.regionprops_table(mask, properties=props)
    f(); return min((lambda:(t:=time.perf_counter(),f(),time.perf_counter()-t)[-1])() for _ in range(reps))*1e3
base=["image","area","area_bbox","area_convex","equivalent_diameter_area","bbox","centroid",
      "euler_number","extent","perimeter","solidity","perimeter_crofton","area_filled"]
print(f"\nregionprops WITHOUT inertia       = {bench(base):7.1f} ms")
print(f"regionprops WITH inertia+eigvals  = {bench(base+['inertia_tensor','inertia_tensor_eigvals']):7.1f} ms")
