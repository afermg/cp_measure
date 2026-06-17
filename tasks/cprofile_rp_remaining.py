import numpy as np, cProfile, pstats, os, skimage.measure
d = np.load("/ictstr01/groups/ml01/workspace/ttreis/projects/cp_measure_3tier_bench/data/large.npz")
mask = d["mask_int"].astype(np.int32); pixels = d["pixels"].astype(np.float32)
# post-#77 + post-inertia-removal property set
props=["image","area","area_bbox","area_convex","equivalent_diameter_area","bbox","centroid",
       "euler_number","extent","perimeter","solidity","perimeter_crofton","area_filled"]
f=lambda: skimage.measure.regionprops_table(mask, pixels, properties=props)
f()
pr=cProfile.Profile(); pr.enable()
for _ in range(5): f()
pr.disable()
ps=pstats.Stats(pr).sort_stats("tottime")
print(f"{'tottime/5':>9} {'cumtime/5':>9} {'ncalls':>7}  function")
for (fl,l,fn),(cc,nc,tt,ct,_) in sorted(ps.stats.items(),key=lambda kv:-kv[1][2])[:16]:
    print(f"{tt/5*1e3:9.1f} {ct/5*1e3:9.1f} {nc:>7}  {os.path.basename(fl)}:{l}({fn})")
