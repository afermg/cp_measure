import numpy as np, cProfile, pstats, os
from cp_measure.core.measureobjectsizeshape import get_sizeshape
d = np.load("/ictstr01/groups/ml01/workspace/ttreis/projects/cp_measure_3tier_bench/data/large.npz")
mask = d["mask_int"].astype(np.int32); pixels = d["pixels"].astype(np.float32)
get_sizeshape(mask, pixels)
pr=cProfile.Profile(); pr.enable()
for _ in range(5): get_sizeshape(mask, pixels)
pr.disable()
ps=pstats.Stats(pr).sort_stats("tottime")
print(f"{'tottime/5':>9} {'cumtime/5':>9} {'ncalls':>7}  function")
for (f,l,fn),(cc,nc,tt,ct,_) in sorted(ps.stats.items(),key=lambda kv:-kv[1][2])[:16]:
    print(f"{tt/5*1e3:9.1f} {ct/5*1e3:9.1f} {nc:>7}  {os.path.basename(f)}:{l}({fn})")
