import numpy as np, skimage.measure
d = np.load("/ictstr01/groups/ml01/workspace/ttreis/projects/cp_measure_3tier_bench/data/large.npz")
mask = d["mask_int"].astype(np.int32)
ul = np.unique(mask); ul = ul[ul>0]; n=len(ul)
ref = skimage.measure.regionprops_table(mask, properties=["moments_central","inertia_tensor"])

rows, cols = np.nonzero(mask); lab = mask[rows,cols]; obj = np.searchsorted(ul, lab)
rmin = np.full(n,1<<30); cmin = np.full(n,1<<30)
np.minimum.at(rmin,obj,rows); np.minimum.at(cmin,obj,cols)
lr = (rows-rmin[obj]).astype(np.float64); lc=(cols-cmin[obj]).astype(np.float64)
m00 = np.bincount(obj,minlength=n).astype(float)
cr = np.bincount(obj,weights=lr,minlength=n)/m00
cc = np.bincount(obj,weights=lc,minlength=n)/m00
# DIRECT centered scatter (skimage's actual method): mu[p,q]=sum((lr-cr)^p (lc-cc)^q)
dr = lr - cr[obj]; dc = lc - cc[obj]
drp=[np.ones_like(dr),dr,dr*dr,dr*dr*dr]; dcp=[np.ones_like(dc),dc,dc*dc,dc*dc*dc]
mu = np.zeros((n,4,4))
for p in range(4):
    for q in range(4):
        mu[:,p,q]=np.bincount(obj,weights=drp[p]*dcp[q],minlength=n)
md = max(np.max(np.abs(mu[:,p,q]-ref[f"moments_central-{p}-{q}"])) for p in range(4) for q in range(4))
print(f"central moments (DIRECT centered scatter)  max|diff vs skimage| = {md:.2e}")
# inertia tensor derives from central moments: skimage uses mu normalized by mu00
it = ref
# inertia_tensor[0,0]=mu[2,0]/mu00, [1,1]=mu[0,2]/mu00, [0,1]=[1,0]=-mu[1,1]/mu00
i00 = mu[:,2,0]/mu[:,0,0]; i11=mu[:,0,2]/mu[:,0,0]; i01=-mu[:,1,1]/mu[:,0,0]
md_i = max(np.max(np.abs(i00-it["inertia_tensor-0-0"])), np.max(np.abs(i11-it["inertia_tensor-1-1"])),
           np.max(np.abs(i01-it["inertia_tensor-0-1"])))
print(f"inertia_tensor (from central moments)      max|diff vs skimage| = {md_i:.2e}")
