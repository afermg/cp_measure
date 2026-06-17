import time, numpy as np, numba
from cp_measure.primitives._moments import spatial_moments_2d
d = np.load("/ictstr01/groups/ml01/workspace/ttreis/projects/cp_measure_3tier_bench/data/large.npz")
mask = d["mask_int"].astype(np.int32)
ul = np.unique(mask); ul=ul[ul>0]; n=len(ul)

@numba.njit(cache=True)
def _kernel(rows, cols, obj, n):
    rmin=np.full(n,1<<30,np.int64); cmin=np.full(n,1<<30,np.int64)
    for k in range(obj.shape[0]):
        o=obj[k]
        if rows[k]<rmin[o]: rmin[o]=rows[k]
        if cols[k]<cmin[o]: cmin[o]=cols[k]
    raw=np.zeros((n,4,4)); central=np.zeros((n,4,4))
    for k in range(obj.shape[0]):
        o=obj[k]; lr=float(rows[k]-rmin[o]); lc=float(cols[k]-cmin[o]); rp=1.0
        for p in range(4):
            cp=1.0
            for q in range(4):
                raw[o,p,q]+=rp*cp; cp*=lc
            rp*=lr
    cr=np.empty(n); cc=np.empty(n)
    for o in range(n):
        cr[o]=raw[o,1,0]/raw[o,0,0]; cc[o]=raw[o,0,1]/raw[o,0,0]
    for k in range(obj.shape[0]):
        o=obj[k]; dr=(rows[k]-rmin[o])-cr[o]; dc=(cols[k]-cmin[o])-cc[o]; rp=1.0
        for p in range(4):
            cp=1.0
            for q in range(4):
                central[o,p,q]+=rp*cp; cp*=dc
            rp*=dr
    return raw, central

def numba_moments(mask):
    rows,cols=np.nonzero(mask); obj=np.searchsorted(ul,mask[rows,cols]).astype(np.int64)
    return _kernel(rows.astype(np.int64),cols.astype(np.int64),obj,n)

numba_moments(mask)  # JIT warmup
def b(f,r=9):
    f(); return np.median([(lambda:(t:=time.perf_counter(),f(),(time.perf_counter()-t)*1e3)[-1])() for _ in range(r)])
t_np=b(lambda: spatial_moments_2d(mask)); t_nb=b(lambda: numba_moments(mask))
print(f"numpy spatial_moments_2d (32 bincounts) = {t_np:6.2f} ms")
print(f"numba 2-pass kernel (full, incl prep)   = {t_nb:6.2f} ms   ({t_np/t_nb:.1f}x)")
rows,cols=np.nonzero(mask); obj=np.searchsorted(ul,mask[rows,cols]).astype(np.int64)
ri,ci=rows.astype(np.int64),cols.astype(np.int64)
print(f"  numba kernel only (no nonzero prep)   = {b(lambda: _kernel(ri,ci,obj,n)):6.2f} ms")
