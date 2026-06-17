import time, numpy as np, skimage.measure
d = np.load("/ictstr01/groups/ml01/workspace/ttreis/projects/cp_measure_3tier_bench/data/large.npz")
mask = d["mask_int"].astype(np.int32)
ul = np.unique(mask); ul = ul[ul>0]; n = len(ul)

# skimage reference: spatial + central moments (binary shape, order 3)
ref = skimage.measure.regionprops_table(mask, properties=["moments","moments_central","centroid"])

def raw_moments_scatter(mask, ul):
    """Per-object raw spatial moments M[p,q]=sum(lr^p lc^q) in LOCAL (bbox) coords, like skimage."""
    rows, cols = np.nonzero(mask)
    lab = mask[rows, cols]
    obj = np.searchsorted(ul, lab)
    n = len(ul)
    # per-object bbox min (local origin) — skimage uses region.image local coords
    rmin = np.full(n, 1<<30); cmin = np.full(n, 1<<30)
    np.minimum.at(rmin, obj, rows); np.minimum.at(cmin, obj, cols)
    lr = (rows - rmin[obj]).astype(np.float64)
    lc = (cols - cmin[obj]).astype(np.float64)
    M = np.zeros((n,4,4))
    rp = [np.ones_like(lr), lr, lr*lr, lr*lr*lr]
    cp = [np.ones_like(lc), lc, lc*lc, lc*lc*lc]
    for p in range(4):
        for q in range(4):
            M[:,p,q] = np.bincount(obj, weights=rp[p]*cp[q], minlength=n)
    return M

def central_from_raw(M):
    """Translation to central moments (skimage convention), order 3."""
    mu = np.zeros_like(M)
    cr = M[:,1,0]/M[:,0,0]; cc = M[:,0,1]/M[:,0,0]
    # skimage _moments.moments_central: shift by centroid via the standard formula
    for p in range(4):
        for q in range(4):
            s = 0.0
            for i in range(p+1):
                for j in range(q+1):
                    from math import comb
                    s = s + comb(p,i)*comb(q,j)*((-cr)**(p-i))*((-cc)**(q-j))*M[:,i,j]
            mu[:,p,q] = s
    return mu

M = raw_moments_scatter(mask, ul)
mu = central_from_raw(M)
# compare
md_raw = max(np.max(np.abs(M[:,p,q]-ref[f"moments-{p}-{q}"])) for p in range(4) for q in range(4))
md_cen = max(np.max(np.abs(mu[:,p,q]-ref[f"moments_central-{p}-{q}"])) for p in range(4) for q in range(4))
print(f"raw spatial moments  max|scatter - skimage| = {md_raw:.2e}")
print(f"central moments      max|scatter - skimage| = {md_cen:.2e}")

def bench(fn,reps=7):
    fn(); return min((lambda:(a:=time.perf_counter(),fn(),time.perf_counter()-a)[-1])() for _ in range(reps))*1e3
t_sk = bench(lambda: skimage.measure.regionprops_table(mask, properties=["moments","moments_central"]))
t_sc = bench(lambda: central_from_raw(raw_moments_scatter(mask, ul)))
print(f"\nskimage moments+central = {t_sk:7.1f} ms")
print(f"scatter moments+central = {t_sc:7.1f} ms   ({t_sk/t_sc:.1f}x)  [numpy; numba would cut the scatter further]")
