import time, numpy as np, numba, skimage.measure
from skimage.measure import grid_points_in_poly
d = np.load("/ictstr01/groups/ml01/workspace/ttreis/projects/cp_measure_3tier_bench/data/large.npz")
mask = d["mask_int"].astype(np.int32)
ul = np.unique(mask); ul = ul[ul>0]; n=len(ul)
ref = skimage.measure.regionprops_table(mask, properties=["area_convex","bbox"])
refac = ref["area_convex"]
bb = np.column_stack([ref["bbox-0"],ref["bbox-1"],ref["bbox-2"],ref["bbox-3"]])
STRIDE=np.int64(10_000)

@numba.njit
def _hulls(px, py, offsets, n, stride):
    total = px.shape[0]
    out_x = np.empty(total, np.float64); out_y = np.empty(total, np.float64)
    hoff = np.zeros(n + 1, np.int64); cur = 0
    for o in range(n):
        s = offsets[o]; e = offsets[o + 1]; m = e - s
        if m == 0: hoff[o+1]=cur; continue
        key = px[s:e]*stride + py[s:e]; order = np.argsort(key)
        sx = px[s:e][order]; sy = py[s:e][order]
        ux = np.empty(m, np.int64); uy = np.empty(m, np.int64); u = 0
        for i in range(m):
            if u==0 or sx[i]!=ux[u-1] or sy[i]!=uy[u-1]: ux[u]=sx[i]; uy[u]=sy[i]; u+=1
        if u < 3:
            for i in range(u): out_x[cur]=ux[i]/2.0; out_y[cur]=uy[i]/2.0; cur+=1
            hoff[o+1]=cur; continue
        hx=np.empty(2*u,np.int64); hy=np.empty(2*u,np.int64); k=0
        for i in range(u):
            while k>=2 and (hx[k-1]-hx[k-2])*(uy[i]-hy[k-2])-(hy[k-1]-hy[k-2])*(ux[i]-hx[k-2])<=0: k-=1
            hx[k]=ux[i]; hy[k]=uy[i]; k+=1
        low=k+1
        for i in range(u-2,-1,-1):
            while k>=low and (hx[k-1]-hx[k-2])*(uy[i]-hy[k-2])-(hy[k-1]-hy[k-2])*(ux[i]-hx[k-2])<=0: k-=1
            hx[k]=ux[i]; hy[k]=uy[i]; k+=1
        for i in range(k-1): out_x[cur]=hx[i]/2.0; out_y[cur]=hy[i]/2.0; cur+=1
        hoff[o+1]=cur
    return out_x[:cur], out_y[:cur], hoff

def boundary_mask(m):
    H,W=m.shape; pad=np.pad(m,1); fg=m>0; allsame=np.ones_like(fg)
    for di in (-1,0,1):
        for dj in (-1,0,1):
            if di==0 and dj==0: continue
            allsame &= (pad[1+di:H+1+di,1+dj:W+1+dj]==m)
    return fg & ~allsame

def build_points_boundary(mask):
    bnd=boundary_mask(mask); rows,cols=np.nonzero(bnd); obj=np.searchsorted(ul,mask[rows,cols])
    r2=rows.astype(np.int64)*2; c2=cols.astype(np.int64)*2
    px=np.concatenate([r2-1,r2+1,r2,r2]); py=np.concatenate([c2,c2,c2-1,c2+1]); o4=np.concatenate([obj]*4)
    order=np.argsort(o4,kind="stable"); px,py,o4=px[order],py[order],o4[order]
    offs=np.zeros(n+1,np.int64); np.add.at(offs,o4+1,1); offs=np.cumsum(offs)
    return px,py,offs

pxb,pyb,offb=build_points_boundary(mask); hx,hy,hoff=_hulls(pxb,pyb,offb,n,STRIDE)
diffs=np.empty(n)
for o in range(n):
    rmin,cmin,rmax,cmax=bb[o]
    v=np.column_stack([hx[hoff[o]:hoff[o+1]]-rmin, hy[hoff[o]:hoff[o+1]]-cmin])
    diffs[o]=(int((grid_points_in_poly((int(rmax-rmin),int(cmax-cmin)),v,binarize=False)>=1).sum()) if len(v)>=3 else 0)-refac[o]
print(f"boundary numba hull + raster: exact {(diffs==0).sum()}/{n}, max|diff| {np.abs(diffs).max():.0f}")
print(f"boundary pts {len(pxb)} vs all-fg pts {4*int((mask>0).sum())} ({4*int((mask>0).sum())/len(pxb):.0f}x fewer)")
def b(f,r=7):
    f(); return np.median([(lambda:(t:=time.perf_counter(),f(),(time.perf_counter()-t)*1e3)[-1])() for _ in range(r)])
def full():
    p,q,of=build_points_boundary(mask); ax,ay,ho=_hulls(p,q,of,n,STRIDE); out=np.empty(n)
    for o in range(n):
        rmin,cmin,rmax,cmax=bb[o]; v=np.column_stack([ax[ho[o]:ho[o+1]]-rmin,ay[ho[o]:ho[o+1]]-cmin])
        out[o]=(grid_points_in_poly((int(rmax-rmin),int(cmax-cmin)),v,binarize=False)>=1).sum() if len(v)>=3 else 0
    return out
print(f"\nskimage area_convex             = {b(lambda: skimage.measure.regionprops_table(mask, properties=['area_convex'])):6.1f} ms")
print(f"numba boundary-hull (build+kern)= {b(lambda: _hulls(*build_points_boundary(mask),n,STRIDE)):6.1f} ms")
print(f"FULL numba convex (hull+raster) = {b(full):6.1f} ms")
