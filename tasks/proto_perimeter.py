import time, numpy as np, skimage.measure
from math import sqrt
d = np.load("/ictstr01/groups/ml01/workspace/ttreis/projects/cp_measure_3tier_bench/data/large.npz")
mask = d["mask_int"].astype(np.int32)
ul = np.unique(mask); ul=ul[ul>0]; n=len(ul)
ref = skimage.measure.regionprops_table(mask, properties=["perimeter"])["perimeter"]

def perimeter_batched(mask):
    # pad so shifts are clean (background border, matches skimage's isolated bbox + cval=0)
    m = np.pad(mask, 1)
    H,W = m.shape
    c = m[1:-1,1:-1]                                   # center labels
    fg = c > 0
    # 8-neighbour same-label indicator
    def same(di,dj): return m[1+di:H-1+di, 1+dj:W-1+dj] == c
    up,down,left,right = same(-1,0),same(1,0),same(0,-1),same(0,1)
    ul_,ur,dl,dr = same(-1,-1),same(-1,1),same(1,-1),same(1,1)
    # STREL_4 erosion: survives if all 4-conn are same label; border = fg & not(all 4)
    border = fg & ~(up & down & left & right)
    # perimeter_image at border pixels = 1 + 2*(4conn border same-label) + 10*(diag border same-label)
    # need border indicator of neighbours too (border AND same label)
    def bsame(di,dj):
        nb_lab = m[1+di:H-1+di, 1+dj:W-1+dj]
        # neighbour is a border pixel of the SAME label
        # compute neighbour border via its own 4-conn — approximate by recomputing border globally
        return None
    # global border map (any label), then mask by same-label
    fullc = m>0
    def s2(a,di,dj): return np.pad(a,1)[1+di:H-1+di,1+dj:W-1+dj] if False else None
    # recompute border on padded grid for neighbour lookups
    mm = m
    cc = mm
    return border  # placeholder

# Simpler exact approach: replicate skimage per-pixel via global border then same-label conv
def perimeter_batched2(mask):
    m = np.pad(mask, 1).astype(np.int64)
    lab = m[1:-1,1:-1]
    fg = lab > 0
    H,W = m.shape
    nb = lambda di,dj: m[1+di:H-1+di, 1+dj:W-1+dj]
    same = lambda di,dj: nb(di,dj) == lab
    # border (STREL_4 erosion)
    border = fg & ~(same(-1,0)&same(1,0)&same(0,-1)&same(0,1))
    bi = border.astype(np.int64)
    # border map padded for neighbour access
    bp = np.pad(border, 1)
    labp = np.pad(lab, 1)
    Hb,Wb = bp.shape
    def bnb(di,dj):  # neighbour is border AND same label as center
        nbb = bp[1+di:Hb-1+di, 1+dj:Wb-1+dj]
        nbl = labp[1+di:Hb-1+di, 1+dj:Wb-1+dj]
        return (nbb & (nbl==lab)).astype(np.int64)
    pim = (bi*1
           + 2*(bnb(-1,0)+bnb(1,0)+bnb(0,-1)+bnb(0,1))
           + 10*(bnb(-1,-1)+bnb(-1,1)+bnb(1,-1)+bnb(1,1)))
    pim = pim * border  # only border-centre pixels contribute nonzero weights
    w = np.zeros(50); w[[5,7,15,17,25,27]]=1; w[[21,33]]=sqrt(2); w[[13,23]]=(1+sqrt(2))/2
    # histogram pim per label, weighted
    flat_lab = lab[border]; flat_val = pim[border]
    obj = np.searchsorted(ul, flat_lab)
    out = np.zeros(n)
    for v in np.unique(flat_val):
        if v<50 and w[v]!=0:
            out += w[v]*np.bincount(obj[flat_val==v], minlength=n)
    return out

pb = perimeter_batched2(mask)
diff = np.abs(pb-ref)
print(f"batched perimeter (4-conn) vs skimage: max|diff| {diff.max():.2e}  exact {(diff<1e-9).sum()}/{n}")
def bench(f,reps=4):
    f(); return min((lambda:(t:=time.perf_counter(),f(),time.perf_counter()-t)[-1])() for _ in range(reps))*1e3
print(f"skimage perimeter (per-region) = {bench(lambda: skimage.measure.regionprops_table(mask, properties=['perimeter'])):6.1f} ms")
print(f"batched perimeter              = {bench(lambda: perimeter_batched2(mask)):6.1f} ms")
