import time, numpy
import centrosome.zernike, centrosome.cpmorphology, scipy.sparse
from cp_measure.utils import masks_to_ijv
from cp_measure.core.measureobjectintensitydistribution import get_radial_zernikes, M_CATEGORY

def make(size, n_obj, radius, seed=1):
    rng = numpy.random.default_rng(seed)
    masks = numpy.zeros((size, size), numpy.int32)
    yy, xx = numpy.mgrid[0:size, 0:size]
    centers = rng.integers(radius+1, size-radius-1, size=(n_obj, 2))
    for lab,(cy,cx) in enumerate(centers,1):
        masks[(yy-cy)**2+(xx-cx)**2 < radius*radius] = lab
    return masks, rng.random((size, size)).astype(numpy.float64)

# check l_ sorted
m,_ = make(512, 10, 30); ijv = masks_to_ijv(m)
print("l_ sorted ascending?", bool(numpy.all(numpy.diff(ijv[:,2]) >= 0)))

def rz_reduceat(labels, pixels, zernike_degree=9, weight_inside=True):
    if labels.ndim == 3: return {}
    zidx = centrosome.zernike.get_zernike_indexes(zernike_degree + 1)
    ul = numpy.unique(labels); ul = ul[ul > 0]
    ij, r = centrosome.cpmorphology.minimum_enclosing_circle(labels, ul)
    ijv = masks_to_ijv(labels)
    l_ = ijv[:, 2]
    in_b = (ijv[:, 0] < pixels.shape[0]) & (ijv[:, 1] < pixels.shape[1])
    ijv, l_ = ijv[in_b], l_[in_b]
    yx = (ijv[:, :2] - ij[l_ - 1, :]) / r[l_ - 1, numpy.newaxis]
    w = pixels[ijv[:, 0], ijv[:, 1]]
    results = {}
    if len(l_) == 0:
        for mp in ("Magnitude","Phase"):
            for n,m in zidx: results[f"{M_CATEGORY}_Zernike{mp}_{n}_{m}"] = numpy.zeros(0)
        return results
    if weight_inside:
        z = centrosome.zernike.construct_zernike_polynomials(yx[:,1], yx[:,0], zidx, weight=w)
    else:
        z = centrosome.zernike.construct_zernike_polynomials(yx[:,1], yx[:,0], zidx) * w[:,None]
    bounds = numpy.searchsorted(l_, ul)            # l_ is sorted -> contiguous segments
    V = numpy.add.reduceat(z, bounds, axis=0)      # (nlab, K) complex, single C reduction
    areas = numpy.bincount(numpy.searchsorted(ul, l_), minlength=len(ul)).astype(float)
    vr, vi = V.real, V.imag
    magnitude = numpy.sqrt(vr*vr + vi*vi) / areas[:, None]
    phase = numpy.arctan2(vr, vi)
    for i,(n,m) in enumerate(zidx):
        results[f"{M_CATEGORY}_ZernikeMagnitude_{n}_{m}"] = magnitude[:, i]
        results[f"{M_CATEGORY}_ZernikePhase_{n}_{m}"] = phase[:, i]
    return results

def bench(fn, reps=5):
    fn(); return min((lambda:(a:=time.perf_counter(),fn(),time.perf_counter()-a)[-1])() for _ in range(reps))

for label, size, n_obj, rad in [("1080 medium",1080,40,40),("1080 sparse",1080,40,20),("2160 sparse",2160,60,25),("256 dense",256,8,40)]:
    masks, pixels = make(size, n_obj, rad)
    fg=float((masks>0).mean()); nobj=int((numpy.unique(masks)>0).sum())
    ref = get_radial_zernikes(masks, pixels)
    t_ref = bench(lambda: get_radial_zernikes(masks, pixels))
    print(f"\n{label}: {size}^2 fg={fg:.1%} obj={nobj}  CURRENT={t_ref*1e3:.1f}ms")
    for wi in (True, False):
        got = rz_reduceat(masks, pixels, weight_inside=wi)
        md = max(numpy.nanmax(numpy.abs(got[k]-ref[k])) for k in ref)
        t = bench(lambda w=wi: rz_reduceat(masks, pixels, weight_inside=w))
        print(f"   reduceat weight_inside={wi!s:5} {t*1e3:7.1f} ms  speedup {t_ref/t:4.1f}x  maxdiff {md:.1e}")
