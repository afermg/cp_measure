import numpy, warnings
import centrosome.zernike
from cp_measure.core.measureobjectsizeshape import _zernike_scores
from cp_measure.core.measureobjectintensitydistribution import get_radial_zernikes, M_CATEGORY
warnings.filterwarnings("ignore")

def rz_reuse(labels, pixels, zd=9):
    if labels.ndim == 3: return {}
    zidx = centrosome.zernike.get_zernike_indexes(zd + 1)
    ul = numpy.unique(labels); ul = ul[ul > 0]
    vr, vi, _ = _zernike_scores(labels, ul, zidx, weight=pixels)
    results = {}
    if len(ul) == 0:
        for mp in ("Magnitude","Phase"):
            for n,m in zidx: results[f"{M_CATEGORY}_Zernike{mp}_{n}_{m}"] = numpy.zeros(0)
        return results
    areas = numpy.bincount(labels.ravel())[ul].astype(float)
    mag = numpy.sqrt(vr*vr+vi*vi)/areas[:,None]; ph = numpy.arctan2(vr, vi)
    for i,(n,m) in enumerate(zidx):
        results[f"{M_CATEGORY}_ZernikeMagnitude_{n}_{m}"]=mag[:,i]
        results[f"{M_CATEGORY}_ZernikePhase_{n}_{m}"]=ph[:,i]
    return results

def cmp(name, labels, pixels):
    try:
        ref = get_radial_zernikes(labels, pixels); cur="ok"
    except Exception as e:
        ref=None; cur=f"CRASH: {type(e).__name__}"
    got = rz_reuse(labels, pixels)
    if ref is None:
        print(f"  {name:24} current={cur:18}  reuse=ok ({len(got)} keys)  -> reuse FIXES this"); return
    diffs=[numpy.nanmax(numpy.abs(got[k]-ref[k])) for k in ref if got[k].size]
    md=max(diffs) if diffs else 0.0
    print(f"  {name:24} current=ok  keys={list(got)==list(ref)} maxdiff={md:.1e}")

rng=numpy.random.default_rng(0)
cmp("empty mask", numpy.zeros((40,40),numpy.int32), rng.random((40,40)))
m=numpy.zeros((32,32),numpy.int32); m[16,16]=1; m[5:15,5:15]=2
cmp("single-pixel + normal", m, rng.random((32,32)))
m=numpy.zeros((96,96),numpy.int32); m[10:30,10:30]=1; m[40:60,40:60]=3; m[70:90,70:90]=7
cmp("non-contiguous {1,3,7}", m, rng.random((96,96)))
m=numpy.zeros((64,64),numpy.int32); m[0:20,0:20]=1; m[40:64,40:64]=2
cmp("edge-touching", m, rng.random((64,64)))
print(f"  {'3D returns empty':24} reuse={rz_reuse(numpy.zeros((4,16,16),numpy.int32),numpy.zeros((4,16,16)))=={}}")
