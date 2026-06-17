import time, numpy, scipy.ndimage, scipy.sparse, skimage.morphology
from cp_measure.core.measuregranularity import get_granularity
from cp_measure.utils import _ensure_np_array as fix

def make(size, n_obj, radius, seed=1):
    rng = numpy.random.default_rng(seed)
    masks = numpy.zeros((size, size), numpy.int32)
    yy, xx = numpy.mgrid[0:size, 0:size]
    centers = rng.integers(radius+1, size-radius-1, size=(n_obj, 2))
    for lab,(cy,cx) in enumerate(centers,1):
        masks[(yy-cy)**2+(xx-cx)**2 < radius*radius] = lab
    pix = scipy.ndimage.gaussian_filter(rng.random((size,size)),3)+0.3*rng.random((size,size))
    return masks, pix.astype(numpy.float64)

def gran_patched(mask, pixels, subsample_size=0.25, image_sample_size=0.25,
                 element_size=10, granular_spectrum_length=16):
    # ----- VERBATIM from source: downsample + background subtraction -----
    orig_shape = numpy.array(pixels.shape); orig_pixels = pixels; orig_mask = mask
    new_shape = orig_shape.copy()
    if subsample_size < 1:
        new_shape = (orig_shape*subsample_size).astype(int)
        i,j = numpy.mgrid[0:new_shape[0],0:new_shape[1]].astype(float)/subsample_size
        pixels = scipy.ndimage.map_coordinates(pixels,(i,j),order=1)
        mask = scipy.ndimage.map_coordinates(mask,(i,j),order=0).astype(orig_mask.dtype)
    else:
        pixels = pixels.copy(); mask = mask.copy()
    if image_sample_size < 1:
        back_shape = new_shape * image_sample_size
        i,j = numpy.mgrid[0:back_shape[0],0:back_shape[1]].astype(float)/image_sample_size
        back_pixels = scipy.ndimage.map_coordinates(pixels,(i,j),order=1)
        back_mask = numpy.ones(back_pixels.shape, dtype=bool)
    else:
        back_pixels = pixels; back_mask = numpy.ones(back_pixels.shape, dtype=bool); back_shape = new_shape
    radius = element_size
    footprint = skimage.morphology.disk(radius, dtype=bool)
    bpm = numpy.zeros_like(back_pixels); bpm[back_mask==1]=back_pixels[back_mask==1]
    back_pixels = skimage.morphology.erosion(bpm, footprint=footprint)
    bpm = numpy.zeros_like(back_pixels); bpm[back_mask==1]=back_pixels[back_mask==1]
    back_pixels = skimage.morphology.dilation(bpm, footprint=footprint)
    if image_sample_size < 1:
        i,j = numpy.mgrid[0:new_shape[0],0:new_shape[1]].astype(float)
        i *= float(back_shape[0]-1)/float(new_shape[0]-1); j *= float(back_shape[1]-1)/float(new_shape[1]-1)
        back_pixels = scipy.ndimage.map_coordinates(back_pixels,(i,j),order=1)
    pixels -= back_pixels; pixels[pixels<0]=0
    # ----- granular spectrum loop -----
    ng = granular_spectrum_length
    footprint = skimage.morphology.disk(1, dtype=bool)
    ero = pixels.copy()
    unique_labels = numpy.unique(orig_mask); unique_labels = unique_labels[unique_labels>0]
    range_ = numpy.arange(1, numpy.max(orig_mask)+1); maxlab=int(numpy.max(orig_mask))
    current_mean = fix(scipy.ndimage.mean(orig_pixels, orig_mask, range_))
    start_mean = numpy.maximum(current_mean, numpy.finfo(float).eps)

    # ----- NEW: precompute fused upsample+mean matrix D (once) -----
    i,j = numpy.mgrid[0:orig_shape[0],0:orig_shape[1]].astype(float)
    i *= float(new_shape[0]-1)/float(orig_shape[0]-1); j *= float(new_shape[1]-1)/float(orig_shape[1]-1)
    r=i.ravel(); c=j.ravel(); r0=numpy.floor(r).astype(int); c0=numpy.floor(c).astype(int); fr=r-r0; fc=c-c0
    r1=numpy.minimum(r0+1,new_shape[0]-1); c1=numpy.minimum(c0+1,new_shape[1]-1); nd1=int(new_shape[1])
    cols=numpy.concatenate([r0*nd1+c0,r0*nd1+c1,r1*nd1+c0,r1*nd1+c1])
    data=numpy.concatenate([(1-fr)*(1-fc),(1-fr)*fc,fr*(1-fc),fr*fc])
    labL=orig_mask.ravel(); rows=numpy.concatenate([labL,labL,labL,labL]); keep=rows>0
    D=scipy.sparse.coo_matrix((data[keep],(rows[keep],cols[keep])),
                              shape=(maxlab+1,int(new_shape[0])*int(new_shape[1]))).tocsr()
    count=numpy.bincount(labL, minlength=maxlab+1).astype(float)[1:maxlab+1]

    results={}
    for gid in range(1, ng+1):
        ero_mask=ero.copy()
        ero=skimage.morphology.erosion(ero_mask, footprint=footprint)
        rec=skimage.morphology.reconstruction(ero, pixels, footprint=footprint)
        gss=numpy.zeros((0,))
        if unique_labels.any():
            with numpy.errstate(invalid="ignore"):
                new_mean=(D @ rec.ravel())[1:maxlab+1]/count
            gss=(current_mean-new_mean)*100/start_mean
            current_mean=new_mean
        results[f"Granularity_{gid}"]=gss
    return results

for label,size,nobj,rad in [("1080 dense",1080,140,35),("1080 sparse",1080,40,30),("2160",2160,180,40)]:
    mask,pixels = make(size,nobj,rad)
    ref=get_granularity(mask,pixels); got=gran_patched(mask,pixels)
    md=max(numpy.nanmax(numpy.abs(got[k]-ref[k])) if got[k].size else 0.0 for k in ref)
    nan_ok=all(numpy.array_equal(numpy.isnan(got[k]),numpy.isnan(ref[k])) for k in ref if got[k].size)
    def bench(fn,reps=3):
        fn(); return min((lambda:(a:=time.perf_counter(),fn(),time.perf_counter()-a)[-1])() for _ in range(reps))
    t_ref=bench(lambda:get_granularity(mask,pixels)); t_new=bench(lambda:gran_patched(mask,pixels))
    print(f"{label:12} {size}^2 obj={int((numpy.unique(mask)>0).sum()):3}  CURRENT={t_ref*1e3:5.0f}ms FUSED={t_new*1e3:5.0f}ms  {t_ref/t_new:.2f}x  maxdiff={md:.1e} nan_ok={nan_ok}")
