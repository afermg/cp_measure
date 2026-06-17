"""Profile reconstruction's share of granularity AFTER the bilinear-gather win,
at default (subsample=0.25) and fullres (subsample=1.0). Also captures the actual
(seed, mask) reconstruction inputs the spectrum loop produces, for an A/B of the
int32-packed-queue + triple-raster opts. Synthetic 1080^2 / 144 obj (matches the
prior benchmark setup)."""

import time
import numpy as np

import cp_measure.core.numba._granularity as G
import cp_measure.core.numba.measuregranularity as MG

# --- synthetic 1080^2 / 144-object field with textured intensity ---
rng = np.random.default_rng(0)
side, n = 1080, 144
yy, xx = np.mgrid[0:side, 0:side]
c = rng.integers(0, side, size=(n, 2))
best = np.full((side, side), np.inf)
lab = np.zeros((side, side), np.int32)
for i, (cy, cx) in enumerate(c, 1):
    d = (yy - cy) ** 2 + (xx - cx) ** 2
    m = d < best
    best[m] = d[m]
    lab[m] = i
# textured intensity (granularity needs real greyscale structure)
pix = (rng.random((side, side)) * 0.5 + 0.5 * np.sin(xx / 17.0) * np.sin(yy / 13.0) ** 2)
pix = pix.astype(np.float64)

# --- instrument the reconstruction kernel: count, total time, capture inputs ---
_real_recon = G.reconstruction_by_dilation_2d
captured = []


def make_timed(capture):
    state = {"t": 0.0, "n": 0}

    def timed(seed, mask):
        s = time.perf_counter()
        out = _real_recon(seed, mask)
        state["t"] += time.perf_counter() - s
        state["n"] += 1
        if capture and len(captured) < 16:
            captured.append((seed.copy(), mask.copy()))
        return out

    return timed, state


def run(subsample, capture=False):
    timed, state = make_timed(capture)
    MG.reconstruction_by_dilation_2d = timed
    # warmup (numba compile) + measured
    MG.get_granularity(lab, pix, subsample_size=subsample)
    state["t"] = 0.0
    state["n"] = 0
    t0 = time.perf_counter()
    MG.get_granularity(lab, pix, subsample_size=subsample)
    total = time.perf_counter() - t0
    MG.reconstruction_by_dilation_2d = _real_recon
    print(f"subsample={subsample}: total={total*1e3:7.1f}ms  recon={state['t']*1e3:7.1f}ms "
          f"({100*state['t']/total:4.1f}%)  recon_calls={state['n']}")
    return total, state["t"]


print("== reconstruction share (post-gather) ==")
run(0.25, capture=True)   # default config; capture real recon inputs
run(1.0)                  # fullres

np.savez("/home/icb/tim.treis/lustre/projects/cp_measure_wt_granularity/recon_inputs.npz",
         **{f"seed{i}": s for i, (s, m) in enumerate(captured)},
         **{f"mask{i}": m for i, (s, m) in enumerate(captured)})
print(f"captured {len(captured)} (seed,mask) recon inputs (default config) -> recon_inputs.npz")
