"""Verify the scale==1 degeneracy claim against the real reference search.

Claim: for scale_max == 1 (the float-dtype path), the Costes search result is
closed-form and data-independent:
  - bisection_costes -> (thr_fi_c, thr_si_c) == (0.0, b)
  - linear_costes    -> (thr_fi_c, thr_si_c) == (1.0, a + b)
where a, b is the orthogonal-regression line. If true across many random objects,
a `scale == 1` short-circuit skips the (dead) iterative search while staying exact.
"""

import numpy as np

import cp_measure.core.measurecolocalization as ref


def _obj(seed, n=300):
    rng = np.random.default_rng(seed)
    fi = rng.random(n)
    si = np.clip(0.6 * fi + 0.4 * rng.random(n), 0.0, 1.0)
    return fi, si


def _ab(fi, si):
    nz = (fi > 0) | (si > 0)
    xvar = np.var(fi[nz], ddof=1)
    yvar = np.var(si[nz], ddof=1)
    zvar = np.var(fi[nz] + si[nz], ddof=1)
    covar = 0.5 * (zvar - (xvar + yvar))
    a = ((yvar - xvar) + np.sqrt((yvar - xvar) ** 2 + 4 * covar**2)) / (2 * covar)
    b = si[nz].mean() - a * fi[nz].mean()
    return a, b


def main():
    bis_ok = lin_fast_ok = lin_acc_ok = True
    for seed in range(200):
        fi, si = _obj(seed)
        a, b = _ab(fi, si)
        bf, bs = ref.bisection_costes(fi, si, fi, si, 1)
        if not (bf == 0.0 and bs == b):
            bis_ok = False
            if seed < 3:
                print(f"bisection seed {seed}: got ({bf},{bs}) want (0,{b})")
        lf, ls = ref.linear_costes(fi, si, fi, si, 1, fast_costes="Fast")
        if not (lf == 1.0 and ls == a + b):
            lin_fast_ok = False
            if seed < 3:
                print(f"linear-fast seed {seed}: got ({lf},{ls}) want (1,{a + b})")
        af, as_ = ref.linear_costes(fi, si, fi, si, 1, fast_costes="Accurate")
        if not (af == 1.0 and as_ == a + b):
            lin_acc_ok = False
            if seed < 3:
                print(f"linear-acc seed {seed}: got ({af},{as_}) want (1,{a + b})")

    print(f"\nbisection (0, b)      holds across 200 seeds: {bis_ok}")
    print(f"linear-Fast (1, a+b)  holds across 200 seeds: {lin_fast_ok}")
    print(f"linear-Accurate(1,a+b) holds across 200 seeds: {lin_acc_ok}")


if __name__ == "__main__":
    main()
