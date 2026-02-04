[<img src="https://img.shields.io/badge/arXiv-2407.02607-b31b1b"></img>](https://arxiv.org/abs/2407.02607)
[<img src="https://img.shields.io/badge/OpenReview|forum-5S8ruWKe8l-8c1b13"></img>](https://openreview.net/forum?id=5S8ruWKe8l)
[<img src="https://img.shields.io/badge/OpenReview|pdf-5S8ruWKe8l-8c1b13"></img>](https://openreview.net/pdf?id=5S8ruWKe8l)

# Fast and Stable Riemannian Metrics on SPD Manifolds via Cholesky Product Geometry

This folder provides implementations of the Power Cholesky Metric (PCM) and the Bures-Wasserstein Cholesky Metric (BWCM) on the SPD manifold.

If you find this project helpful, please consider citing us as follows:

```bib
@inproceedings{chen2026fast,
    title={Fast and Stable Riemannian Metrics on {SPD} Manifolds via Cholesky Product Geometry},
    author={Ziheng Chen and Yue Song and Xiao-Jun Wu and Nicu Sebe},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=5S8ruWKe8l}
}
```

## Quickstart

`example.py` executes `expmap`, `logmap`, `geodesic`, `parallel_transport`, `dist`, and `inner` for both metrics.

## Files

- `spd_matrices.py` — metric implementations.
- `test_code.py` — lightweight consistency tests with PASS/FAIL output (`python -m SPD.test_code`).
- `example.py` — simple usage example (see below) (`python SPD.example`).
- `readme.md` — this file.

## Requirements

- Python with `torch` installed.

## Example Usage

```python
import torch as th

from SPD.spd_matrices import (
    SPDMatrices,
    SPDPowerCholeskyMetric,
    SPDBuresWassersteinCholeskyMetric,
)


def main():
    n = 4
    device = "cpu"
    dtype = th.float64

    helper = SPDMatrices(n)
    pcm = SPDPowerCholeskyMetric(n, theta=1.2)
    bwcm = SPDBuresWassersteinCholeskyMetric(n, theta=0.8)

    P = helper.random_spd(device=device, dtype=dtype)
    Q = helper.random_spd(device=device, dtype=dtype)
    U = helper.random_tangent(device=device, dtype=dtype)

    print("=== PCM ===")
    V_pcm = pcm.logmap(P, Q)
    Q_rec_pcm = pcm.expmap(P, V_pcm)
    print("exp(log) rel err:", (Q_rec_pcm - Q).norm() / Q.norm())
    print("dist:", pcm.dist(P, Q))
    print(
        "geodesic endpoints:",
        pcm.geodesic(P, Q, 0.0).sub(P).norm(),
        pcm.geodesic(P, Q, 1.0).sub(Q).norm(),
    )
    U_pt_pcm = pcm.parallel_transport(P, Q, U)
    print("pt inner before/after:", pcm.inner(P, U, U), pcm.inner(Q, U_pt_pcm, U_pt_pcm))

    print("\n=== BWCM ===")
    V_bw = bwcm.logmap(P, Q)
    Q_rec_bw = bwcm.expmap(P, V_bw)
    print("exp(log) rel err:", (Q_rec_bw - Q).norm() / Q.norm())
    print("dist:", bwcm.dist(P, Q))
    print(
        "geodesic endpoints:",
        bwcm.geodesic(P, Q, 0.0).sub(P).norm(),
        bwcm.geodesic(P, Q, 1.0).sub(Q).norm(),
    )
    U_pt_bw = bwcm.parallel_transport(P, Q, U)
    print("pt inner before/after:", bwcm.inner(P, U, U), bwcm.inner(Q, U_pt_bw, U_pt_bw))


if __name__ == "__main__":
    main()

```

## Notes
The SPD MLRs based on PCM and BWCM have been integrated into the [RMLR](https://github.com/GitZH-Chen/RMLR) repository.