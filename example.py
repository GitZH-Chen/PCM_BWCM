import torch as th

from .spd_matrices import (
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
