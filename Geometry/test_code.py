import torch as th

try:
    from .spd_matrices import (
        SPDMatrices,
        SPDPowerCholeskyMetric,
        SPDBuresWassersteinCholeskyMetric,
    )
except ImportError:
    from spd_matrices import (
        SPDMatrices,
        SPDPowerCholeskyMetric,
        SPDBuresWassersteinCholeskyMetric,
    )


def check_point(helper: SPDMatrices, x):
    ok, msg = helper._check_point_on_manifold(x)
    if not ok:
        raise ValueError(f"Point check failed: {msg}")


def check_tangent(helper: SPDMatrices, x, u):
    ok, msg = helper._check_vector_on_tangent(x, u)
    if not ok:
        raise ValueError(f"Tangent check failed: {msg}")


def test_exp_log(metric, helper: SPDMatrices, P, Q, tol=1e-6):
    V = metric.logmap(P, Q)
    check_tangent(helper, P, V)
    Q_rec = metric.expmap(P, V)
    check_point(helper, Q_rec)
    err = (Q_rec - Q).norm() / Q.norm()
    status = "[PASS]" if err < tol else "[FAIL]"
    print(f"{status} exp/log rel error: {err.item():.3e}")


def test_geodesic(metric, helper: SPDMatrices, P, Q, tol=1e-6):
    G0 = metric.geodesic(P, Q, 0.0)
    G1 = metric.geodesic(P, Q, 1.0)
    check_point(helper, G0)
    check_point(helper, G1)
    err0 = (G0 - P).norm()
    err1 = (G1 - Q).norm()
    status0 = "[PASS]" if err0 < tol else "[FAIL]"
    status1 = "[PASS]" if err1 < tol else "[FAIL]"
    print(f"{status0} geodesic at t=0 err={err0.item():.3e}; {status1} geodesic at t=1 err={err1.item():.3e}")


def test_dist(metric, P, Q, tol=1e-6):
    V_P = metric.logmap(P, Q)
    V_Q = metric.logmap(Q, P)
    d = metric.dist(P, Q, is_sqrt=True)
    d_sq = metric.dist(P, Q, is_sqrt=False)
    inner_P = metric.inner(P, V_P, V_P)
    inner_Q = metric.inner(Q, V_Q, V_Q)
    rel_P = (inner_P - d_sq).abs().max() / (d_sq.abs().max() + 1e-12)
    rel_Q = (inner_Q - d_sq).abs().max() / (d_sq.abs().max() + 1e-12)
    statusP = "[PASS]" if rel_P < tol else "[FAIL]"
    statusQ = "[PASS]" if rel_Q < tol else "[FAIL]"
    print(
        f"{statusP}/{statusQ} dist d={d.mean().item():.3e}, d^2={d_sq.mean().item():.3e}, "
        f"<log_PQ,log_PQ> match diff={rel_P.item():.3e}, <log_QP,log_QP> match diff={rel_Q.item():.3e}"
    )


def test_parallel_transport(metric, helper: SPDMatrices, P, Q, tol=1e-6):
    V = helper.random_tangent(shape=(P.shape[0], P.shape[1]), device=P.device, dtype=P.dtype)
    V_t = metric.parallel_transport(P, Q, V)
    V_back = metric.parallel_transport(Q, P, V_t)
    err = (V_back - V).norm() / V.norm()
    # inner product invariance
    inner_before = metric.inner(P, V, V)
    inner_after = metric.inner(Q, V_t, V_t)
    inner_diff = (inner_after - inner_before).abs().max() / (inner_before.abs().max() + 1e-12)
    status_rt = "[PASS]" if err < tol else "[FAIL]"
    status_ip = "[PASS]" if inner_diff < tol else "[FAIL]"
    print(
        f"{status_rt}/{status_ip} pt round-trip err={err.item():.3e}, "
        f"inner before={inner_before.mean().item():.3e}, after={inner_after.mean().item():.3e}, diff={inner_diff.item():.3e}"
    )


def main():
    # settings
    test_mode = "all"  # options: all, exp_log, geodesic, dist, pt
    bs, c, n = 30, 3, 20
    device = "cpu"
    dtype = th.float64
    theta=0.5

    # points and metrics
    pcm = SPDPowerCholeskyMetric(n, theta=theta)
    bwcm = SPDBuresWassersteinCholeskyMetric(n, theta=theta)
    helper = SPDMatrices(n)
    P = helper.random_spd(shape=(bs, c), device=device, dtype=dtype)
    Q = helper.random_spd(shape=(bs, c), device=device, dtype=dtype)

    for metric, name in [
        (pcm, "PCM"),
        (bwcm, "BWCM"),
    ]:
        print(f"\n=== Testing {name} ===")
        check_point(helper, P)
        check_point(helper, Q)
        if test_mode in ("all", "exp_log"):
            test_exp_log(metric, helper, P, Q)
        if test_mode in ("all", "geodesic"):
            test_geodesic(metric, helper, P, Q)
        if test_mode in ("all", "dist"):
            test_dist(metric, P, Q)
        if test_mode in ("all", "pt"):
            test_parallel_transport(metric, helper, P, Q)


if __name__ == "__main__":
    main()
