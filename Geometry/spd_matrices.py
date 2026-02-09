import torch as th
import torch.nn as nn
from typing import Optional, Tuple, Union

__all__ = [
    "SPDMatrices",
    "SPDCholeskyProductMetric",
    "SPDPowerCholeskyMetric",
    "SPDBuresWassersteinCholeskyMetric",
]

class SPDMatrices(nn.Module):
    """Computation for SPD data with [..., n, n]."""

    def __init__(self, n: int):
        super().__init__()
        self.n = n
        self.dim = int(n * (n + 1) / 2)

    def _check_point_on_manifold(
        self, x: th.Tensor, *, atol: float = 1e-5, rtol: float = 1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        ok = th.allclose(x, x.transpose(-1, -2), atol=atol, rtol=rtol)
        if not ok:
            return False, "`x != x.transpose` with atol={}, rtol={}".format(atol, rtol)
        e, _ = th.linalg.eigh(x, "U")
        ok = (e > -atol).min()
        if not ok:
            return False, "eigenvalues of x are not all greater than 0."
        return True, None

    def _check_vector_on_tangent(
        self, x: th.Tensor, u: th.Tensor, *, atol: float = 1e-5, rtol: float = 1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        ok = th.allclose(u, u.transpose(-1, -2), atol=atol, rtol=rtol)
        if not ok:
            return False, "`u != u.transpose` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def random_spd(
        self,
        shape: Union[Tuple[int, ...], int] = (),
        device: Optional[th.device] = None,
        dtype: Optional[th.dtype] = th.float64,
        jitter: float = 1e-6,
    ) -> th.Tensor:
        """
        Generate a random SPD matrix with shape [..., n, n].

        shape: leading batch shape (without trailing n,n); accepts int or tuple.
        device: torch device (default cpu); dtype: torch dtype (default float64).
        jitter: small diagonal loading to ensure positive definiteness.
        """
        if isinstance(shape, int):
            shape = (shape,)
        full_shape = (*shape, self.n, self.n)
        A = th.randn(full_shape, device=device, dtype=dtype)
        sym = A @ A.transpose(-1, -2)
        eye = th.eye(self.n, device=device, dtype=dtype)
        return sym + jitter * eye

    def random_tangent(
        self,
        shape: Union[Tuple[int, ...], int] = (),
        device: Optional[th.device] = None,
        dtype: Optional[th.dtype] = th.float64,
    ) -> th.Tensor:
        """
        Generate a random symmetric tangent vector with shape [..., n, n].

        shape: leading batch shape (without trailing n,n); accepts int or tuple.
        device: torch device (default cpu); dtype: torch dtype (default float64).
        """
        if isinstance(shape, int):
            shape = (shape,)
        full_shape = (*shape, self.n, self.n)
        X = th.randn(full_shape, device=device, dtype=dtype)
        return 0.5 * (X + X.transpose(-1, -2))


    def expmap(self, S: th.Tensor, V: th.Tensor) -> th.Tensor:
        """Exponential map at S of V.

        S: [..., n, n] SPD; V: [..., n, n] symmetric; returns [..., n, n] SPD.
        """
        raise NotImplementedError

    def logmap(self, S: th.Tensor, Q: th.Tensor) -> th.Tensor:
        """Logarithm map at S of Q.

        S: [..., n, n] SPD; Q: [..., n, n] SPD; returns [..., n, n] symmetric.
        """
        raise NotImplementedError

    def geodesic(self, S: th.Tensor, Q: th.Tensor, t: Union[float, th.Tensor]) -> th.Tensor:
        """Geodesic between S and Q at time t.

        S: [..., n, n] SPD; Q: [..., n, n] SPD; t: scalar; returns [..., n, n] SPD.
        """
        return self.expmap(S, t * self.logmap(S, Q))

    def parallel_transport(self, S: th.Tensor, Q: th.Tensor, V: th.Tensor) -> th.Tensor:
        """Parallel transport of V at S to Q.

        S: [..., n, n] SPD; Q: [..., n, n] SPD; V: [..., n, n] symmetric; returns [..., n, n] symmetric.
        """
        raise NotImplementedError

    def dist(self, S: th.Tensor, Q: th.Tensor, keepdim: bool = False, is_sqrt: bool = True) -> th.Tensor:
        """Compute geodesic distance between two points.

        S: [..., n, n] SPD; Q: [..., n, n] SPD;
        returns dist (or squared if is_sqrt=False).
        """
        raise NotImplementedError



class SPDCholeskyProductMetric(SPDMatrices):
    """
    Helpers for metrics induced from the Cholesky product geometry.
    The differential of the Cholesky map and its inverse are shared by PCM/BWCM.
    """

    def __init__(self, n: int, theta: float = 1.0, eps: float = 1e-6):
        super().__init__(n)
        self.theta = theta
        self.eps = eps

    # ---- Cholesky map utilities ----
    def _chol(self, S: th.Tensor) -> th.Tensor:
        return th.linalg.cholesky(S)

    def _chol_inverse(self, L: th.Tensor) -> th.Tensor:
        return L @ L.transpose(-1, -2)

    def _strictly_lower(self, X: th.Tensor) -> th.Tensor:
        return X.tril(-1)

    def _diag(self, X: th.Tensor) -> th.Tensor:
        return th.diagonal(X, dim1=-2, dim2=-1)

    def _diag_embed(self, v: th.Tensor) -> th.Tensor:
        return th.diag_embed(v)

    def _diag_func(self, diag: th.Tensor, power: float) -> th.Tensor:
        """
        Diagonal operator for Cholesky metrics; subclasses override to change the diagonal transform.
        """
        return th.pow(diag, power)

    # ---- Cholesky-manifold operators to be implemented by subclasses ----
    def _c_expmap(self, L: th.Tensor, X: th.Tensor) -> th.Tensor:
        raise NotImplementedError

    def _c_logmap(self, L: th.Tensor, K: th.Tensor) -> th.Tensor:
        raise NotImplementedError

    def _c_parallel_transport(self, L: th.Tensor, K: th.Tensor, X: th.Tensor) -> th.Tensor:
        raise NotImplementedError

    def _c_dist(self, L: th.Tensor, K: th.Tensor, keepdim: bool = False, is_sqrt: bool = True) -> th.Tensor:
        raise NotImplementedError

    def _c_inner(self, L: th.Tensor, X: th.Tensor, Y: th.Tensor, keepdim: bool = False) -> th.Tensor:
        """Inner product on the Cholesky manifold; implemented by subclasses."""
        raise NotImplementedError

    def chol_differential(self, L: th.Tensor, V: th.Tensor) -> th.Tensor:
        """
        (D chol_P)[V]: SPD tangent -> Cholesky tangent.
        (D_L P^{-1})(Y) = L * (L^{-1} Y L^{-T})_(1/2)), where _(1/2) keeps
        strictly lower plus half diagonal.
        """
        L_inv = th.linalg.inv(L)
        S = L_inv @ V @ L_inv.transpose(-1, -2)
        lower = self._strictly_lower(S)
        diag = self._diag(S)
        Y_t = lower + 0.5 * self._diag_embed(diag)
        return L @ Y_t

    def chol_inverse_differential(self, L: th.Tensor, X: th.Tensor) -> th.Tensor:
        """(D chol_P)^{-1} inverse: Cholesky tangent -> SPD tangent, (D_L P)(X)=LX^T + X^T L."""
        XLt = X @ L.transpose(-1, -2)
        return XLt + XLt.transpose(-1, -2)

    # ---- SPD operators induced by the Cholesky map ----
    def expmap(self, S: th.Tensor, V: th.Tensor) -> th.Tensor:
        L = self._chol(S)
        X = self.chol_differential(L, V)
        K = self._c_expmap(L, X)
        return self._chol_inverse(K)

    def logmap(self, S: th.Tensor, Q: th.Tensor) -> th.Tensor:
        L = self._chol(S)
        K = self._chol(Q)
        X = self._c_logmap(L, K)
        return self.chol_inverse_differential(L, X)

    def parallel_transport(self, S: th.Tensor, Q: th.Tensor, V: th.Tensor) -> th.Tensor:
        L = self._chol(S)
        K = self._chol(Q)
        X = self.chol_differential(L, V)
        X_pt = self._c_parallel_transport(L, K, X)
        return self.chol_inverse_differential(K, X_pt)

    def dist(self, S: th.Tensor, Q: th.Tensor, keepdim: bool = False, is_sqrt: bool = True) -> th.Tensor:
        L = self._chol(S)
        K = self._chol(Q)
        return self._c_dist(L, K, keepdim=keepdim, is_sqrt=is_sqrt)

    def inner(self, S: th.Tensor, U: th.Tensor, V: th.Tensor = None, keepdim: bool = False) -> th.Tensor:
        """Riemannian inner product at S. Shapes: S,U,V are [..., n, n]."""
        V = U if V is None else V
        L = self._chol(S)
        X = self.chol_differential(L, U)
        Y = self.chol_differential(L, V)
        return self._c_inner(L, X, Y, keepdim=keepdim)


class SPDPowerCholeskyMetric(SPDCholeskyProductMetric):
    """
    Power Cholesky Metric (PCM) on SPD via the Cholesky product geometry (theta-DPM on Cholesky).
    Implements expmap/logmap/geodesic/parallel transport/dist following Table 1 and Eq. (11).
    """

    def __init__(self, n: int, theta: float = 1.0):
        super().__init__(n, theta=theta)

    def _diag_func(self, diag: th.Tensor, power: float) -> th.Tensor:
        return super()._diag_func(diag, power)

    # ---- operators on the Cholesky manifold (theta-DPM) ----
    def _c_geodesic(self, L: th.Tensor, X: th.Tensor, t: th.Tensor) -> th.Tensor:
        off = self._strictly_lower(L) + t * self._strictly_lower(X)
        diag_L = self._diag(L)
        diag_X = self._diag(X)
        diag_factor = 1.0 + t * self.theta * diag_X / diag_L
        diag_factor = th.clamp(diag_factor, min=self.eps)
        diag_geo = diag_L * self._diag_func(diag_factor, 1.0 / self.theta)
        return off + self._diag_embed(diag_geo)

    def _c_expmap(self, L: th.Tensor, X: th.Tensor) -> th.Tensor:
        return self._c_geodesic(L, X, t=1.0)

    def _c_logmap(self, L: th.Tensor, K: th.Tensor) -> th.Tensor:
        off = self._strictly_lower(K) - self._strictly_lower(L)
        diag_L = self._diag(L)
        diag_ratio = self._diag(K) / diag_L
        diag_term = (1.0 / self.theta) * diag_L * (self._diag_func(diag_ratio, self.theta) - 1.0)
        return off + self._diag_embed(diag_term)

    def _c_parallel_transport(self, L: th.Tensor, K: th.Tensor, X: th.Tensor) -> th.Tensor:
        off = self._strictly_lower(X)
        diag_ratio = self._diag(K) / self._diag(L)
        diag_scale = self._diag_func(diag_ratio, 1.0 - self.theta)
        diag_pt = self._diag(X) * diag_scale
        return off + self._diag_embed(diag_pt)

    def _c_dist(self, L: th.Tensor, K: th.Tensor, keepdim: bool = False, is_sqrt: bool = True) -> th.Tensor:
        off_diff = self._strictly_lower(K) - self._strictly_lower(L)
        off_norm_sq = (off_diff * off_diff).sum(dim=(-1, -2))
        diag_diff = self._diag_func(self._diag(K), self.theta) - self._diag_func(self._diag(L), self.theta)
        diag_norm_sq = (diag_diff * diag_diff).sum(dim=-1) / (self.theta ** 2)
        dist_sq = off_norm_sq + diag_norm_sq
        dist = th.sqrt(dist_sq) if is_sqrt else dist_sq
        if keepdim:
            return dist.unsqueeze(-1).unsqueeze(-1)
        return dist

    def _c_inner(self, L: th.Tensor, X: th.Tensor, Y: th.Tensor, keepdim: bool = False) -> th.Tensor:
        # g_L(X,Y) = <\lfloor X \rfloor, \lfloor Y \rfloor> + <diag(L)^{theta-1}[diag X], diag(L)^{theta-1}[diag Y]>
        off_X = self._strictly_lower(X)
        off_Y = self._strictly_lower(Y)
        off_ip = (off_X * off_Y).sum(dim=(-1, -2))

        diag_L = self._diag(L)
        diag_pow = diag_L.pow(self.theta - 1.0)
        diag_ip = (diag_pow * diag_pow * self._diag(X) * self._diag(Y)).sum(dim=-1)
        res = off_ip + diag_ip
        if keepdim:
            return res.unsqueeze(-1).unsqueeze(-1)
        return res


class SPDBuresWassersteinCholeskyMetric(SPDCholeskyProductMetric):
    """
    Bures-Wasserstein Cholesky Metric with (θ, M)-DBWM on the Cholesky manifold.
    Formulas follow Table 1 with diagonal power θ/2 and M weighting on the diagonal part.
    """

    def __init__(self, n: int, theta: float = 1.0):
        super().__init__(n, theta=theta)

    def _diag_func(self, diag: th.Tensor, power: float) -> th.Tensor:
        return th.pow(diag, power)

    # ---- operators on the Cholesky manifold (DBWM) ----
    def _c_geodesic(self, L: th.Tensor, X: th.Tensor, t: th.Tensor) -> th.Tensor:
        off = self._strictly_lower(L) + t * self._strictly_lower(X)
        diag_L = self._diag(L)
        diag_X = self._diag(X)
        diag_factor = 1.0 + t * (self.theta / 2.0) * diag_X / diag_L
        diag_factor = th.clamp(diag_factor, min=self.eps)
        diag_geo = diag_L * self._diag_func(diag_factor, 2.0 / self.theta)
        return off + self._diag_embed(diag_geo)

    def _c_expmap(self, L: th.Tensor, X: th.Tensor) -> th.Tensor:
        return self._c_geodesic(L, X, t=1.0)

    def _c_logmap(self, L: th.Tensor, K: th.Tensor) -> th.Tensor:
        off = self._strictly_lower(K) - self._strictly_lower(L)
        diag_L = self._diag(L)
        diag_ratio = self._diag(K) / diag_L
        diag_term = (2.0 / self.theta) * diag_L * (self._diag_func(diag_ratio, self.theta / 2.0) - 1.0)
        return off + self._diag_embed(diag_term)

    def _c_parallel_transport(self, L: th.Tensor, K: th.Tensor, X: th.Tensor) -> th.Tensor:
        off = self._strictly_lower(X)
        diag_ratio = self._diag(K) / self._diag(L)
        diag_scale = self._diag_func(diag_ratio, 1.0 - self.theta / 2.0)
        diag_pt = self._diag(X) * diag_scale
        return off + self._diag_embed(diag_pt)

    def _c_dist(self, L: th.Tensor, K: th.Tensor, keepdim: bool = False, is_sqrt: bool = True, M: Optional[th.Tensor] = None) -> th.Tensor:
        off_diff = self._strictly_lower(K) - self._strictly_lower(L)
        off_norm_sq = (off_diff * off_diff).sum(dim=(-1, -2))
        diag_diff = self._diag_func(self._diag(K), self.theta / 2.0) - self._diag_func(self._diag(L), self.theta / 2.0)
        if M is None:
            diag_norm_sq = (diag_diff * diag_diff).sum(dim=-1) / (self.theta ** 2)
        else:
            diag_norm_sq = ((diag_diff / th.sqrt(M)) ** 2).sum(dim=-1) / (self.theta ** 2)
        dist_sq = off_norm_sq + diag_norm_sq
        dist = th.sqrt(dist_sq) if is_sqrt else dist_sq
        if keepdim:
            return dist.unsqueeze(-1).unsqueeze(-1)
        return dist

    def _c_inner(self, L: th.Tensor, X: th.Tensor, Y: th.Tensor, keepdim: bool = False, M: Optional[th.Tensor] = None) -> th.Tensor:
        off_X = self._strictly_lower(X)
        off_Y = self._strictly_lower(Y)
        off_ip = (off_X * off_Y).sum(dim=(-1, -2))

        diag_L = self._diag(L)
        diag_L_pow = diag_L.pow(self.theta - 2.0)
        diag_X = self._diag(X)
        diag_Y = self._diag(Y)
        if M is None:
            diag_ip = 0.25 * (diag_L_pow * diag_X * diag_Y).sum(dim=-1)
        else:
            diag_ip = 0.25 * (diag_L_pow * diag_X * (diag_Y / M)).sum(dim=-1)
        res = off_ip + diag_ip
        if keepdim:
            return res.unsqueeze(-1).unsqueeze(-1)
        return res

    def dist(self, S: th.Tensor, Q: th.Tensor, keepdim: bool = False, is_sqrt: bool = True, M: Optional[th.Tensor] = None) -> th.Tensor:
        L = self._chol(S)
        K = self._chol(Q)
        return self._c_dist(L, K, keepdim=keepdim, is_sqrt=is_sqrt, M=M)

    def inner(self, S: th.Tensor, U: th.Tensor, V: th.Tensor = None, keepdim: bool = False, M: Optional[th.Tensor] = None) -> th.Tensor:
        V = U if V is None else V
        L = self._chol(S)
        X = self.chol_differential(L, U)
        Y = self.chol_differential(L, V)
        return self._c_inner(L, X, Y, keepdim=keepdim, M=M)
