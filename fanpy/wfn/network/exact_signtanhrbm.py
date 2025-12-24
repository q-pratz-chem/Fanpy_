"""
Exact sign-corrected RBM wavefunction (no numerical stabilization)

Ψ(x) = sign(tanh(γ)) * exp(a·x) * ∏_i 2 cosh(b_i + W_i·x)

where x_j ∈ {-1, +1}

Author: Pratiksha G.
"""

import numpy as np
from fanpy.tools import slater, sd_list
from fanpy.wfn.base import BaseWavefunction


class SignTanhRBM(BaseWavefunction):
    r"""
    Exact sign-corrected Restricted Boltzmann Machine wavefunction.

    Notes
    -----
    • This implementation is mathematically exact
    • No log tricks, no safe tanh, no overflow protection
    • Intended for small systems and debugging / validation
    """

    def __init__(
        self,
        nelec,
        nspin,
        nhidden,
        scale=1.0,
        pspace_exc_orders=None,
        hf_init=False,
        params=None,
        memory=None,
    ):
        super().__init__(nelec, nspin, memory=memory)

        self.nhidden = nhidden
        self.hf_init = hf_init
        self.init_scale = scale
        self.pspace_exc_orders = pspace_exc_orders

        self.output_scale = 1.0
        self._template_params = None

        # P-space containers (O(1) lookup)
        self._pspace_list = None
        self._pspace_set = None
        self._pspace_index = None
        self.X = None
        self.n_sds = None

        self._pspace_overlaps = None
        self._pspace_derivs = None

        self._build_pspace_matrix()
        self.assign_params(params)

    # ----------------------------
    #  Properties
    # ----------------------------
    @property
    def spin(self):
        return 0

    @property
    def pspace(self):
        return sd_list.sd_list(
            self.nelec,
            self.nspin,
            exc_orders=self.pspace_exc_orders,
            spin=self.spin,
        )

    @property
    def params_shape(self):
        return [
            (self.nspin,),              # a
            (self.nhidden,),            # b
            (self.nspin, self.nhidden), # W
        ]

    @property
    def nparams(self):
        return sum(np.prod(s) for s in self.params_shape)

    @property
    def params(self):
        return np.concatenate([p.ravel() for p in self._params])

    # ----------------------------
    #  Build P-space matrix
    # ----------------------------
    def _build_pspace_matrix(self):
        pspace = list(self.pspace)
        self._pspace_list = pspace
        self._pspace_set = set(pspace)
        self._pspace_index = {sd: i for i, sd in enumerate(pspace)}

        n = len(pspace)
        X = -np.ones((n, self.nspin))
        for i, sd in enumerate(pspace):
            X[i, slater.occ_indices(sd)] = 1.0

        self.X = X
        self.n_sds = n

    # ----------------------------
    #  Initialization
    # ----------------------------
    def assign_template_params(self, seed=12345):
        rng = np.random.default_rng(seed)

        Nv, Nh = self.nspin, self.nhidden

        # Xavier scaling for tanh
        limit = np.sqrt(self.init_scale / (Nv + Nh))

        W = rng.normal(-limit, limit, size=(Nv, Nh))
        a = rng.normal(-limit, limit, size=Nv)
        b = np.zeros(Nh)

        #a = rng.normal(0.0, 0.01, size=Nv)
        #b = rng.normal(0.0, 0.01, size=Nh)
        #W = rng.normal(0.0, 0.01, size=(Nv, Nh))

        if self.hf_init:
            x_ref = self.X[0]
            b = -(W.T @ x_ref)
            a = 0.01 * x_ref

        self._template_params = [a, b, W]

    # ----------------------------
    #  Assign parameters
    # ----------------------------
    def assign_params(self, params=None):
        if params is None:
            if self._template_params is None:
                self.assign_template_params()
            params = self._template_params

        if isinstance(params, np.ndarray):
            structured = []
            idx = 0
            for shape in self.params_shape:
                n = np.prod(shape)
                structured.append(params[idx:idx+n].reshape(shape))
                idx += n
            params = structured

        self._params = [np.array(p, dtype=float) for p in params]
        self._pspace_overlaps = None
        self._pspace_derivs = None
        self.output_scale = 1.0

    # ----------------------------
    #  Overlaps (exact RBM)
    # ----------------------------
    def get_overlaps(self, deriv=True):
        a, b, W = self._params
        X = self.X

        # γ = a · x
        gamma = X @ a                        # (n_sds,)

        # θ_i = b_i + W_i · x
        theta = b + X @ W                   # (n_sds, nhidden)

        # amplitude = exp(a·x) * ∏ 2 cosh(theta)
        amp = np.exp(gamma) * np.prod(2.0 * np.cosh(theta), axis=1)

        # sign = sign(tanh(gamma))
        sign = np.sign(np.tanh(gamma))
        sign[gamma == 0.0] = 1.0

        psi = sign * amp

        # normalization
        norm = np.linalg.norm(psi)
        if norm == 0.0:
            raise ValueError("Zero-norm RBM wavefunction")

        self.output_scale = 1.0 / norm
        self._pspace_overlaps = psi

        if not deriv:
            return

        # ---- derivatives ----
        # d/d a_j log Ψ = x_j
        dlog_da = X                          # (n_sds, nspin)

        # d/d b_i log Ψ = tanh(theta_i)
        dlog_db = np.tanh(theta)             # (n_sds, nhidden)

        # d/d W_{ji} log Ψ = x_j tanh(theta_i)
        dlog_dW = X[:, :, None] * dlog_db[:, None, :]  # (n_sds, nspin, nhidden)

        derivs = np.hstack([
            psi[:, None] * dlog_da,
            psi[:, None] * dlog_db,
            (psi[:, None, None] * dlog_dW).reshape(self.n_sds, -1),
        ])

        self._pspace_derivs = derivs

    # ----------------------------
    #  Single overlap (O(1))
    # ----------------------------
    def get_overlap(self, sd, deriv=None, normalized=True):
        if sd not in self._pspace_set:
            if deriv is None:
                return 0.0
            return np.zeros(len(deriv)) if np.ndim(deriv) else 0.0

        if (
            self._pspace_overlaps is None
            or (deriv is not None and self._pspace_derivs is None)
        ):
            self.get_overlaps(deriv=True)

        idx = self._pspace_index[sd]
        val = self._pspace_overlaps[idx]

        if deriv is None:
            return val * self.output_scale if normalized else val

        d = self._pspace_derivs[idx, deriv]
        return d * self.output_scale if normalized else d


