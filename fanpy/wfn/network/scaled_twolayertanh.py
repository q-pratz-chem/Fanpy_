"""Scaled two-layer tanh neural network wavefunction with Xavier initialization.
   Ψ(x) = tanh(W2 · tanh(W1x + b) + c) / tanh(1)
   Author: Pratiksha G. (2025)
"""

import numpy as np
from fanpy.tools import slater, sd_list
from fanpy.wfn.base import BaseWavefunction


class ScaledTwoLayerTanhWfn(BaseWavefunction):
    r"""
    Two-layer tanh neural network wavefunction:

        h = tanh(W1 x + b)
        z = W2 h + c
        Ψ(x) = tanh(z) / tanh(1)

    Xavier-initialized parameters.
    """

    def __init__(self, nelec, nspin, nhidden, params=None, memory=None):
        super().__init__(nelec, nspin, memory=memory)
        self.nhidden = nhidden
        self.output_scale = 1.0
        self._template_params = None
        self._overlap_cache = {}
        self.assign_params(params)
        self.tanh1 = np.tanh(1.0)

    # ---------- parameter bookkeeping ----------
    @property
    def params_shape(self):
        return [
            (self.nhidden, self.nspin),  # W1
            (self.nhidden,),             # b
            (1, self.nhidden),           # W2
            (1,),                        # c
        ]

    @property
    def nparams(self):
        return sum(np.prod(s) for s in self.params_shape)

    @property
    def params(self):
        return np.hstack([p.ravel() for p in self._params])

    # ---------- Xavier initialization ----------
    def assign_template_params(self, seed=12345):
        rng = np.random.default_rng(seed)
        Nv, Nh = self.nspin, self.nhidden

        # Xavier uniform for W1
        limit_W1 = np.sqrt(6 / (Nv + Nh))
        W1 = rng.uniform(-limit_W1, limit_W1, size=(Nh, Nv))
        b = np.zeros(Nh)

        # Xavier uniform for W2 (1 output neuron)
        limit_W2 = np.sqrt(6 / (Nh + 1))
        W2 = rng.uniform(-limit_W2, limit_W2, size=(1, Nh))
        c = np.zeros(1)

        self._template_params = [W1, b, W2, c]

    def assign_params(self, params=None):
        if params is None:
            if self._template_params is None:
                self.assign_template_params()
            params = self._template_params
        if isinstance(params, np.ndarray):
            structured = []
            for shape in self.params_shape:
                n = int(np.prod(shape))
                structured.append(params[:n].reshape(*shape))
                params = params[n:]
            params = structured
        self._params = [np.array(p, copy=True) for p in params]
        self._overlap_cache = {}

    # ---------- helpers ----------
    @staticmethod
    def safe_tanh(x):
        """Stable tanh avoiding NaNs for large |x|."""
        return np.tanh(np.clip(x, -30, 30))

    @property
    def pspace(self):
        return sd_list.sd_list(self.nelec, self.nspin, spin=0)

    # ---------- main overlap computation ----------
    def get_overlaps(self, deriv=None): #  normalized=True
        W1, b, W2, c = self._params
        Nv, Nh = self.nspin, self.nhidden

        sds = self.pspace
        n_sds = len(sds)
        overlaps = np.empty(n_sds)
        derivs = np.empty((n_sds, self.nparams))

        for idx, sd in enumerate(sds):
            x = np.ones(Nv) * -1
            x[slater.occ_indices(sd)] = 1

            # Forward pass
            h = self.safe_tanh(W1 @ x + b)  # (Nh,)
            z = (W2 @ h + c).item()           # scalar
            tanh_z = np.tanh(np.clip(z, -30, 30))
            psi = tanh_z / self.tanh1
            overlaps[idx] = psi

            # Derivatives
            sech2_z = (1 - tanh_z**2) / self.tanh1
            sech2_h = 1 - h**2

            dpsi_dW2 = sech2_z * h
            dpsi_dc = sech2_z
            dpsi_dh = sech2_z * W2.flatten()
            dpsi_dW1 = (dpsi_dh * sech2_h)[:, None] * x[None, :]
            dpsi_db = dpsi_dh * sech2_h

            derivs[idx] = np.hstack([
                dpsi_dW1.ravel(),
                dpsi_db,
                dpsi_dW2.ravel(),
                dpsi_dc,
            ])

            self._overlap_cache[sd] = {"overlap": psi, "derivative": derivs[idx]}

        norm = np.sqrt(np.sum(overlaps**2))
        self.output_scale = 1.0 / (norm if norm > 1e-12 else 1.0)


    def get_overlap(self, sd, deriv=None, normalized=True):
        if sd not in self._overlap_cache:
            self.get_overlaps()
        raw = self._overlap_cache[sd]["overlap"]
        if deriv is None:
            return raw * self.output_scale if normalized else raw
        raw_deriv = self._overlap_cache[sd]["derivative"][deriv]
        return raw_deriv * self.output_scale if normalized else raw_deriv
