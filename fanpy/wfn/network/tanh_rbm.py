"""Implementation of Standard RBM expression by Pratiksha"""

from fanpy.tools import slater, sd_list
from fanpy.wfn.base import BaseWavefunction
import numpy as np
import random
import sys

np.set_printoptions(threshold=sys.maxsize)


class tanhRBM(BaseWavefunction):
    r"""Restricted Boltzmann Machine (RBM) as a Wavefunction
    Expression
    ----------
    ..math::
        \Psi(\textbf{x}) = sign(tanh(\gamma)) tanh(gamma) \prod_{i=1}^{N_h} 2 cosh(\theta_i),
            \text{where} \gamma = \sum_j a_j x_j, \quad \theta_i = b_i + \sum_{i,j} w_{i,j}x_j

    Using the probability distribution representation by RBM as a wavefunction.

    Attributes
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of total spin orbitals (including occupied and virtual, alpha and beta).
    bath : np.array
        Hidden variables (here, spin orbitals) for RBM.
    params : np.array
        Parameters of the RBM without including parameters for sign correction.
    memory : float
        Memory available for the wavefunction.
    orders : np.array
        Orders of interaction considered in the virutal variables i.e. spin orbitals.
        Interaction term with order = 1 :
            \sum_i a_i n_i,
                where a_i : coefficients, n_i : occupation number for spin orbital i.
        Interaction term with order = 2 :
            \sum_{i, j} a_{ij} n_i n_j,
                where a_i : coefficients,
                    n_i, n_j  : occupation number for spin orbitals i, j, respectively.

    """

    def __init__(
        self, 
        nelec, 
        nspin, 
        nhidden, 
        scale=1.0, 
        pspace_exc_orders=[1,2],
        hf_init=False,
        params=None, 
        memory=None, 
    ):
        super().__init__(nelec, nspin, memory=memory)

        self.nhidden = nhidden
        self.init_scale = scale
        self.pspace_exc_orders = pspace_exc_orders
        self._pspace_list = None
        self._pspace_set = None
        self._pspace_index = None
        self.output_scale = 1.0  # normalization factor for wavefunction

        self._template_params = None
        
        self._prev_params = None
        self.iter_count = 0

        self._pspace_overlaps = None
        self._pspace_derivs = None
        self.tanh1 = np.tanh(1.0) 
      
        self.hf_init = hf_init
        self.assign_params(params=params)



    # ============================================================
    # Parameter bookkeeping
    # ============================================================

    @property
    def params_shape(self):
        return [
            (self.nspin,),               # a
            (self.nhidden,),             # b
            (self.nspin, self.nhidden),  # w
        ]


    @property
    def nparams(self):
        return np.sum(self.nspin) + self.nhidden + (self.nspin * self.nhidden)


    @property
    def params(self):
        return np.hstack([i.flat for i in self._params])

    @property
    def spin(self):
        return 0

    @property
    def pspace(self):
        return sd_list.sd_list(self.nelec, self.nspin, exc_orders=self.pspace_exc_orders, num_limit=None, spin=0)
        


    # ---------- P-space matrix ------------------
    def _build_pspace_matrix(self):
        """Vectorized occupation vector representation of all Slater determinants in {-1, +1} encoding."""
        pspace = list(self.pspace)
        n = len(pspace)
        X = np.ones((n, self.nspin)) * -1
        for i, sd in enumerate(pspace):
            X[i, slater.occ_indices(sd)] = 1
            
        self.X = X             # (n_sds, nspin)
        self.n_sds = n
        


    # ============================================================
    # Xavier + HF-centered initialization
    # ============================================================


    def assign_template_params(self, hf_init=False, seed=12345):

        print("\nAssigning Xavier-consistent template parameters...")

        rng = np.random.default_rng(seed)
        Nv, Nh = self.nspin, self.nhidden

        # Xavier for tanh
        limit = np.sqrt(self.init_scale / np.sqrt(Nh +Nv))

        w = rng.normal(-limit, limit, size=(Nv, Nh)) # np.zeros((Nv, Nh))
        a = rng.normal(-limit, limit, size=(Nv, )) # np.zeros((Nv, ))
        b = np.zeros(Nh)


        if self.hf_init:
            # HF-informerd residual initialization
            occ = slater.occ_indices(slater.ground(self.nelec, self.nspin))
            x_ref = -np.ones(Nv)
            x_ref[occ] = 1.0

            # Choose b so that theta_HF = 0 => b = - w^T @ hf_mask
            # This centers hidden activations around zero for the HF config
            b = - (w.T @ x_ref)

            # Choose a small visible bias to weakly prefer HF (optional)
            a = 0.05 * x_ref      # very small HF bias; tweak 0.01-0.2 as needed
            
        # Store templated params
        self._template_params = [a, b, w]



    # ============================================================
    # Assign parameters
    # ============================================================


    def assign_params(self, params=None, add_noise=False):
        if params is None:
            if self._template_params is None:
                self.assign_template_params()
            params = self.template_params

        if isinstance(params, np.ndarray):
            structured = []
            idx = 0
            for shape in self.params_shape:
                n = np.prod(shape)      
                structured.append(params[idx:idx+n].reshape(shape))
                idx += n
        else: 
            structured = [np.array(p, float) for p in params]

        self._params = structured

        if self._prev_params is not None:
            # check difference
            deltas = [np.max(np.abs(p - q)) for p, q in zip(params, self._prev_params)]
            delta = max(deltas)
            max_val = max(np.max(np.abs(p)) for p in params)
            # print(f"\n\tIteration {self.iter_count}: max parameter change = {delta}, max parameter value = {max_val}")

            # warn if exploding
            # if max_val > 10:
            #     print(f"⚠️ Parameters exploding beyond 10 at iteration {self.iter_count}!")

        self._prev_params = params.copy()
        self.iter_count += 1
        
        # invalidate caches
        self._pspace_overlaps = None 
        self._pspace_derivs = None
        # Reset normalization (must compute by calling normalize() explicitly)
        self.output_scale = 1.0


    # ============================================================
    # Numerically stable helpers
    # ============================================================


    def log2cosh(self, t):
        """Stable evaluation of log(2*cosh(t)) elementwise."""
        t = np.asarray(t, dtype=np.float64)
        at = np.abs(t)
        # at + log(1 + exp(-2|t|)) is stable for large |t|
        return at + np.log1p(np.exp(-2.0 * at))


    def safe_log_abs_tanh(self, gamma):
        """Return log|tanh(gamma)| in a numerically stable way (scalar gamma)."""
        g = float(gamma)
        # For large |g|, tanh(g) -> ±1, log|tanh| -> 0
        # For small |g|, tanh(g) ~ g -> log|tanh| ~ log|g|
        abs_g = abs(g)
        if abs_g > 1e-6:
            return np.log(abs(np.tanh(g)))
        # Use series / fallback to log(|g|) with tiny safety offset
        return np.log(abs_g + 1e-300)


    def safe_dlogtanh_over_dgamma(self, gamma):
        """
        Compute (1 - tanh^2(gamma)) / tanh(gamma) safely for scalar gamma.
        This equals d/dgamma log|tanh(gamma)|.
        For small gamma use series expansion: tanh(g) = g - g^3/3 + ...
        (1 - tanh^2)/tanh ≈ 1/g - g/3.
        """
        g = float(gamma)
        tg = np.tanh(g)
        abs_g = abs(g)
        if abs_g > 1e-6 and tg != 0.0:
            return (1.0 - tg * tg) / tg
        # small gamma: use series approx
        if abs_g < 1e-300:
            # avoid division by zero: return large value consistent with 1/g behavior
            return 1.0 / (g + 1e-300)
        # series approx: 1/g - g/3
        return 1.0 / g - g / 3.0
   
 
    # ============================================================
    # Vectorized overlaps + derivatives
    # ============================================================


    def get_overlaps(self, deriv=None, normalized=True):
        """
        Compute & cache RBM overlaps using log-trick and return overlaps or derivatives.
        If normalized=True, returned values are multiplied by self.output_scale (a scalar).
        The cache always stores raw (unnormalized) psi and raw derivatives.
        """
        a, b, w = self._params 
        # ---------- forward ----------
        gamma = self.X @ a                       # (n_sds,)
        theta = b + self.X @ W                   # (n_sds, Nh)

        logabs = (
            self.safe_log_abs_tanh(gamma)
            + np.sum(self.log2cosh(theta), axis=1)
        )

        sign = np.sign(np.tanh(gamma))
        sign[gamma == 0.0] = 1.0

        # ---------- log-sum-exp normalization ----------
        Lmax = np.max(logabs)
        s = sign * np.exp(logabs - Lmax)
        norm = np.linalg.norm(s)
        self.output_scale = 1.0 / norm

        # ---------- derivatives ----------
        pref = self.safe_dlogtanh(gamma)                 # (n_sds,)
        dlog_da = self.X * pref[:, None]                 # (n_sds, Nv)
        dlog_db = np.tanh(theta)                         # (n_sds, Nh)
        dlog_dW = self.X[:, :, None] * dlog_db[:, None, :]  # (n_sds, Nv, Nh)

        derivs = np.hstack([
            s[:, None] * dlog_da,
            s[:, None] * dlog_db,
            (s[:, None, None] * dlog_dW).reshape(self.n_sds, -1),
        ])

        self._pspace_overlaps = s
        self._pspace_derivs = derivs


    # ============================================================
    # O(1) overlap lookup
    # ============================================================


    def get_overlap(self, sd, deriv=None, normalized=True):
        """
        Return overlap or derivative for a single Slater determinant `sd`.

        Ensures the cache contains the required data:
        - if only the raw overlap is in cache and derivatives are requested, this
            will call get_overlaps(..., normalized=False) to populate raw derivatives.
        """
        if sd not in self._pspace_set:
            return 0.0 if deriv is None else np.zeros(len(deriv))

        if self._pspace_overlaps is None:
            self.get_overlaps()

        idx = self._pspace_index[sd]
        raw = self._pspace_overlaps[idx]

        if deriv is None:
            return raw * self.output_scale if normalized else raw

        return (
            self._pspace_derivs[idx, deriv] * self.output_scale
            if normalized
            else self._pspace_derivs[idx, deriv]
        ) 


        
      
