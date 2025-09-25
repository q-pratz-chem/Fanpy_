"""Implementation of Standard RBM expression by Pratiksha"""

from fanpy.tools import slater, sd_list
from fanpy.wfn.base import BaseWavefunction
import numpy as np
import random
import sys

np.set_printoptions(threshold=sys.maxsize)


class RBM_SoftPlus_Jastrow(BaseWavefunction):
    r"""Restricted Boltzmann Machine–inspired wavefunction with Softplus activation 
    and an explicit Jastrow factor.

    The trial wavefunction is defined as

    .. math::

    \Psi(\mathbf{x}) = \exp\Bigg[
        \sum_{i=1}^{N_h} \operatorname{softplus}
        \bigg( b_i + \sum_{j=1}^{N_v} W_{ij} x_j \bigg)
        + \sum_{j<k} u_{jk} x_j x_k
        + tanh(\sum_{j=1}^{N_v} a_j x_j)
    \Bigg]

    where

    - :math:`\mathbf{x} = (x_1, x_2, \dots, x_{N_v})` is the occupation-number
    representation of the electronic configuration (e.g. :math:`x_j \in \{0,1\}`).
    - :math:`N_v` is the number of visible units (equal to the number of spin orbitals).
    - :math:`N_h` is the number of hidden units.
    - :math:`a_j` are visible biases.
    - :math:`b_i` are hidden biases.
    - :math:`W_{ij}` are weights connecting visible unit :math:`j` to hidden unit :math:`i`.
    - :math:`u_{jk}` are Jastrow coefficients encoding explicit two-body correlations.
    - :math:`\operatorname{softplus}(z) = \log(1 + e^z)` ensures smooth, bounded growth 
    compared to :math:`\cosh(z)` or :math:`\tanh(z)` activations.

    This ansatz combines the flexibility of RBM-style correlations 
    with an explicit Jastrow factor to capture electron–electron interactions 
    and improve stability.
    """

    def __init__(self, nelec, nspin, nhidden, params=None, memory=None, orders=(1)):

        super().__init__(nelec, nspin, memory=memory)
        self.orders = np.array(orders)
        self.nhidden = nhidden
        self._template_params = None
        
        self.output_scale = 1.0  # normalization factor for wavefunction
        self._prev_params = None
        self.iter_count = 0
        self.assign_params(params=params)

        seed = 12345
        random.seed(seed)
        np.random.seed(seed)
        # os.environ["PYTHONHASHSEED"] = str(seed)

    @property
    def params(self):
        return np.hstack([i.flat for i in self._params])

    @property
    def nparams(self):
        return np.sum(self.nspin**self.orders) + self.nhidden + (self.nspin * self.nspin) + (self.nspin * self.nhidden)


    @property
    def params_shape(self):
        # Coefficient matrix for interaction order = 1 with size (nspin)
        # Coefficient matrix for interaction order = 2 with size (nspin, nspin)
        # Coefficient matrix for hidden variables with size (nhidden)
        # weights matrix of size nhidden x nspin
        return [((self.nspin,) * self.orders)] + [(self.nhidden,)] + [(self.nspin, self.nspin)] + [(self.nspin, self.nhidden)]

    @property
    def template_params(self):
        return self._template_params

    @property
    def pspace(self):
        return sd_list.sd_list(self.nelec, self.nspin, num_limit=None, spin=0) #spin=0 : spin restricted sds
    
    @property
    def spin(self):
        return 0
    
    # def init_safe_params(self, hf_init=False, seed=12345):
    def assign_template_params(self, hf_init=False, seed=12345):

        rng = np.random.default_rng(seed)

        Nv = self.nspin
        Nh = self.nhidden
        # base scale depends on visible size (helps keep sums small)
        scale = 1.0 / max(1.0, np.sqrt(Nv))

        # Option A: small random initialization (recommended default)
        sigma_w = 0.005 * scale        # weight stddev
        sigma_a = 1e-4                # visible bias scale
        sigma_b = 1e-4                # hidden bias scale
        sigma_J = 1e-6                # Jastrow scale (small)

        a = rng.normal(0.0, sigma_a, size=(Nv,))
        b = rng.normal(0.0, sigma_b, size=(Nh,))
        W = rng.normal(0.0, sigma_w, size=(Nv, Nh))
        J = rng.normal(0.0, sigma_J, size=(Nv, Nv))
        # make J symmetric and zero diagonal (optional)
        J = 0.5 * (J + J.T)
        np.fill_diagonal(J, 0.0)

        if hf_init:
            # HF-informed residual initialization:
            # Compute HF occupation mask in {-1, +1} format used by the RBM
            occ = slater.occ_indices(slater.ground(self.nelec, self.nspin))
            hf_mask = np.ones(Nv, dtype=float) * -1.0
            hf_mask[occ] = 1.0

            # Choose b so that theta_HF ≈ 0  => b = -W.T @ hf_mask
            # This centers hidden activations at softplus(0)=ln2 for HF.
            b = - (W.T @ hf_mask)

            # Choose a small visible bias to weakly prefer HF (optional)
            a = 1e-2 * hf_mask  # very small HF bias; tweak 0.01–0.2 as needed
        else:
            b = rng.normal(0.0, sigma_b, size=(Nh,))
        # Store template
        self._template_params = [a, b, J, W]
        # self.assign_params(self.template_params)



    # def assign_template_params(self):
    #     params = []

    #     # Initialize visible bias (a) based on mean occupation
    #     # occ_mask = np.zeros(self.nspin, dtype=np.float64)  # Ensure correct dtype for occ_mask

    #     # # Initialize an array to store the sum of occupations for each spin orbital
    #     # occupation_sum = np.zeros(self.nspin, dtype=np.float64)

    #     # # Iterate over all configurations in self.pspace
    #     # for sd in self.pspace:
    #     #     occ_indices = slater.occ_indices(sd)  # Get occupied indices for this configuration
    #     #     occ_mask.fill(0)  # Reset occ_mask for each configuration
    #     #     occ_mask[occ_indices] = 1.0  # Mark the occupied spins
    #     #     occupation_sum += occ_mask  # Add the current configuration's occupation to the sum

    #     # # Compute the mean occupation across all configurations for each spin orbital
    #     # mean_occupation = occupation_sum / len(self.pspace)

    #     # print("Mean occupation per spin orbital:", mean_occupation)

    #     # # Initialize visible bias based on mean occupation
    #     # visible_bias = mean_occupation + np.random.uniform(-0.02, 0.02, size=self.nspin)

    #     # Initialize visible bias (a) based on HF
    #     # occ = slater.occ_indices(slater.ground(self.nelec, self.nspin))
    #     # hf_string = np.ones(self.nspin, dtype=float) * -1.0 
    #     # hf_string[occ] = 1.0
    #     # visible_bias = hf_string + np.random.uniform(-0.02, 0.02, size=self.nspin)
    #     # print("visible_bias (a) initialized to HF + noise:", visible_bias) 
    #     # print("Ground state:", slater.ground(self.nelec, self.nspin), slater.occ_indices(slater.ground(self.nelec, self.nspin)), hf_string)

    #     # Initialize visible bias (a) to small random values
    #     visible_bias = np.random.uniform(-0.01, 0.001, size=self.nspin)
    #     params.append(visible_bias)

    #     # Initialize hidden bias (b) to small random values
    #     # hidden_bias = hf_string + np.random.uniform(-0.02, 0.02, size=self.nhidden)
    #     hidden_bias = np.random.uniform(-0.01, 0.001, size=self.nhidden)
    #     params.append(hidden_bias)
        
    #     # jastrow J (start near zero)
    #     J = np.random.uniform(-1e-4, 1e-4, size=(self.nspin, self.nspin))
    #     # optional: force symmetry initially
    #     J = 0.5 * (J + J.T)
    #     params.append(J)

    #     # Initialize weights (w) to small random values
    #     weights = np.random.uniform(-0.01, 0.001, size=(self.nspin, self.nhidden)) 
    #     params.append(weights)

    #     self._template_params = params


    def assign_params(self, params=None, add_noise=False):
        if params is None:
            if self._template_params is None:
                self.assign_template_params()
            params = self.template_params

        if isinstance(params, np.ndarray):
            structured_params = []
            for param_shape in self.params_shape:
                structured_params.append(params[: np.prod(param_shape)].reshape(*param_shape))
                params = params[np.prod(param_shape) :]
            params = structured_params
            #print("Inside assign_params")

        if self._prev_params is not None:
            # check difference
            deltas = [np.max(np.abs(p - q)) for p, q in zip(params, self._prev_params)]
            delta = max(deltas)
            max_val = max(np.max(np.abs(p)) for p in params)
            # print(f"\n\tIteration {self.iter_count}: max parameter change = {delta}, max parameter value = {max_val}")

            # warn if exploding
            if max_val > 10:
                print(f"⚠️ Parameters exploding beyond 10 at iteration {self.iter_count}!")

        self._prev_params = params.copy()
        self.iter_count += 1
        
        # Invalidate the overlap cache as parameters have changed
        self._overlap_cache = {}
        
        # Reset normalization (must compute by calling normalize() explicitly)
        self.output_scale = 1.0

        # store parameters
        self._params = params


    def softplus(self, x):
        # numerically stable softplus: log(1 + exp(x))
        x = np.asarray(x, dtype=np.float64)
        # use np.log1p(np.exp(x)) — stable for moderate x
        # for large positive x, softplus ~ x, but np.exp(x) may overflow; use where:
        out = np.empty_like(x)
        large = x > 50
        small = x < -50
        mid = ~(large | small)
        out[large] = x[large]  # log1p(exp(x)) ~ x for large x
        out[small] = np.exp(x[small])  # log1p(exp(x)) ~ exp(x) for very negative x (tiny)
        out[mid] = np.log1p(np.exp(x[mid]))
        return out
    

    def log_softplus(self, x):
        # compute log( softplus(x) ) = log( log1p(exp(x)) ) safely
        # softplus(x) is log1p(exp(x)); we take log of that.
        # compute s = softplus(x) then log(s); but better to compute via stable repr:
        s = self.softplus(x)
        # avoid log(0)
        return np.log(np.maximum(s, 1e-300))


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
        return 1.0 / g - g / 3.0 # series approx: 1/g - g/3
    

    def normalize(self, pspace=None):
        """Normalize the RBM wavefunction such that <Psi|Psi> = 1."""
        # Compute all overlaps (no derivatives)
        self.get_overlaps(deriv=None, normalized=False)
        # print("Raw overlaps (unnormalized): ", [self._overlap_cache[sd]['overlap'] for sd in self.pspace])

        if pspace is not None:
            if len(pspace) != len(self._overlap_cache):
                raise ValueError("Provided pspace length does not match cached overlaps length.")
            # Ensure the order of overlaps matches the provided pspace
            # overlaps_dict = {sd: ov for sd, ov in zip(self.pspace, overlaps)}
            overlaps = np.array([self._overlap_cache[sd]['overlap'] for sd in pspace])
        else:
            overlaps = np.array([self._overlap_cache[sd]['overlap'] for sd in self.pspace])            

        norm = np.sqrt(np.sum(np.abs(overlaps) ** 2))

        if norm == 0.0:
            raise ValueError("Wavefunction has zero norm; cannot normalize.")

        # store scalar scale factor applied when returning overlaps
        self.output_scale = 1.0 / norm

        return norm


    def get_overlaps(self, deriv=None, normalized=True):
        """
        Compute & cache RBM overlaps using log-trick and return overlaps or derivatives.
        If normalized=True, returned values are multiplied by self.output_scale (a scalar).
        The cache always stores raw (unnormalized) psi and raw derivatives.
        """

        # ensure float64 arithmetic
        a = np.asarray(self._params[0], dtype=np.float64)  # (Nv,)
        b = np.asarray(self._params[1], dtype=np.float64)  # (Nh,)
        J = np.asarray(self._params[2], dtype=np.float64)  # (Nv, Nv) jastrow
        w = np.asarray(self._params[3], dtype=np.float64)  # (Nv, Nh)
        assert w.shape == (self.nspin, self.nhidden)
        assert J.shape == (self.nspin, self.nspin)

        def check_params(a, b, w, threshold=10.0):
            max_val = max(np.max(np.abs(a)), np.max(np.abs(b)), np.max(np.abs(w)))
            if max_val > threshold:
                print(f"Warning: parameter magnitude {max_val} exceeds threshold {threshold}")
        
        check_params(a, b, w)

        sds = self.pspace
        if len(sds) == 0:
            return np.array([])

        # We'll populate cache entries for all sds requested (or all if deriv requested)
        compute_deriv = deriv is not None

        for sd in sds:
            # If cached and we have what we need, skip
            if sd in self._overlap_cache:
                if (not compute_deriv) or ("derivative" in self._overlap_cache[sd]):
                    continue

            # build occupation vector x in {-1, +1} (our convention)
            occ_mask = np.ones(self.nspin, dtype=np.float64) * -1.0
            occ_mask[slater.occ_indices(sd)] = 1.0  # shape (Nv,)

            # scalar gamma and vector theta
            gamma = float(np.dot(a, occ_mask))          # scalar
            # Note: w is (Nv, Nh), so w.T @ occ_mask -> (Nh,)
            theta = b + (w.T @ occ_mask)               # shape (Nh,)

             # --- Jastrow term (log-space)
            # use 0.5 * n^T J n to avoid double counting for symmetric J
            log_jastrow = 0.5 * float(occ_mask @ (J @ occ_mask))  # scalar

            # --- replace log_sum_coshs with sum log_softplus(theta)
            log_sum_softplus = np.sum(self.log_softplus(theta))

            # combine logs
            logabs_tanh = self.safe_log_abs_tanh(gamma)  # scalar # log |tanh(gamma)|
            logabs_psi = logabs_tanh + log_sum_softplus + log_jastrow  # scalar

            sign_gamma = np.sign(np.tanh(gamma)) if gamma != 0.0 else 1e-300
            psi = sign_gamma * np.exp(logabs_psi)

            # store raw overlap (unnormalized)
            self._overlap_cache[sd] = {"overlap": psi}

            if compute_deriv:
                # raw derivatives d log|psi| / d param
                pref_a = self.safe_dlogtanh_over_dgamma(gamma)  # scalar
                dlog_da = occ_mask * pref_a                      # (Nv,)

                # for softplus: d/dtheta log(softplus(theta)) = softplus'(theta) / softplus(theta)
                # softplus'(x) = 1 / (1 + exp(-x)) = sigmoid(x)
                sigmoid_theta = 1.0 / (1.0 + np.exp(-theta))
                # softplus(theta) = log1p(exp(theta)) but we used its log already; derivative of log(softplus) is:
                dlog_dtheta = sigmoid_theta / np.maximum(np.exp(self.softplus(theta)), 1e-300) 

                dlog_db = dlog_dtheta                      # (Nh,)
                dlog_dW_temp = dlog_db[:, None] * occ_mask[None, :]  # (Nh, Nv)
                dlog_dW = dlog_dW_temp.T  # (Nv, Nh) to match params order

                # Jastrow derivative: d logpsi / d J_ij = 0.5 * n_i * n_j + 0.5 * n_j * n_i = n_i * n_j (since we treat full J)
                dlog_dJ = occ_mask[:, None] * occ_mask[None, :]  # (Nv, Nv)
                # if you store full J this is fine; flatten
                dlog_dJ_flat = dlog_dJ.ravel()

                # derivatives of psi = psi * dlog
                dpsi_da = psi * dlog_da                       # (Nv,)
                dpsi_db = psi * dlog_db                       # (Nh,)
                dpsi_dJ = psi * dlog_dJ_flat                  # (Nv*Nv,)
                dpsi_dW = psi * dlog_dW                       # (Nv, Nh)
        
                # Flatten into [a (Nv,), b (Nh,), W (Nv*Nh,) ] matching params_shape
                derivs_flat = np.hstack([dpsi_da, dpsi_db, dpsi_dJ, dpsi_dW.ravel()])
                self._overlap_cache[sd]["derivative"] = derivs_flat

        # After loop return requested arrays in order of sds
        if not compute_deriv:
            overlaps_raw = np.array([self._overlap_cache[sd]["overlap"] for sd in sds])
            if normalized:
                return overlaps_raw * self.output_scale
            return overlaps_raw
        else:
            derivs_raw =  np.array([self._overlap_cache[sd]["derivative"] for sd in sds])
            if normalized:
                return derivs_raw * self.output_scale
            return derivs_raw
        

    def get_overlap(self, sd, deriv=None, normalized=True):
        """
        Return overlap or derivative for a single Slater determinant `sd`.

        Ensures the cache contains the required data:
        - if only the raw overlap is in cache and derivatives are requested, this
            will call get_overlaps(..., normalized=False) to populate raw derivatives.
        """
        need_deriv = deriv is not None
        
        # If sd missing entirely OR derivatives are requested but not cached, compute them
        if (sd not in self._overlap_cache) or (need_deriv and "derivative" not in self._overlap_cache[sd]):
            # Request raw cache population. Pass normalized=False to avoid circular calls to normalize().
            self.get_overlaps(deriv=None if deriv is None else list(range(self.nparams)))

        # Now the cache must contain requested entries
        if deriv is None:
            # Return the cached raw overlap, scaled if requested
            raw = self._overlap_cache[sd]["overlap"]
            return raw * self.output_scale if normalized else raw
        else:
            # Return the cached raw derivatives for the specified indices, scaled if requested
            raw_deriv = self._overlap_cache[sd]["derivative"][deriv]
            return raw_deriv * self.output_scale if normalized else raw_deriv

        
      