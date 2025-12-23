"""Scaled two-layer tanh neural network wavefunction with Xavier initialization.
   Ψ(x) = tanh(W2 · tanh(W1x + b) + c) / tanh(1)
   Author: Pratiksha G. (2025)
"""

import numpy as np
from fanpy.tools import slater, sd_list
from fanpy.wfn.base import BaseWavefunction
import torch


class ScaledTwoLayerTanhWfn(BaseWavefunction):
    r"""
    Two-layer tanh neural network wavefunction:

        h = tanh(W1 x + b)
        z = W2 h + c
        Ψ(x) = tanh(z) / tanh(1)

    where x is the occupation vector of a Slater determinant encoded in
    {-1, +1}. No bias parameters are used. The output is normalized over the 
    pspace of determinants.

    The parameter ``num_layers`` counts the total number of layers
    excluding the input layer and including the output layer.

    Attributes
    ----------
    X : np.ndarray
        Matrix of occupation vectors for all Slater determinants in the P-space,
        encoded in {-1, +1}. Shape (n_sds, nspin).
    output_scale : float
        Normalization factor applied to the wavefunction amplitudes.

    Properties
    ----------
    nparams : int
        Total number of variational parameters (weights).
    params_shape : list of tuple of int
        Shapes of the individual weight matrices for each layer.
    spin : int
        Spin of the wavefunction.
    pspace : list of int
        List of Slater determinants included in the P-space.



    Methods
    -------
    __init__(self, nelec, nspin, nhidden, num_layers, scale=1.0,
             pspace_exc_orders=None, params=None, memory=None)
        Initialize the neural network wavefunction.
    assign_template_params(self, seed=12345)
        Construct default Xavier-initialized weights.
    assign_params(self, params=None)
        Assign and structure the wavefunction parameters.
    get_overlaps(self, deriv=True)
        Compute overlaps of all P-space Slater determinants with the
        wavefunction and optionally their derivatives.
    get_overlap(self, sd, deriv=None, normalized=True)
        Compute the overlap of a single Slater determinant with the
        wavefunction.
    """

    def __init__(
        self, 
        nelec, 
        nspin, 
        nhidden, 
        scale=1.0, 
        pspace_exc_orders=None, 
        hf_init=False, 
        hf_mo_coeff=None, 
        params=None, 
        memory=None
    ):
        """
        Parameters
        ----------
        nelec : int
            Number of electrons.
        nspin : int
            Number of spin orbitals (alpha + beta).
        params : np.ndarray
            Flattened array of neural network weights.
        memory : float
            Memory available for the wavefunction.
        nhidden : int
            Number of hidden units in each hidden layer.
        init_scale : float
            Scaling factor used in Xavier initialization of the weights.
        pspace_exc_orders : tuple of int or None
            Allowed excitation orders defining the P-space of Slater determinants.
        """        
        super().__init__(nelec, nspin, memory=memory)
        self.nhidden = nhidden
        self.init_scale = scale
        self.output_scale = 1.0
        self._template_params = None
        
        self.pspace_exc_orders=pspace_exc_orders
        self._pspace_list = None
        self._pspace_set = None
        self._pspace_index = None
        self._pspace_overlaps = None
        self._pspace_derivs = None
        
        self.tanh1 = np.tanh(1.0) 
        
        # build determinant matrix ONCE
        self._build_pspace_matrix()
        
        # self._overlap_cache = {}
        self.hf_init = hf_init
        self.hf_mo_coeff = hf_mo_coeff
        self.assign_params(params=params)
              
        
    

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
        return np.concatenate([p.ravel() for p in self._params])

    @property
    def spin(self):
        return 0 
    
    @property
    def pspace(self):
       # return sd_list.sd_list(self.nelec, self.nspin, spin=self.spin)
        return sd_list.sd_list(self.nelec, self.nspin, exc_orders=self.pspace_exc_orders, spin=self.spin)
    
    # ---------- helpers ----------
    @staticmethod
    def safe_tanh(x):
        """Stable tanh avoiding NaNs for large |x|."""
        return np.tanh(np.clip(x, -30, 30))

    
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
        

    # ---------- Xavier initialization ----------
    def assign_template_params(self, seed=12345):
        rng = np.random.default_rng(seed)
        
        #rng = np.random.default_rng(seed)
        Nv, Nh = self.nspin, self.nhidden
        print(f"\nBuilding wavefunction space with excitation orders = {self.pspace_exc_orders}")
        print(f"Number of hidden units = {int(self.nhidden/self.nspin)}.nspin = {self.nhidden}")
        
        # Xavier uniform for W1
        print(f"\nWeights initalized using Xavier uniform initialization. scale = {self.init_scale}")
        limit_W1 = np.sqrt(self.init_scale / (Nv + Nh))
        W1 = rng.uniform(-limit_W1, limit_W1, size=(Nh, Nv))
        b = np.zeros(Nh)

        # Xavier uniform for W2 (1 output neuron)
        limit_W2 = np.sqrt(self.init_scale / (Nh + 1))
        W2 = rng.uniform(-limit_W2, limit_W2, size=(1, Nh))
        c = np.zeros(1)
        
        if self.hf_init:
            print("\tInitializing parameters so HF reference SD has highest amplitude.\n")
            # ---- Stragy 1: Make the ANN output the largest amplitude for the HF reference determinant
            # 1) HF occupation vector
            # 2) Make W1 small so hidden activations are small for all SDs
            # 4) Hidden layer is ~0 => output is tanh(c)
            # 3) Choose b so that W1 @ x_ref + b = 0  => h_ref = tanh(0) = 0
            # 5) Make W2 small to avoid large variance for excited SDs
            # set c > 0 so HF amplitude is the largest
            
            # x_ref = self.X[0]   # shape (Nv,)            
            # W1 *= 0.1
            # b = -(W1 @ x_ref)         
            # c = np.array([0.5])   # gives tanh(0.5) ≈ 0.46
            # W2 *= 0.1
            
            # ---- Strategy 3: HF orbital coefficient matrix C
            # -------------------------------------------------------
            # 1. Build spin-orbital MO coefficient matrix
            #    mo_coeff: (n_ao, n_mo) → we want (nspin, n_mo)
            # -------------------------------------------------------
            C = self.hf_mo_coeff           # (n_ao, n_mo)
            Cmo = C.T                      # (n_mo, n_ao)

            # expand to α/β spin blocks
            Cspin = np.vstack([Cmo, Cmo])  # (2*n_mo = nspin, n_ao)

            # Normalize each column so hidden activations aren’t large
            Cspin = Cspin / np.linalg.norm(Cspin, axis=0, keepdims=True)

            # -------------------------------------------------------
            # 2. Initialize W1 using MO patterns
            # -------------------------------------------------------
            # let each hidden node be a random linear combination of MO columns
            coeff = rng.normal(scale=0.1, size=(Nh, Cspin.shape[1]))  # mixing coefficients
            W1 = coeff @ Cspin.T   # shape (Nh, nspin)

            # -------------------------------------------------------
            # 3. Bias chosen so HF reference has small activation
            # -------------------------------------------------------
            x_ref = self.X[0] # shape nspin
            b = -(W1 @ x_ref)                  # makes h_ref ≈ 0
            
            c  = np.array([0.3])   # HF amplitude tanh(0.3)/tanh(1)
        
        self._template_params = [W1, b, W2, c]

        
    def assign_params(self, params=None):
        # print("Assigning params...")
        if params is None:
            if self._template_params is None:
                self.assign_template_params()
            params = self._template_params
        
        # numpy  -> torch conversion if needed
        
        if isinstance(params, (list, tuple)):
            structured = [np.array(p, dtype=float) for p in params]
        else:
            # flat numpy array -> structured torch params
            params = np.array(params, dtype=float)
            structured = []
            idx = 0            
            for shape in self.params_shape:
                n_ = np.prod((shape))
                block = params[idx:idx+n_].reshape(shape)
                structured.append(block)
                idx += n_
                
        self._params = structured
        
        # Build pspace index and invalidate caches
        # self._overlap_cache = {}
        self._pspace_list = self.pspace # ordered list 
        self._pspace_set = set(self._pspace_list) # O(1) membership
        # map sd -> index in arrays (used in direct indexing later)
        self._pspace_index = {sd: idx for idx, sd in enumerate(self._pspace_list)}
        # prepare empty arrays so integrate_sd_wfn can index without calling get_overlap
        self._pspace_overlaps = None # will be filled by get_overlaps()
        self._pspace_derivs = None # shape (n_sds, nparams)
    
    
    # ---------- main overlap computation ----------
    def get_overlaps(self, deriv=True): #  normalized=True
        """
        Computes ψ(x_i) for all determinants in a single batched GPU pass.
        Also computes derivatives wrt all parameters (optional).
        """

        # import time; time0 = time.time()
        W1, b, W2, c = self._params   # shapes: (Nh,Nv), (Nh,), (1,Nh), (1,)
        
        n_sds = self.n_sds
        
        # ---- Forward (vectorized) ----
        # X : (n_sdds, Nv)
        
        h = self.safe_tanh(self.X @ W1.T + b)       # (n_sds, Nh)
        z = (h @ W2.T).reshape(-1) + c[0]              # (n_sds, )
        tanh_z = self.safe_tanh(z)                  # (n_sds, )
        overlaps = tanh_z / self.tanh1
        
        
        # ---- Backward (Vectorized derivatives) ----
        Nh, Nv = self.nhidden, self.nspin
        sech2_z = (1 - tanh_z**2) / self.tanh1       # (n_sds, )
        sech2_h = 1 - h**2                           # (n_sds, Nh)
        
        # W2 derivatives
        # dpsi/dW2_j = sech2_z * h_j
        dW2 = (sech2_z[:, None] * h)                 # (n_sds, Nh)

        # dpsi/dc
        dc = sech2_z                                 # (n_sds, )

        # dpsi/dh
        dpsi_dh = sech2_z[:, None] * W2.flatten() # (n_sds, Nh)

        # dpsi/dW1
        # (n_sds, Nh) * (n_sds, Nh) and multiply by X
        tmp = dpsi_dh * sech2_h
        dW1 = tmp[:, :, None] * self.X[:, None, :]  # (n_sds, Nh, Nv)
        dW1 = dW1.reshape(self.n_sds, Nh * Nv)                     # (n_sds, Nh*Nv)

        # dpsi/db
        db = tmp                             # (n_sds, Nh)

        # final derivative matrix
        derivs = np.hstack([dW1, db, dW2, dc[:, None]])
        
        
        # cache results
        #for i, sd in enumerate(self.pspace):
        #    self._overlap_cache[sd] = {
        #        "overlap": overlaps[i].item(), 
        #        "derivative": derivs[i]
        #    }
        self._pspace_overlaps = overlaps.astype(float, copy=True)
        self._pspace_derivs = derivs.astype(float, copy=True)
        
        # normalization / output scale
        norm_sq = np.sum(self._pspace_overlaps**2)
        self.output_scale = 1.0 / (np.sqrt(norm_sq) if norm_sq > 1e-12 else 1.0)

        # if __debug__:
        #     print("[Timer] get_overlaps took {:.3f} seconds.".format(time.time() - time0))

        
    def get_overlap(self, sd, deriv=None, normalized=True):
        """Compute overlap of given Slater determinant with the wavefunction.
        
        sd in set is constant-time and dictionary lookup is constant-time,
        so the thousands of lookups in integrate_sd_wfn become extremely cheap.
        """
        
        # Fast membership test
        if sd not in self._pspace_set:
            if deriv is None:
                return 0.0
            else:
                return np.zeros(self.nparams)
        

        if self._pspace_overlaps is None or self._pspace_derivs is None:
            # Compute and populate both 
            self.get_overlaps()
                
        # If cache arrays filled, use them
        idx = self._pspace_index[sd]
        raw = self._pspace_overlaps[idx]
        
        if deriv is None:
            return float(raw * self.output_scale) if normalized else float(raw)

        raw_deriv = self._pspace_derivs[idx, deriv]
        return (raw_deriv * self.output_scale) if normalized else raw_deriv