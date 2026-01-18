"""Scaled n-layer tanh neural network wavefunction with Xavier initialization.
   Author: Pratiksha G. (2025)
"""

import numpy as np
from fanpy.tools import slater, sd_list
from fanpy.wfn.base import BaseWavefunction
import torch


class DeepTanhWfn(BaseWavefunction):
    r"""
    Bias-free deep tanh neural network wavefunction.
    
    The wavefunction amplitude for a Slater determinant is represented by a
    fully-connected neural network with hyperbolic tangent activations,

  
        h0 = x
        h_l = tanh(W_l h_{l-1} ),  l = 1, ..., L
        z = W_{L+1} h_L 
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
        num_layers=2,
        scale=1.0, 
        pspace_exc_orders=None, 
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
        num_layers : int
            Total number of layers excluding the input layer and including
            the output layer. Equal to (number of hidden layers + output layer).
        init_scale : float
            Scaling factor used in Xavier initialization of the weights.
        pspace_exc_orders : tuple of int or None
            Allowed excitation orders defining the P-space of Slater determinants.
        """
        super().__init__(nelec, nspin, memory=memory)

        self.nhidden = nhidden
        self.num_layers = num_layers
        self.init_scale = scale
        self.output_scale = 1.0
        
        self.pspace_exc_orders=pspace_exc_orders
        self._template_params = None

        self._pspace_list = None
        self._pspace_set = None
        self._pspace_index = None
        self._pspace_overlaps = None
        self._pspace_derivs = None
        
        self.tanh1 = np.tanh(1.0) 
        
        # build determinant matrix ONCE
        self._build_pspace_matrix()
        
        # self._overlap_cache = {}
        self.assign_params(params=params)
              
        
    

    # ---------- parameter bookkeeping ----------
    @property
    def params_shape(self):
        shapes = []
        
        # hidden layers (num_layers - 1)
        if self.num_layers > 1:
            shapes.append((self.nhidden, self.nspin))  # W1
            shapes.append((self.nhidden,))             # b1
            
            for _ in range(1, self.num_layers - 1):
                shapes.append((self.nhidden, self.nhidden))  # W_l
                shapes.append((self.nhidden,))             # b_l
                
        # output layer
        shapes.append((1, self.nhidden if self.num_layers > 1 else self.nspin)) # W_out
        shapes.append((1,))    #c

        return shapes


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
        params = []
        fan_in = self.nspin
        
        print(f"\nBuilding wavefunction space with excitation orders = {self.pspace_exc_orders}")
        print(f"Number of hidden units = {int(self.nhidden/self.nspin)}.nspin = {self.nhidden}")
        print(f"Number of layers = {self.num_layers} = #hidden layers + 1 output layer\n")
        print(f"len(self.pspace) = {len(self.pspace)}\n") 
        
        # hidden layers # Xavier uniform for W_l
        print(f"\nWeights initalized using Xavier uniform initialization. scale = {self.init_scale}")
        for l in range(self.num_layers -1): 
            fan_out = self.nhidden
            limit = np.sqrt(self.init_scale / (fan_in + fan_out))
            
            W = rng.uniform(-limit, limit, size=(fan_out, fan_in))
            b = np.zeros(fan_out)
            
            params.extend([W, b])
            fan_in = fan_out
        
        # output layer
        limit = np.sqrt(self.init_scale / (fan_in + 1))
        Wout = rng.uniform(-limit, limit, size=(1, fan_in))
        c = np.zeros(1)
        params.extend([Wout, c])
        
        self._template_params = params

        
    def assign_params(self, params=None):
        # print("Assigning params...")
        if params is None:
            if self._template_params is None:
                self.assign_template_params()
            params = self._template_params
    
        
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
        Compute overlaps and derivatives for all Slater determinants in P-space.

        Derivatives are returned in the same order as self._params, i.e.
        [W1, W2, ..., W_out], flattened.
        """
        # import time; time0 = time.time()
        

        params = self._params
        n_sds = self.n_sds
        
        # ---- Forward (vectorized) ----
        h = self.X  # h0 : (n_sds, Nv)
        idx = 0
        activations = []   # h_l
        preacts = []       # z_l
        
        for l in range(0, self.num_layers-1):
            W = params[idx]
            b = params[idx + 1]
            idx += 2
            
            z = h @ W.T + b
            h = np.tanh(z) 
            
            preacts.append(z)
            activations.append(h)

        # output
        W_out = params[idx]
        c = params[idx + 1]
        
        z_out = (h @ W_out.T).ravel() + c[0] 
        tanh_z = np.tanh(z_out)
        overlaps = tanh_z / self.tanh1


        if not deriv:
            self._pspace_overlaps = overlaps.astype(float, copy=True)
            # normalization / output scale
            norm = np.linalg.norm(self._pspace_overlaps)
            self.output_scale = 1.0 / (norm if norm > 1e-12 else 1.0)
            return

        # ---- Backward pass ----
        blocks = []
        dWs = [None] * len(params)

        # output layer
        sech2_out = (1 - tanh_z ** 2) / self.tanh1     # (n_sds, )
        dW_out = sech2_out[:, None] * h                # (n_sds, nhidden)
        dc = sech2_out[:, None]                        # (n_sds, 1)
        
        # dWs[-1] = dW_out

        delta = sech2_out[:, None] * W_out             # (n_sds, nhidden)
        

        # hidden layers (reverse order)  
        for l in reversed(range(self.num_layers - 1)):
            h_prev = self.X if l == 0 else activations[l-1]
            h_l = activations[l]
            
            # apply activation derivative FIRST
            delta = delta * (1.0 - h_l**2)         # ∂ψ/∂z_l
    
            # gradient wrt W_l
            dW = delta[:, :, None] * h_prev[:, None, :]  # (n_sds, out, in)
            db = delta
            
            blocks.insert(0, db.reshape(n_sds, -1))
            blocks.insert(0, dW.reshape(n_sds, -1))
             
            # propagate backward FIRST
            W = params[2 * l]
            delta_ = (delta @ W)
            
            # # then apply acitvation derivative of previous layer
            # sech2_prev = 1.0 - h_prev**2
            # delta = delta_ * sech2_prev
            
        # append output layer derivatives
        blocks.append(dW_out.reshape(n_sds, -1))
        blocks.append(dc.reshape(n_sds, -1))
        
        # flatten output grads
        derivs = np.hstack(blocks)

        # --------------------------------------------------
        # Sanity check
        # --------------------------------------------------
        assert derivs.shape == (n_sds, self.nparams), (
            f"Derivative mismatch: {derivs.shape[1]} vs {self.nparams}"
        )

        # cache results
        self._pspace_overlaps = overlaps.astype(float, copy=True)
        self._pspace_derivs = derivs.astype(float, copy=True)
        
        # normalization / output scale
        norm = np.linalg.norm(self._pspace_overlaps)
        self.output_scale = 1.0 / (norm if norm > 1e-12 else 1.0)

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
                return np.zeros(len(deriv))
        

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
