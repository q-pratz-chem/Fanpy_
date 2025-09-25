# test_rbm_softplus_jastrow.py
import numpy as np
import pytest
from fanpy.tools import slater
from fanpy.wfn.network.softplus_jastrow_rbm import RBM_SoftPlus_Jastrow  # adjust import to your path


@pytest.fixture
def rbm():
    # simple system: 2 electrons in 4 spin orbitals, 2 hidden units
    nelec, nspin, nhid = 2, 4, 2
    wfn = RBM_SoftPlus_Jastrow(nelec, nspin, nhid)
    wfn.assign_template_params(hf_init=True, seed=123)
    return wfn


def test_param_shapes(rbm):
    # Check consistency of nparams vs flattened params
    flat = rbm.params
    assert flat.shape[0] == rbm.nparams
    # All param blocks have finite values
    for p in rbm._params:
        assert np.all(np.isfinite(p))


def test_overlap_values(rbm):
    overlaps = rbm.get_overlaps(normalized=False)
    assert overlaps.shape[0] == len(rbm.pspace)
    assert np.all(np.isfinite(overlaps))
    # HF determinant should exist in pspace
    hf_sd = slater.ground(rbm.nelec, rbm.nspin)
    psi_hf = rbm.get_overlap(hf_sd)
    assert np.isfinite(psi_hf)


def test_normalization(rbm):
    # Call normalize() to rescale the wavefunction
    norm = rbm.normalize()
    overlaps = rbm.get_overlaps()
    squared_norm = np.sum(np.abs(overlaps) ** 2)
    assert np.isclose(squared_norm, 1.0, rtol=1e-10, atol=1e-12)


def test_derivative_consistency(rbm):
    # pick one determinant
    sd = rbm.pspace[0]
    eps = 1e-6
    analytic = rbm.get_overlap(sd, deriv=np.arange(rbm.nparams), normalized=False)
    numeric = []
    base = rbm.get_overlap(sd, normalized=False)
    for i in range(rbm.nparams):
        shift = np.zeros(rbm.nparams)
        shift[i] = eps
        rbm.assign_params(rbm.params + shift)
        plus = rbm.get_overlap(sd, normalized=False)
        rbm.assign_params(rbm.params - 2*shift)
        minus = rbm.get_overlap(sd, normalized=False)
        rbm.assign_params(rbm.params + shift)  # restore
        num = (plus - minus) / (2 * eps)
        numeric.append(num)
    numeric = np.array(numeric)
    # Compare within tolerance
    assert np.allclose(analytic, numeric, rtol=1e-4, atol=1e-4)

