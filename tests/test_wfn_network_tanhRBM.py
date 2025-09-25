import numpy as np
import pytest
from fanpy.tools import slater
from fanpy.wfn.network.tanh_rbm import tanhRBM


@pytest.fixture
def tanh_rbm():
    # simple system: 2 electrons in 4 spin orbitals, 2 hidden units
    nelec, nspin, nhid = 2, 4, 2
    wfn = tanhRBM(nelec, nspin, nhid)
    wfn.assign_template_params(hf_init=True, seed=123)
    return wfn


def test_assign_params_and_cache_invalidation(tanh_rbm):
    # Store old params and cache
    old_cache = tanh_rbm._overlap_cache.copy()
    old_params = [p.copy() for p in tanh_rbm._params]

    # Assign new params
    tanh_rbm.assign_params()
    # Cache should be invalidated
    assert tanh_rbm._overlap_cache == {}
    # Params should be updated (template params used)
    for new, old in zip(tanh_rbm._params, old_params):
        assert not np.allclose(new, old)


def test_assign_template_params(tanh_rbm):
    # Ensure template params are created
    tanh_rbm.assign_template_params()
    assert tanh_rbm.template_params is not None
    # Check shapes
    a, b, w = tanh_rbm.template_params
    assert a.shape[0] == tanh_rbm.nspin
    assert b.shape[0] == tanh_rbm.nhidden
    assert w.shape == (tanh_rbm.nspin, tanh_rbm.nhidden)


def test_get_overlaps_shapes(tanh_rbm):
    overlaps = tanh_rbm.get_overlaps(normalized=False)
    # Number of overlaps matches pspace
    assert len(overlaps) == len(tanh_rbm.pspace)
    # Each overlap is a scalar
    assert np.all([np.isscalar(o) or np.ndim(o) == 0 for o in overlaps])


def test_get_overlap_consistency(tanh_rbm):
    # Check that get_overlap for single SD matches get_overlaps result
    sd = tanh_rbm.pspace[0]
    overlap_single = tanh_rbm.get_overlap(sd, normalized=False)
    overlaps_all = tanh_rbm.get_overlaps(normalized=False)
    assert np.isclose(overlap_single, overlaps_all[0])



def test_normalization(tanh_rbm):
    # Call normalize() to rescale the wavefunction
    norm = tanh_rbm.normalize()
    overlaps = tanh_rbm.get_overlaps()
    squared_norm = np.sum(np.abs(overlaps) ** 2)
    assert np.isclose(squared_norm, 1.0, rtol=1e-10, atol=1e-12)


def test_safe_log_abs_tanh_and_derivative(tanh_rbm):
    # Test small and large gamma values
    gammas = [0.0, 1e-8, -1e-8, 1.0, -1.0, 10.0, -10.0]
    for g in gammas:
        log_val = tanh_rbm.safe_log_abs_tanh(g)
        deriv_val = tanh_rbm.safe_dlogtanh_over_dgamma(g)
        # log|tanh| should be finite
        assert np.isfinite(log_val)
        # derivative should be finite
        assert np.isfinite(deriv_val)


def test_overlaps_with_random_params(tanh_rbm):
    # Assign random parameters and check overlaps remain finite
    rng = np.random.default_rng(12345)
    a = rng.uniform(-0.1, 0.1, size=tanh_rbm.nspin)
    b = rng.uniform(-0.1, 0.1, size=tanh_rbm.nhidden)
    w = rng.uniform(-0.1, 0.1, size=(tanh_rbm.nspin, tanh_rbm.nhidden))
    tanh_rbm.assign_params([a, b, w])
    overlaps = tanh_rbm.get_overlaps(normalized=False)
    assert np.all(np.isfinite(overlaps))


def test_tanh_rbm_overlap_and_derivative(tanh_rbm):
    # Pick first Slater determinant
    sd = tanh_rbm.pspace[0]

    # --- Test overlap is finite ---
    overlap = tanh_rbm.get_overlap(sd)
    assert np.isfinite(overlap), "Overlap is not finite"

    # --- Test derivative consistency using finite differences ---
    eps = 1e-6
    analytic = tanh_rbm.get_overlap(sd, deriv=np.arange(tanh_rbm.nparams), normalized=False)
    numeric = []

    base_params = tanh_rbm.params.copy()
    for i in range(tanh_rbm.nparams):
        shift = np.zeros_like(base_params)
        shift[i] = eps

        tanh_rbm.assign_params(base_params + shift)
        plus = tanh_rbm.get_overlap(sd, normalized=False)

        tanh_rbm.assign_params(base_params - shift)
        minus = tanh_rbm.get_overlap(sd, normalized=False)

        numeric.append((plus - minus) / (2 * eps))

    tanh_rbm.assign_params(base_params)  # restore original
    numeric = np.array(numeric)

    assert np.allclose(analytic, numeric, rtol=1e-4, atol=1e-6), \
        "Analytic and numeric derivatives do not match"

    # --- Test normalization ---
    norm = tanh_rbm.normalize()
    assert np.isclose(norm, np.linalg.norm(tanh_rbm.get_overlaps(normalized=False))), \
        "Normalization factor mismatch"


