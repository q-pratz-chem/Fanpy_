import numpy as np
import pytest
from fanpy.tools import slater
from fanpy.wfn.network.scaled_twolayertanh import ScaledTwoLayerTanhWfn


@pytest.fixture
def wfn():
    """Fixture: small wavefunction instance."""
    return ScaledTwoLayerTanhWfn(nelec=2, nspin=4, nhidden=3)


def test_param_shapes_and_count(wfn):
    """Check parameter structure and nparams consistency."""
    shapes = wfn.params_shape
    flat_total = sum(np.prod(s) for s in shapes)
    assert wfn.nparams == flat_total
    assert len(wfn._params) == len(shapes)
    for p, s in zip(wfn._params, shapes):
        assert p.shape == s
    assert wfn.params.shape == (flat_total,)


def test_no_nan_overlaps(wfn):
    """Ensure overlaps are finite and non-NaN."""
    wfn.get_overlaps()
    overlaps = np.array([v['overlap'] for v in wfn._overlap_cache.values()])
    assert np.all(np.isfinite(overlaps)), "Overlap contains NaN/Inf"
    assert np.any(overlaps != 0), "All overlaps are zero"


def test_normalization_is_one(wfn):
    """Verify normalization is enforced."""
    wfn.get_overlaps()
    overlaps = np.array([v["overlap"] for v in wfn._overlap_cache.values()])
    scaled_overlaps = overlaps * wfn.output_scale
    norm = np.sqrt(np.sum(np.abs(scaled_overlaps) ** 2))
    assert np.isclose(norm, 1.0, atol=1e-8), f"Wavefunction not normalized (‖Ψ‖={norm})"


def test_assign_params_from_flat_array(wfn):
    """Ensure assign_params correctly reshapes a flat NumPy array into structured parameters."""
    # 1️⃣ Create a random flat parameter array with correct total length
    total_params = wfn.nparams
    flat_params = np.arange(total_params, dtype=float)  # deterministic increasing sequence

    # 2️⃣ Assign these params directly
    wfn.assign_params(flat_params)

    # 3️⃣ Verify that internal _params were reshaped correctly
    structured_shapes = wfn.params_shape
    for p, expected_shape in zip(wfn._params, structured_shapes):
        assert p.shape == expected_shape, f"Expected {expected_shape}, got {p.shape}"

    # 4️⃣ Verify that params (re-flattened) matches original order
    re_flat = wfn.params
    assert np.allclose(re_flat, flat_params), \
        f"Flattened params differ after reshape. diff={np.max(np.abs(re_flat - flat_params))}"


def test_overlap_caching_consistency(wfn):
    """Ensure cached overlaps give same result as direct call."""
    ground = slater.ground(wfn.nelec, wfn.nspin)
    psi_1 = wfn.get_overlap(ground)
    psi_2 = wfn.get_overlap(ground)
    assert np.isclose(psi_1, psi_2, atol=1e-12)


def test_derivative_shape_and_finiteness(wfn):
    """Verify derivatives have correct size and finite values."""
    overlaps = wfn.get_overlaps()
    ground = slater.ground(wfn.nelec, wfn.nspin)
    deriv = wfn.get_overlap(ground, deriv=list(range(wfn.nparams)))
    assert deriv.shape[0] == wfn.nparams
    assert np.all(np.isfinite(deriv)), "Derivative contains NaN/Inf"


def test_finite_difference_gradient(wfn):
    """Compare analytic vs finite-diff gradient for one parameter."""
    ground = slater.ground(wfn.nelec, wfn.nspin)
    eps = 1e-6
    idx = 0  # test first parameter (W1[0,0])

    analytic = wfn.get_overlap(ground, deriv=[idx], normalized=False)[0]

    # finite difference approximation
    # recompute cache for perturbed params
    orig_val = wfn._params[0].flat[0]

    wfn._params[0].flat[0] = orig_val + eps
    wfn.get_overlaps()
    psi_plus = wfn.get_overlap(ground, normalized=False)
    
    wfn._params[0].flat[0] = orig_val - eps
    wfn.get_overlaps()
    psi_minus = wfn.get_overlap(ground, normalized=False)
    
    wfn._params[0].flat[0] = orig_val  # restore
    numeric = (psi_plus - psi_minus) / (2 * eps)

    assert np.isfinite(analytic)
    assert np.isfinite(numeric)
    assert np.isclose(analytic, numeric, rtol=1e-3, atol=1e-4), \
        f"Gradient mismatch: analytic={analytic}, numeric={numeric}"

