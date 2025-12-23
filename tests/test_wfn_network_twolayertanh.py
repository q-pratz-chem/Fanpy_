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


def test_hf_init_basic_behavior():
    """HF init: finite params, HF hidden activations ~0, HF overlap nonzero."""
    nspin = 4
    nmo = nspin // 2

    hf_mo_coeff = np.eye(nmo)

    wfn = ScaledTwoLayerTanhWfn(
        nelec=2,
        nspin=nspin,
        nhidden=3,
        hf_init=True,
        hf_mo_coeff=hf_mo_coeff
    )

    # params finite
    for p in wfn._params:
        assert np.all(np.isfinite(p))

    # HF hidden pre-activation ~ 0
    W1, b, _, _ = wfn._params
    x_ref = wfn.X[0]
    assert np.allclose(W1 @ x_ref + b, 0.0, atol=1e-8)

    # HF overlap exists
    wfn.get_overlaps()
    hf_sd = wfn.pspace[0]
    olp = wfn.get_overlap(hf_sd)
    assert np.isfinite(olp) and abs(olp) > 0.0


def test_safe_tanh():
    """Test numerical stability, clipping, and NaN-safety of safe_tanh()."""
    from fanpy.wfn.network.scaled_twolayertanh import ScaledTwoLayerTanhWfn

    # 1. Normal values should match np.tanh
    x_normal = np.linspace(-5, 5, 20)
    out_normal = ScaledTwoLayerTanhWfn.safe_tanh(x_normal)
    expected_normal = np.tanh(x_normal)
    np.testing.assert_allclose(out_normal, expected_normal, rtol=1e-7, atol=1e-7)

    # 2. Very large positive values should be clipped to +30
    x_large_pos = np.array([1000.0])
    out_large_pos = ScaledTwoLayerTanhWfn.safe_tanh(x_large_pos)
    assert np.isclose(out_large_pos, np.tanh(30.0))

    # 3. Very large negative values should be clipped to -30
    x_large_neg = np.array([-1000.0])
    out_large_neg = ScaledTwoLayerTanhWfn.safe_tanh(x_large_neg)
    assert np.isclose(out_large_neg, np.tanh(-30.0))

    # 4. Should never produce NaN or Inf
    x_extreme = np.array([-1e12, 0.0, 1e12])
    out_extreme = ScaledTwoLayerTanhWfn.safe_tanh(x_extreme)
    assert np.all(np.isfinite(out_extreme)), "safe_tanh produced NaN or Inf"

    # 5. Scalars and vectors both supported
    scalar = 3.0
    vec = np.array([-3.0, 0.0, 3.0])
    assert np.isclose(
        ScaledTwoLayerTanhWfn.safe_tanh(scalar),
        np.tanh(np.clip(scalar, -30, 30))
    )
    np.testing.assert_allclose(
        ScaledTwoLayerTanhWfn.safe_tanh(vec),
        np.tanh(np.clip(vec, -30, 30))
    )


def test_no_nan_overlaps(wfn):
    """Ensure overlaps are finite and non-NaN."""
    wfn.get_overlaps()
    #overlaps = np.array([v['overlap'] for v in wfn._overlap_cache.values()])
    overlaps = wfn._pspace_overlaps
    assert np.all(np.isfinite(overlaps)), "Overlap contains NaN/Inf"
    assert np.any(overlaps != 0), "All overlaps are zero"


def test_normalization_is_one(wfn):
    """Verify normalization is enforced."""
    wfn.get_overlaps()
    #overlaps = np.array([v["overlap"] for v in wfn._overlap_cache.values()])
    overlaps = wfn._pspace_overlaps
    scaled_overlaps = overlaps * wfn.output_scale
    norm = np.sqrt(np.sum(np.abs(scaled_overlaps) ** 2))
    assert np.isclose(norm, 1.0, atol=1e-8), f"Wavefunction not normalized (‖Ψ‖={norm})"


def test_sd_not_in_pspace_overlaps(wfn):
    """Ensure overlap and deriv are zero for sd not in pspace"""
    wfn.get_overlaps()
    #overlaps = np.array([v['overlap'] for v in wfn._overlap_cache.values()])
    overlaps = wfn._pspace_overlaps
    sd = 0b1100
    olp = wfn.get_overlap(sd)
    assert olp == 0.0

    deriv_indices = [0,1]
    olp_deriv = wfn.get_overlap(sd, deriv=deriv_indices)
    assert isinstance(olp_deriv, np.ndarray)
    assert olp_deriv.shape == (len(deriv_indices),)
    assert np.all(olp_deriv == 0.0)
  
    deriv_indices = 0
    olp_deriv = wfn.get_overlap(sd, deriv=deriv_indices)
    assert olp_deriv == 0.0


def test_assign_params_from_flat_array(wfn):
    """Ensure assign_params correctly reshapes a flat NumPy array into structured parameters."""
    # Create a random flat parameter array with correct total length
    total_params = wfn.nparams
    flat_params = np.arange(total_params, dtype=float)  # deterministic increasing sequence

    # Assign these params directly
    wfn.assign_params(flat_params)

    # Verify that internal _params were reshaped correctly
    structured_shapes = wfn.params_shape
    for p, expected_shape in zip(wfn._params, structured_shapes):
        assert p.shape == expected_shape, f"Expected {expected_shape}, got {p.shape}"

    # Verify that params (re-flattened) matches original order
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
    wfn.get_overlaps()
    overlaps = wfn._pspace_overlaps
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

