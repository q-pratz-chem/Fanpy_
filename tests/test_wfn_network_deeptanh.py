import numpy as np
import pytest
from fanpy.tools import slater
from fanpy.wfn.network.deeptanh import DeepTanhWfn


@pytest.fixture(params=[2, 3])
def wfn(request):
    """Small DeepTanh wavefunction with varying depth."""
    return DeepTanhWfn(
        nelec=2,
        nspin=4,
        nhidden=3,
        num_layers=request.param,
        pspace_exc_orders=[2],
    )


def test_param_shapes_and_count(wfn):
    shapes = wfn.params_shape
    flat_total = sum(np.prod(s) for s in shapes)

    assert wfn.nparams == flat_total
    assert len(wfn._params) == len(shapes)

    for p, s in zip(wfn._params, shapes):
        assert p.shape == s

    assert wfn.params.shape == (flat_total,)


def test_safe_tanh():
    x = np.array([-1e12, -3.0, 0.0, 3.0, 1e12])
    y = DeepTanhWfn.safe_tanh(x)

    assert np.all(np.isfinite(y))
    np.testing.assert_allclose(
        y,
        np.tanh(np.clip(x, -30, 30)),
        rtol=1e-7,
        atol=1e-7,
    )


def test_no_nan_overlaps(wfn):
    wfn.get_overlaps()
    overlaps = wfn._pspace_overlaps

    assert np.all(np.isfinite(overlaps))
    assert np.any(overlaps != 0.0)


def test_normalization_is_one(wfn):
    wfn.get_overlaps()
    scaled = wfn._pspace_overlaps * wfn.output_scale
    norm = np.linalg.norm(scaled)

    assert np.isclose(norm, 1.0, atol=1e-8)


def test_sd_not_in_pspace_overlaps(wfn):
    wfn.get_overlaps()

    sd = 0b1100  # not necessarily in pspace
    olp = wfn.get_overlap(sd)
    assert olp == 0.0

    deriv = wfn.get_overlap(sd, deriv=[0, 1, 2])
    assert deriv.shape == (3,)
    assert np.all(deriv == 0.0)


def test_assign_params_from_flat_array(wfn):
    flat = np.arange(wfn.nparams, dtype=float)
    wfn.assign_params(flat)

    for p, shape in zip(wfn._params, wfn.params_shape):
        assert p.shape == shape

    np.testing.assert_allclose(wfn.params, flat)


def test_overlap_caching_consistency(wfn):
    ground = slater.ground(wfn.nelec, wfn.nspin)

    psi1 = wfn.get_overlap(ground)
    psi2 = wfn.get_overlap(ground)

    assert np.isclose(psi1, psi2, atol=1e-12)


def test_derivative_shape_and_finiteness(wfn):
    wfn.get_overlaps()
    ground = slater.ground(wfn.nelec, wfn.nspin)

    deriv = wfn.get_overlap(ground, deriv=list(range(wfn.nparams)))

    assert deriv.shape == (wfn.nparams,)
    assert np.all(np.isfinite(deriv))


def test_get_overlaps_no_deriv_sets_scale_and_returns(wfn):
    """
    Ensure get_overlaps(deriv=False):
    - populates _pspace_overlaps
    - computes output_scale correctly
    - does NOT populate derivatives
    """

    # Call the branch under test
    ret = wfn.get_overlaps(deriv=False)

    # Should explicitly return None
    assert ret is None

    # Overlaps must be populated
    assert wfn._pspace_overlaps is not None
    overlaps = wfn._pspace_overlaps

    # Must be finite floats
    assert overlaps.dtype == float
    assert np.all(np.isfinite(overlaps))

    # output_scale must match definition
    norm = np.linalg.norm(overlaps)
    expected_scale = 1.0 / (norm if norm > 1e-12 else 1.0)
    assert np.isclose(wfn.output_scale, expected_scale)

    # Derivatives should NOT be computed
    assert wfn._pspace_derivs is None



def test_finite_difference_gradient(wfn):
    ground = slater.ground(wfn.nelec, wfn.nspin)

    eps = 1e-6
    idx = 0  # first parameter

    # analytic
    wfn.get_overlaps()
    analytic = wfn.get_overlap(ground, deriv=[idx], normalized=False)[0]

    # numeric
    flat = wfn.params.copy()

    flat[idx] += eps
    wfn.assign_params(flat)
    wfn.get_overlaps()
    psi_plus = wfn.get_overlap(ground, normalized=False)

    flat[idx] -= 2 * eps
    wfn.assign_params(flat)
    wfn.get_overlaps()
    psi_minus = wfn.get_overlap(ground, normalized=False)

    numeric = (psi_plus - psi_minus) / (2 * eps)

    assert np.isfinite(analytic)
    assert np.isfinite(numeric)
    assert np.isclose(analytic, numeric, rtol=1e-3, atol=1e-4), (
        f"Gradient mismatch: analytic={analytic}, numeric={numeric}"
    )

