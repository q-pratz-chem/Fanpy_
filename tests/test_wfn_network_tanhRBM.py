import numpy as np
import pytest
from unittest.mock import patch, PropertyMock
from fanpy.wfn.network.tanh_rbm import tanhRBM


@pytest.fixture
def tanh_rbm():
    """A simple RBM: 2e in 4 spin orbitals, 2 hidden units."""
    wfn = tanhRBM(nelec=2, nspin=4, nhidden=2)
    wfn.assign_template_params(hf_init=True, seed=123)
    return wfn


# ─────────────────────────────── PARAMS / CACHE ─────────────────────────────── #

def test_assign_params_inits_and_populates_cache(tanh_rbm):
    tanh_rbm.assign_params()
    assert len(tanh_rbm._overlap_cache) > 0
    assert tanh_rbm.output_scale > 0


def test_assign_template_params_shapes(tanh_rbm):
    a, b, w = tanh_rbm.template_params
    assert a.shape == (tanh_rbm.nspin,)
    assert b.shape == (tanh_rbm.nhidden,)
    assert w.shape == (tanh_rbm.nspin, tanh_rbm.nhidden)


# ─────────────────────────────── NORMALIZATION ─────────────────────────────── #

def test_normalize_computes_output_scale_correctly(tanh_rbm):
    tanh_rbm.get_overlaps(normalized=False)
    tanh_rbm.output_scale = 1.0  # force recomputation
    tanh_rbm.normalize()
    overlaps = np.array([v["overlap"] for v in tanh_rbm._overlap_cache.values()])
    expected = 1.0 / np.sqrt(np.sum(np.abs(overlaps) ** 2))
    assert np.isclose(tanh_rbm.output_scale, expected, rtol=1e-15)


def test_normalize_raises_when_no_cached_overlaps(tanh_rbm):
    tanh_rbm._overlap_cache = {}
    tanh_rbm.output_scale = 1.0
    with pytest.raises(ValueError, match="No cached overlaps"):
        tanh_rbm.normalize()


def test_normalize_raises_when_zero_norm(tanh_rbm):
    tanh_rbm._overlap_cache = {sd: {"overlap": 0.0} for sd in tanh_rbm.pspace}
    tanh_rbm.output_scale = 1.0
    with pytest.raises(ValueError, match="zero norm"):
        tanh_rbm.normalize()


def test_normalize_already_normalized_with_missing_pspace_entries(tanh_rbm, capsys):
    tanh_rbm.get_overlaps(normalized=True)
    tanh_rbm.output_scale = 0.5  # mark as already normalized

    # Create a pspace containing some extra SDs not in cache
    fake_pspace = list(tanh_rbm.pspace) + [max(tanh_rbm.pspace) + 500]

    # Should trigger the 'missing' warning and early return
    tanh_rbm.normalize(pspace=fake_pspace)
    out = capsys.readouterr().out
    assert "missing" in out or "Warning" in out


def test_normalize_already_normalized_but_all_zero_overlaps(tanh_rbm):
    # Pretend the wavefunction is already normalized
    tanh_rbm.output_scale = 0.5

    # Make all cached overlaps zero
    tanh_rbm._overlap_cache = {
        sd: {"overlap": 0.0, "derivative": np.zeros(tanh_rbm.nparams)}
        for sd in tanh_rbm.pspace
    }

    # This should raise the specific ValueError
    with pytest.raises(ValueError, match="zero norm; cannot normalize"):
        tanh_rbm.normalize(pspace=tanh_rbm.pspace)


# ─────────────────────────────── OVERLAPS ─────────────────────────────── #

def test_get_overlaps_shapes(tanh_rbm):
    tanh_rbm.get_overlaps(normalized=False)
    overlaps = [v["overlap"] for v in tanh_rbm._overlap_cache.values()]
    assert len(overlaps) == len(tanh_rbm.pspace)
    assert all(np.ndim(o) == 0 for o in overlaps)


def test_overlap_with_empty_pspace(tanh_rbm):
    with patch.object(type(tanh_rbm), "pspace", new_callable=PropertyMock, return_value=[]):
        tanh_rbm._overlap_cache = {}
        overlaps = tanh_rbm.get_overlaps(normalized=False)
        assert overlaps.size == 0


def test_get_overlap_inserts_missing_sd_and_returns_zero(tanh_rbm):
    tanh_rbm.get_overlaps(normalized=False)
    fake_sd = max(tanh_rbm.pspace) + 999
    assert fake_sd not in tanh_rbm._overlap_cache
    val = tanh_rbm.get_overlap(fake_sd)
    assert val == 0.0
    assert fake_sd in tanh_rbm._overlap_cache


def test_get_overlaps_raises_zero_norm_after_stabilization(tanh_rbm):
    # Manually zero out parameters so everything becomes flat
    tanh_rbm._params = [
        np.zeros(tanh_rbm.nspin),
        np.zeros(tanh_rbm.nhidden),
        np.zeros((tanh_rbm.nspin, tanh_rbm.nhidden)),
    ]
    # pspace will still be valid
    with pytest.raises(ValueError, match="zero norm after stablization"):
        tanh_rbm.get_overlaps(normalized=False)


def test_get_overlap_returns_derivative_scaled_by_output_scale(tanh_rbm):
    tanh_rbm.get_overlaps(normalized=False)
    sd = tanh_rbm.pspace[0]
    raw = tanh_rbm._overlap_cache[sd]["derivative"]
    inds = np.arange(3)
    out = tanh_rbm.get_overlap(sd, deriv=inds, normalized=True)
    assert np.allclose(out, raw[inds] * tanh_rbm.output_scale)


def test_get_overlap_returns_scaled_raw_overlap(tanh_rbm):
    tanh_rbm.get_overlaps(normalized=False)
    sd = tanh_rbm.pspace[0]
    raw = tanh_rbm._overlap_cache[sd]["overlap"]
    out = tanh_rbm.get_overlap(sd, deriv=None, normalized=True)
    assert np.isclose(out, raw * tanh_rbm.output_scale, rtol=1e-12)


# ─────────────────────────────── NUMERICAL / ANALYTICAL DERIV ─────────────────────────────── #

def test_tanh_rbm_overlap_and_derivative_consistency(tanh_rbm):
    sd = tanh_rbm.pspace[0]
    eps = 1e-6
    analytic = tanh_rbm.get_overlap(sd, deriv=np.arange(tanh_rbm.nparams), normalized=False)
    numeric = []
    base = tanh_rbm.params.copy()
    for i in range(tanh_rbm.nparams):
        shift = np.zeros_like(base)
        shift[i] = eps
        tanh_rbm.assign_params(base + shift)
        plus = tanh_rbm.get_overlap(sd, normalized=False)
        tanh_rbm.assign_params(base - shift)
        minus = tanh_rbm.get_overlap(sd, normalized=False)
        numeric.append((plus - minus) / (2 * eps))
    tanh_rbm.assign_params(base)
    assert np.allclose(analytic, numeric, rtol=1e-4, atol=1e-6)


# ─────────────────────────────── MISC / EDGE UTILS ─────────────────────────────── #

@pytest.mark.parametrize("g", [0.0, 1e-8, -1e-8, 1.0, -1.0, 10.0, -10.0])
def test_safe_log_and_derivative_are_finite(tanh_rbm, g):
    assert np.isfinite(tanh_rbm.safe_log_abs_tanh(g))
    assert np.isfinite(tanh_rbm.safe_dlogtanh_over_dgamma(g))


def test_overlaps_with_random_params(tanh_rbm):
    rng = np.random.default_rng(123)
    a = rng.uniform(-0.1, 0.1, tanh_rbm.nspin)
    b = rng.uniform(-0.1, 0.1, tanh_rbm.nhidden)
    w = rng.uniform(-0.1, 0.1, (tanh_rbm.nspin, tanh_rbm.nhidden))
    tanh_rbm.assign_params([a, b, w])
    tanh_rbm.get_overlaps(normalized=False)
    overlaps = [v["overlap"] for v in tanh_rbm._overlap_cache.values()]
    assert np.all(np.isfinite(overlaps))


def test_spin_property(tanh_rbm):
    assert tanh_rbm.spin == 0



# import numpy as np
# import pytest
# from fanpy.tools import slater
# from fanpy.wfn.network.tanh_rbm import tanhRBM


# @pytest.fixture
# def tanh_rbm():
#     # simple system: 2 electrons in 4 spin orbitals, 2 hidden units
#     nelec, nspin, nhid = 2, 4, 2
#     wfn = tanhRBM(nelec, nspin, nhid)
#     wfn.assign_template_params(hf_init=True, seed=123)
#     return wfn


# def test_assign_params_and_cache_invalidation(tanh_rbm):
#     # Assign new params
#     tanh_rbm.assign_params()

#     # Cache should be invalidated
#     # assert tanh_rbm._overlap_cache == {}
#     assert len(tanh_rbm._overlap_cache) > 0, "Cache not populated automatically after param assignment"



# def test_assign_template_params(tanh_rbm):
#     # Ensure template params are created
#     tanh_rbm.assign_template_params()
#     assert tanh_rbm.template_params is not None

#     # Check shapes
#     a, b, w = tanh_rbm.template_params
#     assert a.shape[0] == tanh_rbm.nspin
#     assert b.shape[0] == tanh_rbm.nhidden
#     assert w.shape == (tanh_rbm.nspin, tanh_rbm.nhidden)


# def test_normalize_warns_with_missing_sds(tanh_rbm, capsys):
#     # Populate cache with only half of the determinants
#     half_sds = tanh_rbm.pspace[: len(tanh_rbm.pspace)//2]
#     tanh_rbm.get_overlaps(normalized=False)
#     # Keep only some cached entries to simulate missing ones
#     tanh_rbm._overlap_cache = {sd: tanh_rbm._overlap_cache[sd] for sd in half_sds}

#     # Call normalize with full pspace → triggers 'missing' branch
#     tanh_rbm.normalize(pspace=tanh_rbm.pspace)
#     captured = capsys.readouterr()

#     # Verify that warning about missing SDs appears
#     assert "missing" in captured.out
#     assert "normalize() called with pspace" in captured.out


# def test_normalize_raises_when_no_cached_overlaps(tanh_rbm):
#     # Clear the overlap cache completely
#     tanh_rbm._overlap_cache = {}
#     # Ensure output_scale is reset so it goes into the fallback branch
#     tanh_rbm.output_scale = 1.0

#     # Expect ValueError since overlaps.size == 0
#     with pytest.raises(ValueError) as excinfo:
#         tanh_rbm.normalize()
#     assert "No cached overlaps" in str(excinfo.value)


# def test_normalize_raises_when_zero_norm(tanh_rbm):
#     # Fill cache with zeros
#     tanh_rbm._overlap_cache = {sd: {"overlap": 0.0} for sd in tanh_rbm.pspace}
#     tanh_rbm.output_scale = 1.0  # ensure fallback branch executes

#     with pytest.raises(ValueError) as excinfo:
#         tanh_rbm.normalize()
#     assert "zero norm" in str(excinfo.value)


# def test_get_overlap_inserts_missing_sd_and_returns_zero(tanh_rbm):
#     # Ensure overlap cache is populated for other SDs
#     tanh_rbm.get_overlaps(normalized=False)
    
#     # Pick a Slater determinant not in pspace (force "missing" case)
#     fake_sd = max(tanh_rbm.pspace) + 1000
    
#     # Confirm it’s not in cache
#     assert fake_sd not in tanh_rbm._overlap_cache

#     # Call get_overlap → should add dummy entry and return 0.0
#     result = tanh_rbm.get_overlap(fake_sd, normalized=True)
#     assert np.isclose(result, 0.0), "Should return 0.0 for missing SD"

#     # Cache should now contain the fake_sd key
#     assert fake_sd in tanh_rbm._overlap_cache
#     cached_entry = tanh_rbm._overlap_cache[fake_sd]
#     assert "overlap" in cached_entry and "derivative" in cached_entry
#     assert np.allclose(cached_entry["derivative"], np.zeros(tanh_rbm.nparams))


# def test_normalize_computes_output_scale_correctly(tanh_rbm):
#     # Populate a valid cache with finite overlaps
#     tanh_rbm.get_overlaps(normalized=False)
#     # Reset output_scale to force recomputation
#     tanh_rbm.output_scale = 1.0

#     # Call normalize()
#     tanh_rbm.normalize()
#     overlaps = np.array([v["overlap"] for v in tanh_rbm._overlap_cache.values()])
#     expected_scale = 1.0 / np.sqrt(np.sum(np.abs(overlaps) ** 2))

#     assert np.isclose(tanh_rbm.output_scale, expected_scale, rtol=1e-15)
#     assert tanh_rbm.output_scale > 0.0


# def test_get_overlap_returns_derivative_scaled_by_output_scale(tanh_rbm):
#     # Ensure cache populated
#     tanh_rbm.get_overlaps(normalized=False)

#     # Pick an SD that exists in cache
#     sd = tanh_rbm.pspace[0]

#     # Manually retrieve raw derivative and output_scale
#     raw_deriv = tanh_rbm._overlap_cache[sd]["derivative"]
#     scale = tanh_rbm.output_scale

#     # Call get_overlap with deriv indices
#     deriv_inds = np.arange(min(3, tanh_rbm.nparams))  # just a few
#     out = tanh_rbm.get_overlap(sd, deriv=deriv_inds, normalized=True)

#     # Check output matches scaled derivative
#     expected = raw_deriv[deriv_inds] * scale
#     assert np.allclose(out, expected, rtol=1e-12, atol=1e-14)


# def test_overlap_with_empty_pspace(tanh_rbm):
#     # Create a subclass to override pspace property dynamically
#     class DummyRBM(tanh_rbm.__class__):
#         @property
#         def pspace(self):
#             return []

#     dummy = DummyRBM(tanh_rbm.nelec, tanh_rbm.nspin, tanh_rbm.nhidden)
#     dummy.assign_template_params()
#     dummy._overlap_cache = {}

#     overlaps = dummy.get_overlaps(normalized=False)
#     assert overlaps.size == 0, "Overlaps should be empty for empty pspace"



# # def test_normalize_computes_output_scale_correctly(tanh_rbm):
# #     # Populate a valid cache with finite overlaps
# #     tanh_rbm.get_overlaps(normalized=False)
# #     # Reset output_scale to force recomputation
# #     tanh_rbm.output_scale = 1.0

# #     # Call normalize()
# #     tanh_rbm.normalize()
# #     overlaps = np.array([v["overlap"] for v in tanh_rbm._overlap_cache.values()])
# #     expected_scale = 1.0 / np.sqrt(np.sum(np.abs(overlaps) ** 2))

# #     assert np.isclose(tanh_rbm.output_scale, expected_scale, rtol=1e-15)
# #     assert tanh_rbm.output_scale > 0.0


# def test_normalize_with_correct_pspace(tanh_rbm):
#     tanh_rbm.get_overlaps(normalized=False)

#     # Normalize with correct pspace
#     tanh_rbm.normalize(pspace=tanh_rbm.pspace)
#     norm = np.sqrt(np.sum([np.abs(entry["overlap"]) ** 2 for entry in tanh_rbm._overlap_cache.values()]))

#     # Check norm is positive
#     assert norm > 0.0
#     assert np.isclose(1.0 / norm, tanh_rbm.output_scale, rtol=1e-20), "Normalization scale is not consistent"
   
#     # Check that output_scale is updated
#     assert tanh_rbm.output_scale != 1.0


# def test_normalize_raises_for_zero_norm(tanh_rbm):
#     # Manually create a fake overlap cache where all overlaps are zero
#     tanh_rbm._overlap_cache = {sd: {"overlap": 0.0} for sd in tanh_rbm.pspace}

#     # Now call normalize and expect a ValueError
#     with pytest.raises(ValueError) as excinfo:
#         tanh_rbm.normalize()

#     # Verify correct error message
#     assert "zero norm" in str(excinfo.value)


# def test_assign_params_warns_on_exploding_params(tanh_rbm, capsys):
#     # Create parameter arrays with very large values to trigger the warning
#     a = np.random.randint(5, 20, size=tanh_rbm.nspin)
#     b = np.random.randint(10, 30, size=tanh_rbm.nhidden)
#     w = np.full((tanh_rbm.nspin, tanh_rbm.nhidden), 20.0)

#     # Assign once to set _prev_params (needed for comparison)
#     tanh_rbm.assign_params([a, b, w])

#     # Call get_overlaps to invoke check_params()
#     # tanh_rbm.get_overlaps(normalized=False)

#     # Capture printed output and assert that the warning message appears
#     captured = capsys.readouterr()

#     # assert "Parameters exploding beyond 10" in captured.out
#     # assert "exceeds threshold" in captured.out

# def test_norm_zero_gets_overlaps(tanh_rbm):
#     a = np.full(tanh_rbm.nspin, 0.0)
#     b = np.full(tanh_rbm.nhidden, 0.0)
#     w = np.full((tanh_rbm.nspin, tanh_rbm.nhidden), 0.0)

    
#     with pytest.raises(ValueError) as excinfo: 
#         tanh_rbm.assign_params([a, b, w])

#     assert "Wavefunction has zero norm" in str(excinfo.value)

# def test_get_overlaps_shapes(tanh_rbm):
#     tanh_rbm.get_overlaps(normalized=False)

#     # Extract overlaps from the cache
#     overlaps = [entry["overlap"] for entry in tanh_rbm._overlap_cache.values()]

#     # Number of overlaps matches number of determinants in pspace
#     assert len(overlaps) == len(tanh_rbm.pspace)

#     # Each overlap is a scalar
#     assert np.all([np.isscalar(o) or np.ndim(o) == 0 for o in overlaps])


# def test_normalization(tanh_rbm):
#     # Call normalize() to rescale the wavefunction
#     tanh_rbm.get_overlaps(normalized=False)
#     tanh_rbm.normalize()
#     overlaps = [entry["overlap"] * tanh_rbm.output_scale for entry in tanh_rbm._overlap_cache.values()]
#     squared_norm = np.sum(np.abs(overlaps) ** 2)
#     assert np.isclose(squared_norm, 1.0, rtol=1e-10, atol=1e-12)


# def test_safe_log_abs_tanh_and_derivative(tanh_rbm):
#     # Test small and large gamma values
#     gammas = [0.0, 1e-8, -1e-8, 1.0, -1.0, 10.0, -10.0]
#     for g in gammas:
#         log_val = tanh_rbm.safe_log_abs_tanh(g)
#         deriv_val = tanh_rbm.safe_dlogtanh_over_dgamma(g)

#         # log|tanh| should be finite
#         assert np.isfinite(log_val)
        
#         # derivative should be finite
#         assert np.isfinite(deriv_val)

# def test_spin(tanh_rbm):
#     assert tanh_rbm.spin == 0

# def test_overlaps_with_random_params(tanh_rbm):
#     # Assign random parameters and check overlaps remain finite
#     rng = np.random.default_rng(12345)
#     a = rng.uniform(-0.1, 0.1, size=tanh_rbm.nspin)
#     b = rng.uniform(-0.1, 0.1, size=tanh_rbm.nhidden)
#     w = rng.uniform(-0.1, 0.1, size=(tanh_rbm.nspin, tanh_rbm.nhidden))
#     tanh_rbm.assign_params([a, b, w])
#     tanh_rbm.get_overlaps(normalized=False)
#     overlaps = [entry["overlap"] for entry in tanh_rbm._overlap_cache.values()]
#     assert np.all(np.isfinite(overlaps))


# def test_tanh_rbm_overlap_and_derivative(tanh_rbm):
#     # Pick first Slater determinant
#     sd = tanh_rbm.pspace[0]

#     # --- Test overlap is finite ---
#     overlap = tanh_rbm.get_overlap(sd)
#     assert np.isfinite(overlap), "Overlap is not finite"

#     # --- Test derivative consistency using finite differences ---
#     eps = 1e-6
#     analytic = tanh_rbm.get_overlap(sd, deriv=np.arange(tanh_rbm.nparams), normalized=False)
#     numeric = []

#     base_params = tanh_rbm.params.copy()
#     for i in range(tanh_rbm.nparams):
#         shift = np.zeros_like(base_params)
#         shift[i] = eps

#         tanh_rbm.assign_params(base_params + shift)
#         plus = tanh_rbm.get_overlap(sd, normalized=False)

#         tanh_rbm.assign_params(base_params - shift)
#         minus = tanh_rbm.get_overlap(sd, normalized=False)

#         numeric.append((plus - minus) / (2 * eps))

#     tanh_rbm.assign_params(base_params)  # restore original
#     numeric = np.array(numeric)

#     assert np.allclose(analytic, numeric, rtol=1e-4, atol=1e-6), \
#         "Analytic and numeric derivatives do not match"



