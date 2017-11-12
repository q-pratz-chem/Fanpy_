"""Test wfns.wavefunction.composite.lin_comb."""
from nose.tools import assert_raises
import numpy as np
import types
from wfns.wfn.base import BaseWavefunction
from wfns.wfn.composite.lin_comb import LinearCombinationWavefunction


class Container:
    pass


class TestWavefunction(BaseWavefunction):
    """Base wavefunction that bypasses abstract class structure."""
    _spin = None
    _seniority = None

    def get_overlap(self):
        pass

    @property
    def spin(self):
        return self._spin

    @property
    def seniority(self):
        return self._seniority

    @property
    def template_params(self):
        return np.identity(10)


def test_assign_wfns():
    """Test LinearCombinationWavefunction.assign_wfns."""
    test_wfn = TestWavefunction(4, 10)
    test = Container()
    assert_raises(TypeError, LinearCombinationWavefunction.assign_wfns, test, (1, test_wfn))
    assert_raises(TypeError, LinearCombinationWavefunction.assign_wfns, test, (test_wfn, 2))
    test.nelec = 4
    assert_raises(ValueError, LinearCombinationWavefunction.assign_wfns, test,
                  (test_wfn, TestWavefunction(5, 10)))
    test.dtype = np.float64
    assert_raises(ValueError, LinearCombinationWavefunction.assign_wfns, test,
                  (test_wfn, TestWavefunction(4, 10, dtype=complex)))
    test.memory = np.inf
    assert_raises(ValueError, LinearCombinationWavefunction.assign_wfns, test,
                  (test_wfn, TestWavefunction(4, 10, memory='2gb')))
    assert_raises(ValueError, LinearCombinationWavefunction.assign_wfns, test, (test_wfn, ))
    # NOTE: wavefunctions with different numbers of spin orbitals are allowed
    LinearCombinationWavefunction.assign_wfns(test, (test_wfn, TestWavefunction(4, 12)))
    assert test.wfns[0].nelec == 4
    assert test.wfns[0].nspin == 10
    assert test.wfns[1].nelec == 4
    assert test.wfns[1].nspin == 12


def test_spin():
    """Test LinearCombinationWavefunction.spin."""
    test_wfn = TestWavefunction(4, 10)
    test_wfn._spin = 3
    test = LinearCombinationWavefunction(4, 10, (test_wfn, )*3)
    assert test.spin == 3
    test = LinearCombinationWavefunction(4, 10, (test_wfn, TestWavefunction(4, 10)))
    assert test.spin is None


def test_seniority():
    """Test LinearCombinationWavefunction.seniority."""
    test_wfn = TestWavefunction(4, 10)
    test_wfn._seniority = 3
    test = LinearCombinationWavefunction(4, 10, (test_wfn, )*3)
    assert test.seniority == 3
    test = LinearCombinationWavefunction(4, 10, (test_wfn, TestWavefunction(4, 10)))
    assert test.seniority is None


def test_template_params():
    """Test LinearCombinationWavefunction.template_params."""
    test_wfn = TestWavefunction(4, 10)
    test = LinearCombinationWavefunction(4, 10, (test_wfn, )*3)
    assert np.allclose(test.template_params, np.array([1, 0, 0]))
    assert np.allclose(test.template_params, np.array([1, 0, 0]))
    test = LinearCombinationWavefunction(4, 10, (test_wfn, TestWavefunction(4, 10)))
    assert np.allclose(test.template_params, np.array([1, 0]))


# TODO: test deriv functionality
def test_get_overlap():
    """Test LinearCombinationWavefunction.get_overlap.

    Make simple CI wavefunction as a demonstration.

    """
    test_wfn_1 = TestWavefunction(4, 10)
    test_wfn_1.params = np.array([0.9])
    test_wfn_2 = TestWavefunction(4, 10)
    test_wfn_2.params = np.array([0.1])

    def olp_one(self, sd, deriv=None):
        if sd == 0b0101:
            return self.params[0]
        else:
            return 0.0

    def olp_two(self, sd, deriv=None):
        if sd == 0b1010:
            return self.params[0]
        else:
            return 0.0

    test_wfn_1.get_overlap = types.MethodType(olp_one, test_wfn_1)
    test_wfn_2.get_overlap = types.MethodType(olp_two, test_wfn_2)

    test = LinearCombinationWavefunction(4, 10, (test_wfn_1, test_wfn_2),
                                         params=np.array([0.7, 0.3]))
    assert test.get_overlap(0b0101) == 0.7 * test_wfn_1.params[0] + 0.3 * 0
    assert test.get_overlap(0b1010) == 0.7 * 0 + 0.3 * test_wfn_2.params[0]
